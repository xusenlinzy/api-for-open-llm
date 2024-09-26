from __future__ import annotations

import gc
import time
import uuid
from typing import (
    Any,
    Dict,
    List,
    Iterator,
    TYPE_CHECKING,
)

import torch

from api.protocol import ChatCompletionMessageParam

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedModel


import queue
from threading import Thread
import torchvision.transforms as T
import transformers
from torchvision.transforms.functional import InterpolationMode
from transformers import BitsAndBytesConfig, TextIteratorStreamer

transformers.logging.set_verbosity_error()

# THUDM/cogvlm2-llama3-chat-19B
# THUDM/cogvlm2-llama3-chinese-chat-19B

@torch.inference_mode()
def generate_stream_cogvlm2(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    params: Dict[str, Any],
) -> Iterator:
    """
    Generates text in a streaming manner using the ChatGLM model.

    Args:
        model: The pre-trained model.
        tokenizer: The tokenizer used for tokenizing the input.
        params: A dictionary containing the input parameters.

    Yields:
        A dictionary representing each generated text completion.

    """
    inputs = params["inputs"]
    model_name = params.get("model", "llm")

    query, history, images, system_message = prompt_history_images_system_from_messages(inputs, img_tok='')

    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=images, template_version='chat')

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(model.device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(model.device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(model.device),
        'images': [[input_by_model['images'][0].to(model.device).to(model.dtype)]] if images else None,
    }

    new_params = dict(temperature = float(params.get("temperature", 1.0)),
                      max_new_tokens = int(params.get("max_tokens", 256)),
                      repetition_penalty = float(params.get("repetition_penalty", 1.0)),
                      top_p = float(params.get("top_p", 1.0)),
                      top_k = int(params.get("top_k", 50)))

    generation_kwargs = dict(
        **inputs,
        **new_params,
    )

    input_echo_len = 0
    generated_text, previous_text = "", ""
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    for i, new_text in enumerate(threaded_streaming_generator(generate=model.generate, tokenizer=tokenizer, generation_kwargs=generation_kwargs)):
        end = new_text.find(tokenizer.eos_token)
        if end != -1:
            new_text = new_text[:end]

        generated_text += new_text
        delta_text = generated_text[len(previous_text):]
        previous_text = generated_text
        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "delta": delta_text,
            "text": generated_text,
            "logprobs": None,
            "finish_reason": None,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
        }

        if end != -1:
            break

    gc.collect()
    torch.cuda.empty_cache()

def prompt_history_images_system_from_messages(messages: list[ChatCompletionMessageParam], img_tok = "<image>\n"):
    history = []
    images = []
    prompt = ''
    system_prompt = None

    for m in messages:
        if m['role'] == 'user':
            p = ''
            for c in m['content']:
                if c['type'] == 'image_url':
                    image = url_to_image(c['image_url']['url'])
                    images.extend([image])
                    p = img_tok + p
                if c['type'] == 'text':
                    p += c['text']

            prompt += p
        elif m['role'] == 'assistant':
            for c in m['content']:
                if c['type'] == 'text':
                    history.extend([(prompt, c['text'])])
                    prompt = ''
        elif m['role'] == 'system':
            for c in m['content']:
                if c['type'] == 'text':
                    system_prompt = c['text']

    return prompt, history, images, system_prompt


def url_to_image(image_url: str):
    from PIL import Image
    from io import BytesIO

    if image_url.startswith("data:"):
        import base64

        image_bytes = base64.b64decode(image_url.split(",")[1])
    else:
        import urllib.request

        with urllib.request.urlopen(image_url) as f:
            image_bytes = f.read()

    return Image.open(BytesIO(image_bytes)).convert("RGB")


def threaded_streaming_generator(generate, tokenizer, generation_kwargs):
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True, timeout=60)

    generation_kwargs['streamer'] = streamer

    exq = queue.Queue()

    def wrapper():
        try:
            with torch.no_grad():
                generate(**generation_kwargs)

        except Exception as e:
            #logger.exception(e)
            exq.put(e)
            streamer.end()

    t = Thread(target=wrapper, daemon=True)
    t.start()

    for text in streamer:
        if text:
            yield text

    if not exq.empty():
        raise exq.get_nowait()
