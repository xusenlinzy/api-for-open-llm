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


@torch.inference_mode()
def generate_stream_minicpm_v(
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
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 50))

    messages = process_minicpmv_messages(inputs)
    streamer = model.chat(
        image=None,
        msgs=messages,
        tokenizer=tokenizer,
        sampling=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stream=True,
    )

    # Todo: fix length for prompt
    input_echo_len = 0

    generated_text, previous_text = "", ""
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    for i, new_text in enumerate(streamer):
        generated_text += new_text
        if generated_text:
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

    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "delta": "",
        "text": generated_text,
        "logprobs": None,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
    }

    gc.collect()
    torch.cuda.empty_cache()


def process_minicpmv_messages(messages: List[ChatCompletionMessageParam]) -> List[Dict]:
    _messages = []
    for message in messages:
        if isinstance(message["content"], str):
            _content = [message["content"]]
        else:
            _content = []
            for c in message["content"]:
                if isinstance(c, dict) and "type" in c:
                    if c["type"] == "text":
                        _content.append(c["text"])
                    elif c["type"] == "image_url":
                        if (
                            isinstance(c["image_url"], dict)
                            and "url" in c["image_url"]
                        ):
                            image = load_image(image_url=c["image_url"]["url"])
                        else:
                            image = load_image(image_url=c["image_url"])
                        _content.insert(0, image)
        _messages.append({"role": message["role"], "content": _content})
    return _messages


def load_image(image_url: str):
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
