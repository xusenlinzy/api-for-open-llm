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

# mx262/MiniMonkey

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def dynamic_preprocess2(image, min_num=1, max_num=12, prior_aspect_ratio=None, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    new_target_ratios = []
    for i in target_ratios:
        if prior_aspect_ratio[0]%i[0] or prior_aspect_ratio[1]%i[1]:
            new_target_ratios.append(i)
        else:
            continue
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, new_target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, min_num=1, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_aspect_ratio

def load_image2(image, input_size=448, min_num=1, max_num=12, target_aspect_ratio=None):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess2(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num, prior_aspect_ratio=target_aspect_ratio)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


@torch.inference_mode()
def generate_stream_minimonkey(
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

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    images, prompt = chatml_prompt_from_messages(inputs)

    # set the max number of tiles in `max_num`, XXX make an option
    pixel_values, target_aspect_ratio = load_image(images[-1], min_num=4, max_num=12)
    pixel_values2 = load_image2(images[-1], min_num=3, max_num=7, target_aspect_ratio=target_aspect_ratio)
    pixel_values = torch.cat([pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0).to(device=model.device, dtype=model.dtype)

    for num_patches in [pixel_values.shape[0]]:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        prompt = prompt.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)

    inputs = dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        target_aspect_ratio=target_aspect_ratio,
    )

    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    new_params = dict(eos_token_id=[eos_token_id, tokenizer.eos_token_id],
                      temperature = float(params.get("temperature", 1.0)),
                      max_new_tokens = int(params.get("max_tokens", 256)),
                      repetition_penalty = float(params.get("repetition_penalty", 1.0)),
                      top_p = float(params.get("top_p", 1.0)),
                      top_k = int(params.get("top_k", 50)))

    generation_kwargs = dict(
        **inputs,
        **new_params,
    )

    # Todo: fix length for prompt
    input_echo_len = 0

    generated_text, previous_text = "", ""
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    for i, new_text in enumerate(threaded_streaming_generator(generate=model.generate, tokenizer=tokenizer, generation_kwargs=generation_kwargs)):
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

    gc.collect()
    torch.cuda.empty_cache()


def chatml_prompt_from_messages(messages: list[ChatCompletionMessageParam], img_tok = "<image>\n"):
    prompt = ''
    images = []
    generation_msg = "<|im_start|>assistant\n"

    if messages and messages[-1]['role'] == 'assistant':
        generation_msg += messages[-1]['content'][0].text
        messages.pop(-1)

    for m in messages:
        if m['role'] == 'user':
            text = ''
            has_image = False

            for c in m['content']:
                if c['type'] == 'image_url':
                    images.extend([ url_to_image(c['image_url']['url']) ])
                    has_image = True
                if c['type'] == 'text':
                    text = c['text']

            img_tag = img_tok if has_image else ''
            prompt += f"<|im_start|>user\n{img_tag}{text}<|im_end|>"
        elif m['role'] == 'assistant':
            for c in m['content']:
                if c['type'] == 'text':
                    prompt += f"<|im_start|>assistant\n{c['text']}<|im_end|>"
        elif m['role'] == 'system':
            for c in m['content']:
                if c['type'] == 'text':
                    prompt += f"<|im_start|>system\n{c['text']}<|im_end|>"

    prompt += generation_msg

    return images, prompt


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