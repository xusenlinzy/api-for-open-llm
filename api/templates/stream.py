from __future__ import annotations

import gc
import time
import uuid
from threading import Thread
from types import MethodType
from typing import (
    Dict,
    Any,
    TYPE_CHECKING,
)

import torch
from transformers import TextIteratorStreamer

from api.templates.utils import apply_stopping_strings

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedModel


@torch.inference_mode()
def generate_stream(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    params: Dict[str, Any],
):
    inputs = params.get("inputs")
    functions = params.get("functions")
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 50))
    max_new_tokens = int(params.get("max_tokens", 256))

    stop_token_ids = params.get("stop_token_ids") or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    stop_strings = params.get("stop", [])

    device = next(model.parameters()).device
    generation_kwargs = dict(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature <= 1e-5:
        generation_kwargs["do_sample"] = False
        generation_kwargs.pop("top_k")

    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        generation_kwargs.update(inputs)
        input_echo_len = len(inputs["input_ids"][0])
    else:
        generation_kwargs["input_ids"] = torch.tensor([inputs], device=device)
        input_echo_len = len(inputs)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs["streamer"] = streamer

    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text, func_call_found = "", False
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    previous_text = ""
    for i, new_text in enumerate(streamer):
        generated_text += new_text
        if functions:
            _, func_call_found = apply_stopping_strings(generated_text, ["Observation:"])
        generated_text, stop_found = apply_stopping_strings(generated_text, stop_strings)

        if generated_text and generated_text[-1] != "�":
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
                "finish_reason": "function_call" if func_call_found else None,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
            }

        if stop_found:
            break

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
