import gc
import re
import time
import uuid
from typing import (
    List,
    Union,
    Dict,
    Any,
    Iterator,
)

import torch
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.generation.logits_process import LogitsProcessor

from api.generation.utils import apply_stopping_strings
from api.utils.protocol import Role


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(response: str) -> str:
    """
    Process the response by stripping leading and trailing whitespace,
    replacing the placeholder for training time, and normalizing punctuation.

    Args:
        response: The input response string.

    Returns:
        The processed response string.
    """
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response


def check_is_chatglm(model) -> bool:
    """
    Checks if the given model is a ChatGLM model.

    Args:
        model: The model to be checked.

    Returns:
        bool: True if the model is a ChatGLM model, False otherwise.
    """
    return "GLMBlock" in getattr(model, "_no_split_modules", [])


@torch.inference_mode()
def generate_stream_chatglm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: Dict[str, Any],
) -> Iterator:
    """
    Generates text in a streaming manner using the ChatGLM model.

    Args:
        model: The pre-trained ChatGLM model.
        tokenizer: The tokenizer used for tokenizing the input.
        params: A dictionary containing the input parameters.

    Yields:
        A dictionary representing each generated text completion.

    """
    inputs = params["inputs"]
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    input_echo_len = len(inputs["input_ids"][0])
    if input_echo_len >= model.config.seq_length:
        logger.warning(f"Input length larger than {model.config.seq_length}")

    inputs = {k: v[:, -model.config.seq_length:].to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_length": min(max_new_tokens + input_echo_len, model.config.seq_length),
        "do_sample": temperature > 1e-5,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len, previous_text = 0, ""
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    for total_ids in model.stream_generate(**inputs, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)

        output_ids = total_ids if echo else total_ids[input_echo_len:]
        response = tokenizer.decode(output_ids)
        response = process_response(response)

        delta_text = response[len(previous_text):]
        previous_text = response

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "delta": delta_text,
            "text": response,
            "logprobs": None,
            "finish_reason": None,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
        }

    # Only last stream result contains finish_reason, we set finish_reason as stop
    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "delta": "",
        "text": response,
        "logprobs": None,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
    }

    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_chatglm_v3(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: Dict[str, Any],
) -> Iterator:
    """
    Generates text in a streaming manner using the ChatGLM model.

    Args:
        model: The pre-trained ChatGLM model.
        tokenizer: The tokenizer used for tokenizing the input.
        params: A dictionary containing the input parameters.

    Yields:
        A dictionary representing each generated text completion.

    """
    inputs = params["inputs"]
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    input_echo_len = len(inputs["input_ids"][0])
    if input_echo_len >= model.config.seq_length:
        logger.warning(f"Input length larger than {model.config.seq_length}")

    inputs = {k: v[:, -model.config.seq_length:].to(model.device) for k, v in inputs.items()}

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
    ]

    gen_kwargs = {
        "max_length": min(max_new_tokens + input_echo_len, model.config.seq_length),
        "do_sample": temperature > 1e-5,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len, previous_text = 0, ""
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        
        output_ids = total_ids[:-1] if echo else total_ids[input_echo_len:-1]
        response = tokenizer.decode(output_ids)
        if response and response[-1] != "�":
            response, stop_found = apply_stopping_strings(response, ["<|observation|>"])

            delta_text = response[len(previous_text):]
            previous_text = response

            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "delta": delta_text,
                "text": response,
                "logprobs": None,
                "finish_reason": "function_call" if stop_found else None,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
            }

            if stop_found:
                break

    # Only last stream result contains finish_reason, we set finish_reason as stop
    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "delta": "",
        "text": response,
        "logprobs": None,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
    }

    gc.collect()
    torch.cuda.empty_cache()


def process_chatglm_messages(
    messages: List[ChatCompletionMessageParam],
    functions: Union[dict, List[dict]] = None,
) -> List[dict]:
    """
    Processes a list of chat messages and returns a modified list of messages.

    Args:
        messages: A list of chat messages to be processed.
        functions: Optional. A dictionary or list of dictionaries representing the available tools.

    Returns:
        A modified list of chat messages.
    """
    _messages = messages
    messages = []

    if functions:
        messages.append(
            {
                "role": Role.SYSTEM,
                "content": "Answer the following questions as best as you can. You have access to the following tools:",
                "tools": functions
            }
        )

    for m in _messages:
        role, content = m["role"], m["content"]
        if role == Role.FUNCTION:
            messages.append({"role": "observation", "content": content})
        elif role == Role.ASSISTANT:
            for response in content.split("<|assistant|>"):
                if "\n" in response:
                    metadata, sub_content = response.split("\n", maxsplit=1)
                else:
                    metadata, sub_content = "", response
                messages.append({"role": role, "metadata": metadata, "content": sub_content.strip()})
        else:
            messages.append({"role": role, "content": content})
    return messages
