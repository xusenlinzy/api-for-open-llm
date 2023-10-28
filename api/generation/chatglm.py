import gc
import re
from copy import deepcopy
from typing import List

import torch
from transformers.generation.logits_process import LogitsProcessor

from api.utils.protocol import Role, ChatMessage


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(response):
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


def process_response_v3(output, history):
    content = ""
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            history.append(
                {

                    "role": Role.ASSISTANT,
                    "metadata": metadata,
                    "content": content
                }
            )
            content = content.replace("[[训练时间]]", "2023年")
        else:
            history.append(
                {
                    "role": Role.ASSISTANT,
                    "metadata": metadata,
                    "content": content
                }
            )
            if history[0]["role"] == Role.SYSTEM and "tools" in history[0]:
                content = "\n".join(content.split("\n")[1:-1])

                def tool_call(**kwargs):
                    return kwargs

                parameters = eval(content)
                content = {
                    "name": metadata.strip(),
                    "parameters": parameters
                }
            else:
                content = {
                    "name": metadata.strip(),
                    "content": content
                }
    return content, history


def check_is_chatglm(model) -> bool:
    return "GLMBlock" in getattr(model, "_no_split_modules", [])


@torch.inference_mode()
def generate_stream_chatglm(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    gen_kwargs = {
        "max_length": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    for total_ids in model.stream_generate(**inputs, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids
        else:
            output_ids = total_ids[input_echo_len:]
        response = tokenizer.decode(output_ids)
        response = process_response(response)

        yield {
            "text": response,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": None,
        }

    # TODO: ChatGLM stop when it reaches max length
    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_chatglm_v3(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
):
    prompt: List[ChatMessage] = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    query, role = prompt[-1].content, prompt[-1].role
    history = [m.dict(exclude_none=True) for m in prompt[:-1]]

    inputs = tokenizer.build_chat_input(query, history=history, role=role)
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
        tokenizer.get_command("<|observation|>")
    ]

    gen_kwargs = {
        "max_length": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    history.append(
        {
            "role": role,
            "content": query
        }
    )

    total_len = 0
    for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids[:-1]
        else:
            output_ids = total_ids[input_echo_len:-1]

        response = tokenizer.decode(output_ids)

        yield {
            "text": response,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": None,
        }

    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    gc.collect()
    torch.cuda.empty_cache()
