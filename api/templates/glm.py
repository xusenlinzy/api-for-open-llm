from __future__ import annotations

import gc
import json
import re
import time
import uuid
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Iterator,
    TYPE_CHECKING,
)

import torch
from loguru import logger
from transformers.generation.logits_process import LogitsProcessor

from api.protocol import ChatCompletionMessageParam, Role
from api.templates.base import ChatTemplate
from api.templates.registry import register_template
from api.templates.utils import apply_stopping_strings

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding


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


@torch.inference_mode()
def generate_stream_chatglm(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
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
    input_ids = params["inputs"]
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    input_echo_len = len(input_ids)
    if input_echo_len >= model.config.seq_length:
        logger.warning(f"Input length larger than {model.config.seq_length}")

    input_ids = torch.tensor([input_ids], device=model.device)

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
    for total_ids in model.stream_generate(input_ids, **gen_kwargs):
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
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
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
    input_ids = params["inputs"]
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    input_echo_len = len(input_ids)
    if input_echo_len >= model.config.seq_length:
        logger.warning(f"Input length larger than {model.config.seq_length}")

    input_ids = torch.tensor([input_ids], device=model.device)
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>")]

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
    for total_ids in model.stream_generate(input_ids, eos_token_id=eos_token_id, **gen_kwargs):
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
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[dict]:
    """
    Processes a list of chat messages and returns a modified list of messages.
    """
    _messages = messages
    messages = []

    if tools:
        messages.append(
            {
                "role": Role.SYSTEM.value,
                "content": "Answer the following questions as best as you can. You have access to the following tools:",
                "tools": [t["function"] for t in tools]
            }
        )

    for m in _messages:
        role, content = m["role"], m["content"]
        if role in [Role.FUNCTION.value, Role.TOOL.value]:
            messages.append(
                {
                    "role": "observation",
                    "content": content,
                }
            )
        elif role == Role.ASSISTANT.value:
            if content is not None:
                for response in content.split("<|assistant|>"):
                    if "\n" in response:
                        metadata, sub_content = response.split("\n", maxsplit=1)
                    else:
                        metadata, sub_content = "", response
                    messages.append(
                        {
                            "role": role,
                            "metadata": metadata,
                            "content": sub_content.strip()
                        }
                    )
        else:
            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )
    return messages


def process_chatglm_messages_v4(
    messages: List[ChatCompletionMessageParam],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[dict]:
    _messages = messages
    messages = []
    msg_has_sys = False

    if tools:
        messages.append(
            {
                "role": Role.SYSTEM.value,
                "content": None,
                "tools": tools
            }
        )

    for m in _messages:
        role, content, func_call = m["role"], m["content"], m.get("function_call")
        if role in [Role.FUNCTION.value, Role.TOOL.value]:
            messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == Role.ASSISTANT.value and func_call is not None:
            for response in content.split("<|assistant|>"):
                if "\n" in response:
                    metadata, sub_content = response.split("\n", maxsplit=1)
                else:
                    metadata, sub_content = "", response
                messages.append(
                    {
                        "role": role,
                        "metadata": metadata,
                        "content": sub_content.strip()
                    }
                )
        else:
            if role == Role.SYSTEM.value and msg_has_sys:
                msg_has_sys = False
                continue
            messages.append({"role": role, "content": content})

    return messages


@register_template("chatglm")
class ChatGLMChatTemplate(ChatTemplate):
    @property
    def chat_template(self) -> str:
        """The reference for this chat template is [this code
        snippet](https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py)
        in the original repository.
        """
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{% set idx = loop.index0 // 2 %}"
            "{{ '[Round ' ~ idx ~ ']\\n' + '问：' + message['content'] + '\\n' + '答：' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("chatglm2")
class ChatGLM2ChatTemplate(ChatTemplate):
    @property
    def chat_template(self) -> str:
        """The reference for this chat template is [this code
        snippet](https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py)
        in the original repository.
        """
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{% set idx = loop.index0 // 2 + 1 %}"
            "{{ '[Round ' ~ idx ~ ']\\n\\n' + '问：' + message['content'] + '\\n\\n' + '答：' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '\\n\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("chatglm3")
class ChatGLM3ChatTemplate(ChatTemplate):
    stop = ["<|user|>", "</s>", "<|observation|>"]
    stop_token_ids = [64795, 64797, 2]
    function_call_available = True

    def _convert_messages_to_ids(
        self,
        messages: List[ChatCompletionMessageParam],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 256,
        max_window_size: Optional[int] = 6144,
        **kwargs,
    ) -> Union[List[int], BatchEncoding]:
        messages = process_chatglm_messages(messages, tools)
        query, role = messages[-1]["content"], messages[-1]["role"]
        return self.tokenizer.build_chat_input(
            query, history=messages[:-1], role=role
        )["input_ids"][0].tolist()

    def parse_assistant_response(
        self,
        output: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[Union[str, Dict[str, Any]]]]:
        content = ""
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response

            if not metadata.strip():
                content = content.strip()
                content = content.replace("[[训练时间]]", "2023年")
            else:
                if tools:
                    content = "\n".join(content.split("\n")[1:-1])

                    def tool_call(**kwargs):
                        return kwargs

                    parameters = eval(content)
                    content = {
                        "function": {
                            "name": metadata.strip(),
                            "arguments": json.dumps(parameters, ensure_ascii=False)
                        },
                        "id": metadata.strip(),
                        "type": "function",
                    }
                else:
                    content = {
                        "name": metadata.strip(),
                        "content": content
                    }
        return output, content

    @property
    def chat_template(self) -> str:
        """The reference for this chat template is [this code
        snippet](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py)
        in the original repository.
        """
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|system|>\\n ' + message['content'] }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|user|>\\n ' + message['content'] + '<|assistant|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '\\n ' + message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("chatglm4")
class ChatGLM4ChatTemplate(ChatTemplate):
    stop = ["<|endoftext|>", "<user>", "<|observation|>"]
    stop_token_ids = [151329, 151336, 151338]

    def _convert_messages_to_ids(
        self,
        messages: List[ChatCompletionMessageParam],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 256,
        max_window_size: Optional[int] = 6144,
        **kwargs,
    ) -> Union[List[int], BatchEncoding]:
        messages = process_chatglm_messages_v4(messages, tools)
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

    def parse_assistant_response(
        self,
        output: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[Union[str, Dict[str, Any]]]]:
        lines = output.strip().split("\n")
        special_tools = ["cogview", "simple_browser"]
        tool_call_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

        content, arguments_json = None, None
        if len(lines) >= 2 and tool_call_pattern.match(lines[0]):
            function_name = lines[0].strip()
            arguments = "\n".join(lines[1:]).strip()

            try:
                arguments_json = json.loads(arguments)
                is_tool_call = True
            except json.JSONDecodeError:
                is_tool_call = function_name in special_tools

            if is_tool_call and tools:
                content = {
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(
                            arguments_json if isinstance(arguments_json, dict) else arguments,
                            ensure_ascii=False,
                        )
                    },
                    "id": function_name,
                    "type": "function",
                }
                if function_name == "simple_browser":
                    search_pattern = re.compile(r'search\("(.+?)"\s*,\s*recency_days\s*=\s*(\d+)\)')
                    match = search_pattern.match(arguments)
                    if match:
                        content["function"]["arguments"] = json.dumps(
                            {
                                "query": match.group(1),
                                "recency_days": int(match.group(2))
                            },
                            ensure_ascii=False,
                        )
                elif function_name == "cogview":
                    content["function"]["arguments"] = json.dumps({"prompt": arguments}, ensure_ascii=False)
            else:
                content = None

        return output, content


@register_template("glm-4v")
class GLM4VChatTemplate(ChatTemplate):
    stop = ["<|endoftext|>", "<user>", "<|observation|>"]
    stop_token_ids = [151329, 151336, 151338]

    def _convert_messages_to_ids(
        self,
        messages: List[ChatCompletionMessageParam],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 256,
        max_window_size: Optional[int] = 6144,
        **kwargs,
    ) -> Union[List[int], "BatchEncoding"]:
        _messages = []
        for message in messages:
            if isinstance(message["content"], str):
                _content, image = message["content"], None
            else:
                _content, image = None, None
                for c in message["content"]:
                    if isinstance(c, dict) and "type" in c:
                        if c["type"] == "text":
                            _content = c["text"]

                        if c["type"] == "image_url":
                            if (
                                isinstance(c["image_url"], dict)
                                and "url" in c["image_url"]
                            ):
                                image = self._load_image(image_url=c["image_url"]["url"])
                            else:
                                image = self._load_image(image_url=c["image_url"])

            msg = {"role": message["role"], "content": _content}
            if image is not None:
                msg["image"] = image
            _messages.append(msg)

        return self.tokenizer.apply_chat_template(
            _messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

    @staticmethod
    def _load_image(image_url: str):
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
