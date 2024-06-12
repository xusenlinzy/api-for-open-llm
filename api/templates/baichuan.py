from __future__ import annotations

from typing import (
    List,
    TYPE_CHECKING,
    Optional,
    Dict,
    Any,
    Union,
)

from openai.types.chat import ChatCompletionMessageParam

from api.protocol import Role
from api.templates.base import ChatTemplate
from api.templates.registry import register_template
from api.templates.utils import parse_messages

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, BatchEncoding


def build_baichuan_chat_input(
    tokenizer: "PreTrainedTokenizer",
    messages: List[ChatCompletionMessageParam],
    context_len: int = 4096,
    max_new_tokens: int = 256
) -> List[int]:
    """
    Builds the input tokens for the Baichuan chat model based on the given messages.

    Refs:
        https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_utils.py 

    Args:
        tokenizer: The PreTrainedTokenizer object.
        messages: A list of ChatCompletionMessageParam objects representing the chat messages.
        context_len: The maximum length of the context (default=4096).
        max_new_tokens: The maximum number of new tokens to be added (default=256).

    Returns:
        List[int]: The input tokens for the Baichuan chat model.
    """
    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for r in rounds[::-1]:
        round_tokens = []
        for message in r:
            if message["role"] == Role.USER.value:
                round_tokens.append(195)
            else:
                round_tokens.append(196)
            round_tokens.extend(tokenizer.encode(message["content"]))

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != Role.ASSISTANT.value:
        input_tokens.append(196)

    return input_tokens[-max_input_tokens:]  # truncate left


@register_template("baichuan")
class BaiChuanChatTemplate(ChatTemplate):
    stop_token_ids = [195, 196]
    stop = ["<reserved_102>", "<reserved_103>"]

    def _convert_messages_to_ids(
        self,
        messages: List[ChatCompletionMessageParam],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 256,
        max_window_size: Optional[int] = 6144,
        **kwargs,
    ) -> Union[List[int], "BatchEncoding"]:
        return build_baichuan_chat_input(
            self.tokenizer,
            messages,
            self.model_max_length,
            max_tokens,
        )

    @property
    def chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}"
            "{% else %}"
            "{{ system_prompt }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<reserved_102>' + message['content'] + '<reserved_103>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("baichuan2")
class BaiChuan2ChatTemplate(BaiChuanChatTemplate):
    stop = ["<reserved_106>", "<reserved_107>"]

    @property
    def chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}"
            "{% else %}"
            "{{ system_prompt }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<reserved_106>' + message['content'] + '<reserved_107>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
        )
