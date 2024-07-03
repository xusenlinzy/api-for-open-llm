from __future__ import annotations

from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    TYPE_CHECKING,
    Tuple,
)

from openai.types.chat import ChatCompletionMessageParam

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, BatchEncoding


class ChatTemplate(ABC):
    """Base class for chat template"""

    system_prompt: Optional[str] = ""
    stop: Sequence[str] = []
    stop_token_ids: Sequence[int] = []
    function_call_available: Optional[bool] = False

    def __init__(
        self,
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        model_max_length: Optional[int] = 8192,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

    def convert_messages_to_ids(
        self,
        messages: List[ChatCompletionMessageParam],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 256,
        max_window_size: Optional[int] = 6144,
        **kwargs,
    ) -> Union[List[int], "BatchEncoding"]:
        try:
            token_ids = self._convert_messages_to_ids(
                messages,
                system,
                tools,
                max_tokens,
                max_window_size,
                **kwargs,
            )
        except NotImplementedError:
            token_ids = self.apply_chat_template(
                messages,
                system,
                tokenize=True,
                **kwargs,
            )

        input_len = len(token_ids)
        min_max_tokens = 256
        if input_len > self.model_max_length - min_max_tokens:
            max_input_tokens = self.model_max_length - min_max_tokens
        else:
            max_input_tokens = max(self.model_max_length - max_tokens, input_len)

        return token_ids[-max_input_tokens:]

    def _convert_messages_to_ids(
        self,
        messages: List[ChatCompletionMessageParam],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 256,
        max_window_size: Optional[int] = 6144,
        **kwargs,
    ) -> Union[List[int], "BatchEncoding"]:
        raise NotImplementedError

    def apply_chat_template(
        self,
        messages: List[ChatCompletionMessageParam],
        system: Optional[str] = None,
        **kwargs,
    ) -> Union[str, List[ChatCompletionMessageParam]]:
        system_prompt = system or self.system_prompt
        return self.tokenizer.apply_chat_template(
            messages,
            chat_template=self.chat_template,
            add_generation_prompt=True,
            system_prompt=system_prompt,
            **kwargs,
        )

    def parse_assistant_response(
        self,
        output: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[Union[str, Dict[str, Any]]]]:
        return output, None

    @property
    def chat_template(self) -> str:
        return self.tokenizer.chat_template or self.tokenizer.default_chat_template
