import asyncio
from typing import (
    Optional,
    List,
    Dict,
    Any,
    AsyncIterator,
    Union,
)

from fastapi import HTTPException
from openai.types.chat import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

from api.apapter import get_prompt_adapter
from api.generation.chatglm import process_chatglm_messages


class VllmEngine:
    def __init__(
        self,
        model: AsyncLLMEngine,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        prompt_name: Optional[str] = None,
        context_len: Optional[int] = -1,
    ):
        self.model = model
        self.model_name = model_name.lower()
        self.tokenizer = tokenizer
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)

        model_config = asyncio.run(self.model.get_model_config())
        if "qwen" in self.model_name:
            self.max_model_len = context_len if context_len > 0 else 8192
        else:
            self.max_model_len = model_config.max_model_len

    def apply_chat_template(
        self, messages: List[ChatCompletionMessageParam], **kwargs,
    ) -> Union[str, List[int]]:
        if "chatglm3" in self.model_name:
            messages = process_chatglm_messages(messages, functions=kwargs.get("functions", None))
            query, role = messages[-1]["content"], messages[-1]["role"]
            return self.tokenizer.build_chat_input(query, history=messages[:-1], role=role)["input_ids"][0].tolist()
        return self.prompt_adapter.apply_chat_template(messages)

    def convert_to_inputs(
        self,
        prompt: Optional[str] = None,
        token_ids: Optional[List[int]] = None,
        max_tokens: Optional[int] = 256,
    ) -> List[int]:
        max_input_tokens = self.max_model_len - max_tokens
        input_ids = token_ids if token_ids else self.tokenizer(prompt).input_ids
        return input_ids[-max_input_tokens:]

    def generate(self, params: Dict[str, Any], request_id: str) -> AsyncIterator:
        prompt_or_messages = params.get("prompt_or_messages")
        if isinstance(prompt_or_messages, list):
            prompt_or_messages = self.apply_chat_template(prompt_or_messages, functions=params.get("functions", None))

        if isinstance(prompt_or_messages, list):
            prompt, token_ids = None, prompt_or_messages
        else:
            prompt, token_ids = prompt_or_messages, None

        token_ids = self.convert_to_inputs(prompt, token_ids, max_tokens=params.get("max_tokens", 256))
        try:
            sampling_params = SamplingParams(
                n=params.get("n", 1),
                presence_penalty=params.get("presence_penalty", 0.),
                frequency_penalty=params.get("frequency_penalty", 0.),
                temperature=params.get("temperature", 0.9),
                top_p=params.get("top_p", 0.8),
                stop=params.get("stop", []),
                stop_token_ids=params.get("stop_token_ids", []),
                max_tokens=params.get("max_tokens", 256),
            )
            result_generator = self.model.generate(
                prompt_or_messages if isinstance(prompt_or_messages, str) else None,
                sampling_params,
                request_id,
                token_ids,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return result_generator

    @property
    def stop(self):
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
