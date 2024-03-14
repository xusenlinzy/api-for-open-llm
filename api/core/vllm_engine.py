import asyncio
import time
from dataclasses import dataclass
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Union,
)

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from openai.types.completion_choice import Logprobs
from openai.types.model import Model
from pydantic import BaseModel
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.transformers_utils.tokenizer import get_tokenizer

from api.adapter import get_prompt_adapter
from api.generation import build_qwen_chat_input


@dataclass
class LoRA:
    name: str
    local_path: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model] = []


class VllmEngine:
    def __init__(
        self,
        model: AsyncLLMEngine,
        model_name: str,
        prompt_name: Optional[str] = None,
        lora_modules: Optional[List[LoRA]] = None,
    ):
        """
        Initializes the VLLMEngine object.

        Args:
            model: The AsyncLLMEngine object.
            model_name: The name of the model.
            prompt_name: The name of the prompt (optional).
        """
        self.model = model
        self.model_name = model_name.lower()
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)

        if lora_modules is None:
            self.lora_requests = []
        else:
            try:
                from vllm.lora.request import LoRARequest
                self.lora_requests = [
                    LoRARequest(
                        lora_name=lora.name,
                        lora_int_id=i,
                        lora_local_path=lora.local_path,
                    ) for i, lora in enumerate(lora_modules, start=1)
                ]
            except ImportError:
                self.lora_requests = []

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running():
            # If the current is instanced by Ray Serve,
            # there is already a running event loop
            event_loop.create_task(self._post_init())
        else:
            # When using single vLLM without engine_use_ray
            asyncio.run(self._post_init())

    async def _post_init(self):
        engine_model_config = await self.model.get_model_config()
        self.max_model_len = engine_model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            engine_model_config.tokenizer,
            tokenizer_mode=engine_model_config.tokenizer_mode,
            trust_remote_code=engine_model_config.trust_remote_code,
        )

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            Model(
                id=self.model_name,
                object="model",
                created=int(time.time()),
                owned_by="vllm"
            )
        ]
        lora_cards = [
            Model(
                id=lora.lora_name,
                object="model",
                created=int(time.time()),
                owned_by="vllm"
            )
            for lora in self.lora_requests
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)

    def create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: Optional[List[Optional[Any]]] = None,
        num_output_top_logprobs: Optional[int] = None,
        initial_text_offset: int = 0,
    ):
        """Create OpenAI-style logprobs."""
        logprobs = Logprobs()
        last_token_len = 0
        if num_output_top_logprobs:
            logprobs.top_logprobs = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is not None:
                token_logprob = step_top_logprobs[token_id].logprob
            else:
                token_logprob = None

            token = step_top_logprobs[token_id].decoded_token
            logprobs.tokens.append(token)
            logprobs.token_logprobs.append(token_logprob)

            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)
            else:
                logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
            last_token_len = len(token)

            if num_output_top_logprobs:
                logprobs.top_logprobs.append(
                    {
                        p.decoded_token: p.logprob
                        for i, p in step_top_logprobs.items()
                    }
                    if step_top_logprobs else None
                )
        return logprobs

    def _maybe_get_lora(self, model_name):
        for lora in self.lora_requests:
            if model_name == lora.lora_name:
                logger.info(f"Lora request: {model_name}")
                return lora
        return None

    def apply_chat_template(
        self,
        messages: List[ChatCompletionMessageParam],
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, List[int]]:
        """
        Applies a chat template to the given messages and returns the processed output.

        Args:
            messages: A list of ChatCompletionMessageParam objects representing the chat messages.
            functions: A dictionary or list of dictionaries representing the functions to be applied (optional).
            tools: A list of dictionaries representing the tools to be used (optional).

        Returns:
            Union[str, List[int]]: The processed output as a string or a list of integers.
        """
        if self.prompt_adapter.function_call_available:
            messages = self.prompt_adapter.postprocess_messages(
                messages, functions, tools,
            )
            if functions or tools:
                logger.debug(f"==== Messages with tools ====\n{messages}")

        if "chatglm3" in self.model_name:
            query, role = messages[-1]["content"], messages[-1]["role"]
            return self.tokenizer.build_chat_input(
                query, history=messages[:-1], role=role
            )["input_ids"][0].tolist()
        elif self.model_name.startswith("qwen") and ("qwen1.5" not in self.model_name) and ("qwen2" not in self.model_name):
            return build_qwen_chat_input(
                self.tokenizer,
                messages,
                functions=functions,
                tools=tools,
            )
        else:
            return self.prompt_adapter.apply_chat_template(messages)

    def convert_to_inputs(
        self,
        prompt: Optional[str] = None,
        token_ids: Optional[List[int]] = None,
        max_tokens: Optional[int] = 256,
    ) -> List[int]:
        input_ids = token_ids or self.tokenizer(prompt).input_ids
        input_len = len(input_ids)
        min_max_tokens = 256
        if input_len > self.max_model_len - min_max_tokens:
            max_input_tokens = self.max_model_len - min_max_tokens
        else:
            max_input_tokens = max(self.max_model_len - max_tokens, input_len)
        return input_ids[-max_input_tokens:]

    @property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            The stop property of the prompt adapter, or None if it does not exist.
        """
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
