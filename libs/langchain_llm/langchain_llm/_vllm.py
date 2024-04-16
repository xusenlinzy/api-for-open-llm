from __future__ import annotations

import time
import traceback
import uuid
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from langchain_community.llms.vllm import VLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.pydantic_v1 import root_validator
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion import Completion
from openai.types.completion_choice import (
    CompletionChoice,
)
from openai.types.completion_usage import CompletionUsage

from ._compat import model_parse
from .adapters.template import (
    get_prompt_adapter,
    BaseTemplate,
)
from .generation import (
    build_qwen_chat_input,
)


class XVLLM(VLLM):
    """vllm model."""

    model_name: str
    """The name of a HuggingFace Transformers model."""

    def call_as_openai(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Completion:
        """Run the LLM on the given prompt and input."""

        from vllm import SamplingParams

        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}
        sampling_params = SamplingParams(**params)
        # call the model
        outputs = self.client.generate([prompt], sampling_params)[0]

        choices = []
        for output in outputs.outputs:
            text = output.text
            choices.append(
                CompletionChoice(
                    index=0,
                    text=text,
                    finish_reason="stop",
                    logprobs=None,
                )
            )

        num_prompt_tokens = len(outputs.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in outputs.outputs)
        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return Completion(
            id=f"cmpl-{str(uuid.uuid4())}",
            choices=choices,
            created=int(time.time()),
            model=self.model_name,
            object="text_completion",
            usage=usage,
        )


class ChatVLLM(BaseChatModel):
    """
    Wrapper for using VLLM as ChatModels.
    """

    llm: XVLLM

    chat_template: Optional[str] = None
    """Chat template for generating completions."""

    max_window_size: Optional[int] = 6144
    """The maximum window size"""

    construct_prompt: bool = True

    prompt_adapter: Optional[BaseTemplate] = None

    tokenizer: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        # Check whether to need to construct prompts or inputs. """
        model_name = values["llm"].model_name
        values["chat_template"] = values["chat_template"].lower() if values["chat_template"] is not None else None
        if not values["prompt_adapter"]:
            try:
                values["prompt_adapter"] = get_prompt_adapter(model_name, values["chat_template"])
            except KeyError:
                values["chat_template"] = None

        values["tokenizer"] = values["llm"].client.get_tokenizer()

        return values

    def _get_parameters(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Performs sanity check, preparing parameters.

        Args:
            stop (Optional[List[str]]): List of stop sequences.

        Returns:
            Dictionary containing the combined parameters.
        """

        params = self.llm._default_params

        # then sets it as configured, or default to an empty list:
        _stop, _stop_token_ids = [], []
        if isinstance(self.prompt_adapter.stop, dict):
            _stop_token_ids = self.prompt_adapter.stop.get("token_ids", [])
            _stop = self.prompt_adapter.stop.get("strings", [])

        stop = stop or []
        if isinstance(stop, str):
            stop = [stop]

        params["stop"] = list(set(_stop + stop))
        params["stop_token_ids"] = list(set(_stop_token_ids))

        params = {**params, **kwargs}

        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from vllm import SamplingParams

        llm_input = self._to_chat_prompt(messages)
        params = self._get_parameters(stop, **kwargs)

        # build sampling parameters
        sampling_params = SamplingParams(**params)
        # call the model

        if isinstance(llm_input, str):
            prompts, prompt_token_ids = [llm_input], None
        else:
            prompts, prompt_token_ids = None, [llm_input]

        outputs = self.llm.client.generate(prompts, sampling_params, prompt_token_ids)

        generations = []
        for output in outputs:
            text = output.outputs[0].text
            generations.append(ChatGeneration(message=AIMessage(content=text)))

        return ChatResult(generations=generations)

    def _to_chat_prompt(self, messages: List[BaseMessage]) -> Union[List[int], Dict[str, Any]]:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("at least one HumanMessage must be provided")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("last message must be a HumanMessage")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self._apply_chat_template(messages_dicts)

    def _apply_chat_template(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, List[int]]:
        """
        Apply chat template to generate model inputs.

        Args:
            messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.
            tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.

        Returns:
            Union[str, List[int]]: The generated inputs.
        """
        if self.prompt_adapter.function_call_available:
            messages = self.prompt_adapter.postprocess_messages(
                messages, functions, tools,
            )
            if functions or tools:
                logger.debug(f"==== Messages with tools ====\n{messages}")

        if "chatglm3" in self.llm.model_name:
            query, role = messages[-1]["content"], messages[-1]["role"]
            return self.tokenizer.build_chat_input(
                query, history=messages[:-1], role=role
            )["input_ids"][0].tolist()
        elif "qwen" in self.llm.model_name:
            return build_qwen_chat_input(
                self.tokenizer,
                messages,
                self.max_window_size,
                functions,
                tools,
            )
        else:
            if getattr(self.tokenizer, "chat_template", None) and not self.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self.prompt_adapter.apply_chat_template(messages)
            return prompt

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @property
    def _llm_type(self) -> str:
        return "vllm-chat-wrapper"

    def call_as_openai(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Call the model and return the output.
        """
        from vllm import SamplingParams

        llm_input = self._apply_chat_template(
            messages,
            functions=kwargs.get("functions"),
            tools=kwargs.get("tools"),
        )
        params = self._get_parameters(stop, **kwargs)

        # build sampling parameters
        sampling_params = SamplingParams(**params)
        # call the model

        if isinstance(llm_input, str):
            prompts, prompt_token_ids = [llm_input], None
        else:
            prompts, prompt_token_ids = None, [llm_input]

        outputs = self.llm.client.generate(prompts, sampling_params, prompt_token_ids)[0]

        choices = []
        for output in outputs.outputs:
            function_call, finish_reason = None, "stop"
            if params.get("functions") or params.get("tools"):
                try:
                    res, function_call = self.prompt_adapter.parse_assistant_response(
                        output.text, params.get("functions"), params.get("tools"),
                    )
                    output.text = res
                except Exception as e:
                    traceback.print_exc()
                    logger.warning("Failed to parse tool call")

            if isinstance(function_call, dict) and "arguments" in function_call:
                finish_reason = "function_call"
                function_call = FunctionCall(**function_call)
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    function_call=function_call,
                )
            elif isinstance(function_call, dict) and "function" in function_call:
                finish_reason = "tool_calls"
                tool_calls = [model_parse(ChatCompletionMessageToolCall, function_call)]
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    tool_calls=tool_calls,
                )
            else:
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text.strip(),
                )

            choices.append(
                Choice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            )

        num_prompt_tokens = len(outputs.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in outputs.outputs)
        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return ChatCompletion(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            choices=choices,
            created=int(time.time()),
            model=self.llm.model_name,
            object="chat.completion",
            usage=usage,
        )
