from __future__ import annotations

import traceback
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    Callable,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.outputs import (
    GenerationChunk,
    ChatGenerationChunk,
)
from langchain_core.pydantic_v1 import root_validator
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion import Completion
from openai.types.completion_choice import (
    CompletionChoice,
    Logprobs,
)
from openai.types.completion_usage import CompletionUsage

from ._compat import model_parse
from .adapters.template import (
    get_prompt_adapter,
    BaseTemplate,
)
from .generation import (
    generate_stream_chatglm,
    check_is_chatglm,
    generate_stream_chatglm_v3,
    check_is_qwen,
    generate_stream,
    check_is_baichuan,
    check_is_xverse,
    build_baichuan_chat_input,
    build_qwen_chat_input,
    build_xverse_chat_input,
)

__all__ = [
    "HuggingFaceLLM",
    "ChatHuggingFace",
]


class HuggingFaceLLM(LLM):
    """HuggingFace language model."""

    model_name: str
    """The name of a HuggingFace Transformers model."""

    model_path: str
    """The path to the HuggingFace Transformers model file."""

    load_model_kwargs: Optional[dict] = {}
    """Keyword arguments to pass to load the model."""

    context_length: Optional[int] = None
    """Context length for generating completions."""

    use_streamer_v2: Optional[bool] = True
    """Support for transformers.TextIteratorStreamer."""

    n: int = 1
    """Number of output sequences to return for the given prompt."""

    temperature: float = 1.0
    """Float that controls the randomness of the sampling."""

    top_p: float = 1.0
    """Float that controls the cumulative probability of the top tokens to consider."""

    top_k: int = -1
    """Integer that controls the number of top tokens to consider."""

    stop: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated."""

    max_new_tokens: int = 512
    """Maximum number of tokens to generate per output sequence."""

    logprobs: Optional[int] = None
    """Number of log probabilities to return per output token."""

    echo: Optional[bool] = False
    """Echo back the prompt in addition to the completion"""

    model: Any  #: :meta private:

    tokenizer: Any  #: :meta private:

    inference_fn: Callable = generate_stream  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        from .adapters.model import load_model_and_tokenizer

        values["model_name"] = values["model_name"].lower()
        model_path = values["model_path"]
        try:
            values["model"], values["tokenizer"] = load_model_and_tokenizer(
                model_name_or_path=model_path,
                **values["load_model_kwargs"],
            )
        except Exception as e:
            raise ValueError(
                f"Could not load model from path: {model_path}. "
                f"Received error {e}"
            )

        return cls._validate_environment(values)

    @staticmethod
    def _validate_environment(values: Dict) -> Dict:
        """
        Prepare for text generation.

        1. Sets the appropriate generate stream function based on the model name and type.
        2. Updates the context length if necessary.
        3. Checks and constructs the prompt.
        4. Sets the context length if it is not already set.
        """
        from .generation.utils import get_context_length

        model_name = values["model_name"]
        if "chatglm3" in model_name:
            values["inference_fn"] = generate_stream_chatglm_v3
            values["use_streamer_v2"] = False
        elif check_is_chatglm(values["model"]):
            values["inference_fn"] = generate_stream_chatglm
        elif check_is_qwen(values["model"]):
            values["context_length"] = values["context_length"] or 8192

        if values["context_length"] is None:
            values["context_length"] = get_context_length(values["model"].config)
            logger.info(f"Context length is set to : {values['context_length']}")

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling vllm."""
        return {
            "n": self.n,
            "max_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop": self.stop,
            "logprobs": self.logprobs,
            "echo": self.echo,
            "temperature": self.temperature,
        }

    def _convert_to_inputs(
        self,
        prompt: str,
        infilling: Optional[bool] = False,
        suffix_first: Optional[bool] = False,
        **kwargs,
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Convert the prompt into input format for the model.

        Args:
            prompt: The prompt to be converted.
            infilling: Whether to perform infilling.
            suffix_first: Whether to append the suffix first.
            **kwargs: Additional keyword arguments.

        Returns:
            The converted inputs.
        """
        if infilling:
            inputs = self.tokenizer(prompt, suffix_first=suffix_first).input_ids
        elif check_is_qwen(self.model):
            inputs = self.tokenizer(prompt, allowed_special="all", disallowed_special=()).input_ids
        elif check_is_chatglm(self.model):
            inputs = self.tokenizer([prompt], return_tensors="pt")
        else:
            inputs = self.tokenizer(prompt).input_ids

        if isinstance(inputs, list):
            max_src_len = self.context_length - kwargs.get("max_tokens", 256) - 1
            inputs = inputs[-max_src_len:]

        return inputs

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name, "model_path": self.model_path}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "huggingface_llm"

    def _get_parameters(
        self,
        prompt: str,
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

        # Raise error if stop sequences are in both input and default params
        if self.stop and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")

        params = self._default_params

        # then sets it as configured, or default to an empty list:
        params["stop"] = self.stop or stop or []

        params = {**params, **kwargs}
        inputs = self._convert_to_inputs(
            prompt,
            infilling=params.get("infilling", False),
            suffix_first=params.get("suffix_first", False),
            max_new_tokens=params.get("max_tokens", 256),
            functions=params.get("functions"),
            tools=params.get("tools"),
        )
        params.update(dict(inputs=inputs, prompt=prompt))

        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain_llm import HuggingFaceLLM

                llm = HuggingFaceLLM(
                    model_name="qwen-7b-chat",
                    model_path="/data/checkpoints/Qwen-7B-Chat",
                    load_model_kwargs={"device_map": "auto"},
                )
                llm("This is a prompt.")
        """
        combined_text_output = ""
        for chunk in self._stream(
            prompt=prompt,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            combined_text_output += chunk.text
        return combined_text_output

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.

        Example:
            .. code-block:: python

                from langchain_llm import HuggingFaceLLM

                llm = HuggingFaceLLM(
                    model_name="qwen-7b-chat",
                    model_path="/data/checkpoints/Qwen-7B-Chat",
                    load_model_kwargs={"device_map": "auto"},
                )
                for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'"):
                    print(chunk, end='', flush=True)

        """
        params = self._get_parameters(prompt, stop, **kwargs)
        result = self.inference_fn(self.model, self.tokenizer, params)
        for part in result:
            logprobs = part.get("logprobs", None)
            chunk = GenerationChunk(
                text=part["delta"],
                generation_info={"logprobs": logprobs},
            )
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    token=chunk.text, verbose=self.verbose, log_probs=logprobs
                )

    def _create_openai_completion(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Completion:
        """
        Creates a completion based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for creating the completion.

        Returns:
            Completion: The generated completion object.
        """
        params = self._get_parameters(prompt, stop, **kwargs)
        result = self.inference_fn(self.model, self.tokenizer, params)
        last_output = None
        for part in result:
            last_output = part

        logprobs = None
        if params.get("logprobs") and last_output["logprobs"]:
            logprobs = model_parse(Logprobs, last_output["logprobs"])

        choice = CompletionChoice(
            index=0,
            text=last_output["text"],
            finish_reason="stop",
            logprobs=logprobs,
        )
        usage = model_parse(CompletionUsage, last_output["usage"])
        return Completion(
            id=last_output["id"],
            choices=[choice],
            created=last_output["created"],
            model=self.model_name,
            object="text_completion",
            usage=usage,
        )

    def _create_openai_stream_completion(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Completion]:
        """
        Generates a stream of completions based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for generating completions.

        Yields:
            Iterator: A stream of completion objects.
        """
        params = self._get_parameters(prompt, stop, **kwargs)
        result = self.inference_fn(self.model, self.tokenizer, params)
        for part in result:
            logprobs = None
            if params.get("logprobs") and part["logprobs"]:
                logprobs = model_parse(Logprobs, part["logprobs"])

            choice = CompletionChoice(
                index=0,
                text=part["delta"],
                finish_reason="stop",
                logprobs=logprobs,
            )
            yield Completion(
                id=part["id"],
                choices=[choice],
                created=part["created"],
                model=self.model_name,
                object="text_completion",
            )

    def call_as_openai(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[Completion, Iterator[Completion]]:
        """Call the model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            Union[Completion, Iterator[Completion]]

        """
        if stream:
            return self._create_openai_stream_completion(prompt, stop, **kwargs)
        else:
            return self._create_openai_completion(prompt, stop, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        tokenized_text = self.tokenizer.tokenize(text.encode("utf-8"))
        return len(tokenized_text)


class ChatHuggingFace(BaseChatModel):
    """
    Wrapper for using Hugging Face LLM's as ChatModels.

    Works with `HuggingFaceLLM` LLMs.
    """

    llm: HuggingFaceLLM

    chat_template: Optional[str] = None
    """Chat template for generating completions."""

    max_window_size: Optional[int] = 6144
    """The maximum window size"""

    construct_prompt: bool = True

    prompt_adapter: Optional[BaseTemplate] = None

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

        values["construct_prompt"] = values["chat_template"] is not None

        if "chatglm3" in model_name:
            logger.info("Using ChatGLM3 Model for Chat!")
        elif check_is_baichuan(values["llm"].model):
            logger.info("Using Baichuan Model for Chat!")
        elif check_is_qwen(values["llm"].model):
            logger.info("Using Qwen Model for Chat!")
        elif check_is_xverse(values["llm"].model):
            logger.info("Using Xverse Model for Chat!")
        else:
            values["construct_prompt"] = True

        values["llm"].echo = False

        return values

    def _get_parameters(
        self,
        messages: Union[List[BaseMessage], List[Dict[str, Any]]],
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

        llm_input = self._apply_chat_template(
            messages,
            max_new_tokens=params.get("max_tokens"),
            functions=params.get("functions"),
            tools=params.get("tools"),
        )
        params.update(dict(inputs=llm_input, prompt=None))

        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

        combined_text_output = ""
        for chunk in self._stream(
            messages,
            stop,
            run_manager,
            **kwargs,
        ):
            combined_text_output += chunk.message.content

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=combined_text_output))],
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:

        messages = self._to_chat_prompt(messages)
        params = self._get_parameters(messages, stop, **kwargs)
        result = self.llm.inference_fn(self.llm.model, self.llm.tokenizer, params)
        for part in result:
            logprobs = part.get("logprobs", None)
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=part["delta"]),
                generation_info={"logprobs": logprobs},
            )
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    token=chunk.text, verbose=self.verbose, log_probs=logprobs
                )

    def _to_chat_prompt(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("at least one HumanMessage must be provided")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("last message must be a HumanMessage")

        return [self._to_chatml_format(m) for m in messages]

    def _apply_chat_template(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        max_new_tokens: Optional[int] = 256,
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Apply chat template to generate model inputs.

        Args:
            messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.
            max_new_tokens (Optional[int], optional): Maximum number of new tokens to generate. Defaults to 256.
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.
            tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.

        Returns:
            Union[List[int], Dict[str, Any]]: The generated inputs.
        """
        if self.prompt_adapter.function_call_available:
            messages = self.prompt_adapter.postprocess_messages(
                messages, functions, tools=tools,
            )
            if functions or tools:
                logger.debug(f"==== Messages with tools ====\n{messages}")

        if self.construct_prompt:
            if getattr(self.llm.tokenizer, "chat_template", None) and not self.chat_template:
                prompt = self.llm.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self.prompt_adapter.apply_chat_template(messages)

            if check_is_qwen(self.llm.model):
                inputs = self.llm.tokenizer(prompt, allowed_special="all", disallowed_special=()).input_ids
            elif check_is_chatglm(self.llm.model):
                inputs = self.llm.tokenizer([prompt], return_tensors="pt")
            else:
                inputs = self.llm.tokenizer(prompt).input_ids

            if isinstance(inputs, list):
                max_src_len = self.llm.context_length - max_new_tokens - 1
                inputs = inputs[-max_src_len:]
        else:
            inputs = self._build_chat_inputs(messages, max_new_tokens, functions, tools)

        return inputs

    def _build_chat_inputs(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        max_new_tokens: Optional[int] = 256,
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        if "chatglm3" in self.llm.model_name:
            query, role = messages[-1]["content"], messages[-1]["role"]
            inputs = self.llm.tokenizer.build_chat_input(query, history=messages[:-1], role=role)
        elif check_is_baichuan(self.llm.model):
            inputs = build_baichuan_chat_input(
                self.llm.tokenizer, messages, self.llm.context_length, max_new_tokens
            )
        elif check_is_qwen(self.llm.model):
            inputs = build_qwen_chat_input(
                self.llm.tokenizer, messages, self.max_window_size, functions, tools,
            )
        elif check_is_xverse(self.llm.model):
            inputs = build_xverse_chat_input(
                self.llm.tokenizer, messages, self.llm.context_length, max_new_tokens
            )
        else:
            raise NotImplementedError
        return inputs

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
        return "huggingface-chat-wrapper"

    def _create_openai_chat_completion(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        Creates a completion based on the given parameters.
        """

        params = self._get_parameters(messages, stop, **kwargs)
        result = self.llm.inference_fn(self.llm.model, self.llm.tokenizer, params)

        last_output = None
        for part in result:
            last_output = part

        function_call, finish_reason = None, "stop"
        if params.get("functions") or params.get("tools"):
            try:
                res, function_call = self.prompt_adapter.parse_assistant_response(
                    last_output["text"], params.get("functions"), params.get("tools"),
                )
                last_output["text"] = res
            except Exception as e:
                traceback.print_exc()
                logger.warning("Failed to parse tool call")

        if isinstance(function_call, dict) and "arguments" in function_call:
            finish_reason = "function_call"
            function_call = FunctionCall(**function_call)
            message = ChatCompletionMessage(
                role="assistant",
                content=last_output["text"],
                function_call=function_call,
            )
        elif isinstance(function_call, dict) and "function" in function_call:
            finish_reason = "tool_calls"
            tool_calls = [model_parse(ChatCompletionMessageToolCall, function_call)]
            message = ChatCompletionMessage(
                role="assistant",
                content=last_output["text"],
                tool_calls=tool_calls,
            )
        else:
            message = ChatCompletionMessage(
                role="assistant",
                content=last_output["text"].strip(),
            )

        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason,
            logprobs=None,
        )
        usage = model_parse(CompletionUsage, last_output["usage"])
        return ChatCompletion(
            id=f"chat{last_output['id']}",
            choices=[choice],
            created=last_output["created"],
            model=self.llm.model_name,
            object="chat.completion",
            usage=usage,
        )

    def _create_openai_chat_stream_completion(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """
        Generates a stream of completions based on the given parameters.
        """

        params = self._get_parameters(messages, stop, **kwargs)
        result = self.llm.inference_fn(self.llm.model, self.llm.tokenizer, params)

        _id, _created, _model = None, None, self.llm.model_name
        has_function_call = False
        for i, output in enumerate(result):
            _id, _created = output["id"], output["created"]
            if i == 0:
                choice = ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=""),
                    finish_reason=None,
                    logprobs=None,
                )
                yield ChatCompletionChunk(
                    id=f"chat{_id}",
                    choices=[choice],
                    created=_created,
                    model=_model,
                    object="chat.completion.chunk",
                )

            finish_reason = output["finish_reason"]
            if len(output["delta"]) == 0 and finish_reason != "function_call":
                continue

            function_call = None
            if finish_reason == "function_call":
                try:
                    _, function_call = self.prompt_adapter.parse_assistant_response(
                        output["text"], params.get("functions"), params.get("tools"),
                    )
                except Exception as e:
                    traceback.print_exc()
                    logger.warning("Failed to parse tool call")

            if isinstance(function_call, dict) and "arguments" in function_call:
                has_function_call = True
                function_call = ChoiceDeltaFunctionCall(**function_call)
                delta = ChoiceDelta(
                    content=output["delta"],
                    function_call=function_call
                )
            elif isinstance(function_call, dict) and "function" in function_call:
                has_function_call = True
                finish_reason = "tool_calls"
                function_call["index"] = 0
                tool_calls = [model_parse(ChoiceDeltaToolCall, function_call)]
                delta = ChoiceDelta(
                    content=output["delta"],
                    tool_calls=tool_calls,
                )
            else:
                delta = ChoiceDelta(content=output["delta"])

            choice = ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
                logprobs=None,
            )
            yield ChatCompletionChunk(
                id=f"chat{_id}",
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
            )

        if not has_function_call:
            choice = ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop",
                logprobs=None,
            )
            yield ChatCompletionChunk(
                id=f"chat{_id}",
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
            )

    def call_as_openai(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Call the model and return the output.
        """
        if stream:
            return self._create_openai_chat_stream_completion(
                messages,
                stop,
                **kwargs,
            )
        else:
            return self._create_openai_chat_completion(
                messages,
                stop,
                **kwargs,
            )
