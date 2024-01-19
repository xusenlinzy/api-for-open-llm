import traceback
from abc import ABC
from typing import (
    Optional,
    List,
    Union,
    Tuple,
    Dict,
    Iterator,
    Any,
)

import torch
from fastapi.responses import JSONResponse
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
from openai.types.completion_choice import CompletionChoice, Logprobs
from openai.types.completion_usage import CompletionUsage
from transformers import PreTrainedModel, PreTrainedTokenizer

from api.adapter import get_prompt_adapter
from api.generation import (
    build_baichuan_chat_input,
    check_is_baichuan,
    generate_stream_chatglm,
    check_is_chatglm,
    generate_stream_chatglm_v3,
    build_qwen_chat_input,
    check_is_qwen,
    generate_stream,
    build_xverse_chat_input,
    check_is_xverse,
)
from api.generation.utils import get_context_length
from api.utils.compat import model_parse
from api.utils.constants import ErrorCode
from api.utils.request import create_error_response

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)


class DefaultEngine(ABC):
    """ 基于原生 transformers 实现的模型引擎 """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Union[str, torch.device],
        model_name: str,
        context_len: Optional[int] = None,
        prompt_name: Optional[str] = None,
        use_streamer_v2: Optional[bool] = False,
    ) -> None:
        """
        Initialize the Default class.

        Args:
            model (PreTrainedModel): The pre-trained model.
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            device (Union[str, torch.device]): The device to use for inference.
            model_name (str): The name of the model.
            context_len (Optional[int], optional): The length of the context. Defaults to None.
            prompt_name (Optional[str], optional): The name of the prompt. Defaults to None.
            use_streamer_v2 (Optional[bool], optional): Whether to use Streamer V2. Defaults to False.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if hasattr(model, "device") else torch.device(device)

        self.model_name = model_name.lower()
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.context_len = context_len
        self.use_streamer_v2 = use_streamer_v2

        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)

        self._prepare_for_generate()
        self._patch_tokenizer()

    def _prepare_for_generate(self) -> None:
        """
        Prepare the object for text generation.

        1. Sets the appropriate generate stream function based on the model name and type.
        2. Updates the context length if necessary.
        3. Checks and constructs the prompt.
        4. Sets the context length if it is not already set.
        """
        self.generate_stream_func = generate_stream
        if "chatglm3" in self.model_name:
            self.generate_stream_func = generate_stream_chatglm_v3
            self.use_streamer_v2 = False
        elif check_is_chatglm(self.model):
            self.generate_stream_func = generate_stream_chatglm
        elif check_is_qwen(self.model):
            self.context_len = 8192 if self.context_len is None else self.context_len

        self._check_construct_prompt()

        if self.context_len is None:
            self.context_len = get_context_length(self.model.config)

    def _check_construct_prompt(self) -> None:
        """ Check whether to need to construct prompts or inputs. """
        self.construct_prompt = self.prompt_name is not None
        if "chatglm3" in self.model_name:
            logger.info("Using ChatGLM3 Model for Chat!")
        elif check_is_baichuan(self.model):
            logger.info("Using Baichuan Model for Chat!")
        elif check_is_qwen(self.model):
            logger.info("Using Qwen Model for Chat!")
        elif check_is_xverse(self.model):
            logger.info("Using Xverse Model for Chat!")
        else:
            self.construct_prompt = True

    def _patch_tokenizer(self) -> None:
        """ 
        Fix the tokenizer by adding the end-of-sequence (eos) token 
        and the padding (pad) token if they are missing.
        """
        from api.adapter.patcher import patch_tokenizer

        patch_tokenizer(self.tokenizer)

    def convert_to_inputs(
        self,
        prompt_or_messages: Union[List[ChatCompletionMessageParam], str],
        infilling: Optional[bool] = False,
        suffix_first: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[List[int], Dict[str, Any]], Union[List[ChatCompletionMessageParam], str]]:
        """
        Convert the prompt or messages into input format for the model.

        Args:
            prompt_or_messages: The prompt or messages to be converted.
            infilling: Whether to perform infilling.
            suffix_first: Whether to append the suffix first.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple containing the converted inputs and the prompt or messages.
        """
        # for completion
        if isinstance(prompt_or_messages, str):
            if infilling:
                inputs = self.tokenizer(
                    prompt_or_messages, suffix_first=suffix_first,
                ).input_ids
            elif check_is_qwen(self.model):
                inputs = self.tokenizer(
                    prompt_or_messages, allowed_special="all", disallowed_special=()
                ).input_ids
            elif check_is_chatglm(self.model):
                inputs = self.tokenizer([prompt_or_messages], return_tensors="pt")
            else:
                inputs = self.tokenizer(prompt_or_messages).input_ids

            if isinstance(inputs, list):
                max_src_len = self.context_len - kwargs.get("max_tokens", 256) - 1
                inputs = inputs[-max_src_len:]

        else:
            inputs, prompt_or_messages = self.apply_chat_template(prompt_or_messages, **kwargs)
        return inputs, prompt_or_messages

    def apply_chat_template(
        self,
        messages: List[ChatCompletionMessageParam],
        max_new_tokens: Optional[int] = 256,
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Tuple[Union[List[int], Dict[str, Any]], Optional[str]]:
        """
        Apply chat template to generate model inputs and prompt.

        Args:
            messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.
            max_new_tokens (Optional[int], optional): Maximum number of new tokens to generate. Defaults to 256.
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.
            tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Union[List[int], Dict[str, Any]], Union[str, None]]: Tuple containing the generated inputs and prompt.
        """
        if self.prompt_adapter.function_call_available:
            messages = self.prompt_adapter.postprocess_messages(
                messages, functions, tools=tools,
            )
            if functions or tools:
                logger.debug(f"==== Messages with tools ====\n{messages}")

        if self.construct_prompt:
            prompt = self.prompt_adapter.apply_chat_template(messages)
            if check_is_qwen(self.model):
                inputs = self.tokenizer(prompt, allowed_special="all", disallowed_special=()).input_ids
            elif check_is_chatglm(self.model):
                inputs = self.tokenizer([prompt], return_tensors="pt")
            else:
                inputs = self.tokenizer(prompt).input_ids

            if isinstance(inputs, list):
                max_src_len = self.context_len - max_new_tokens - 1
                inputs = inputs[-max_src_len:]
            return inputs, prompt
        else:
            inputs = self.build_chat_inputs(
                messages, max_new_tokens, functions, tools, **kwargs
            )
        return inputs, None

    def build_chat_inputs(
        self,
        messages: List[ChatCompletionMessageParam],
        max_new_tokens: Optional[int] = 256,
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[int]:
        if "chatglm3" in self.model_name:
            query, role = messages[-1]["content"], messages[-1]["role"]
            inputs = self.tokenizer.build_chat_input(query, history=messages[:-1], role=role)
        elif check_is_baichuan(self.model):
            inputs = build_baichuan_chat_input(
                self.tokenizer, messages, self.context_len, max_new_tokens
            )
        elif check_is_qwen(self.model):
            inputs = build_qwen_chat_input(
                self.tokenizer, messages, functions=functions, tools=tools,
            )
        elif check_is_xverse(self.model):
            inputs = build_xverse_chat_input(
                self.tokenizer, messages, self.context_len, max_new_tokens
            )
        else:
            raise NotImplementedError
        return inputs

    def _generate(self, params: Dict[str, Any]) -> Iterator[dict]:
        """
        Generates text based on the given parameters.

        Args:
            params (Dict[str, Any]): A dictionary containing the parameters for text generation.

        Yields:
            Iterator: A dictionary containing the generated text and error code.
        """
        prompt_or_messages = params.get("prompt_or_messages")
        inputs, prompt = self.convert_to_inputs(
            prompt_or_messages,
            infilling=params.get("infilling", False),
            suffix_first=params.get("suffix_first", False),
            max_new_tokens=params.get("max_tokens", 256),
            functions=params.get("functions"),
            tools=params.get("tools"),
        )
        params.update(dict(inputs=inputs, prompt=prompt))

        try:
            for output in self.generate_stream_func(self.model, self.tokenizer, params):
                output["error_code"] = 0
                yield output

        except torch.cuda.OutOfMemoryError as e:
            yield {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }

        except (ValueError, RuntimeError) as e:
            traceback.print_exc()
            yield {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }

    def _create_completion_stream(self, params: Dict[str, Any]) -> Iterator[Completion]:
        """
        Generates a stream of completions based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for generating completions.

        Yields:
            Iterator: A stream of completion objects.
        """
        for output in self._generate(params):
            if output["error_code"] != 0:
                yield output
                return

            logprobs = None
            if params.get("logprobs") and output["logprobs"]:
                logprobs = model_parse(Logprobs, output["logprobs"])

            choice = CompletionChoice(
                index=0,
                text=output["delta"],
                finish_reason="stop",
                logprobs=logprobs,
            )
            yield Completion(
                id=output["id"],
                choices=[choice],
                created=output["created"],
                model=output["model"],
                object="text_completion",
            )

    def _create_completion(self, params: Dict[str, Any]) -> Union[Completion, JSONResponse]:
        """
        Creates a completion based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for creating the completion.

        Returns:
            Completion: The generated completion object.
        """
        last_output = None
        for output in self._generate(params):
            last_output = output

        if last_output["error_code"] != 0:
            return create_error_response(last_output["error_code"], last_output["text"])

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
            model=last_output["model"],
            object="text_completion",
            usage=usage,
        )

    def _create_chat_completion_stream(self, params: Dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        """
        Creates a chat completion stream.

        Args:
            params (Dict[str, Any]): The parameters for generating the chat completion.

        Yields:
            Dict[str, Any]: The output of the chat completion stream.
        """
        _id, _created, _model = None, None, None
        has_function_call = False
        for i, output in enumerate(self._generate(params)):
            if output["error_code"] != 0:
                yield output
                return

            _id, _created, _model = output["id"], output["created"], output["model"]
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

    def _create_chat_completion(self, params: Dict[str, Any]) -> Union[ChatCompletion, JSONResponse]:
        """
        Creates a chat completion based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for generating the chat completion.

        Returns:
            ChatCompletion: The generated chat completion.
        """
        last_output = None
        for output in self._generate(params):
            last_output = output

        if last_output["error_code"] != 0:
            return create_error_response(last_output["error_code"], last_output["text"])

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
            model=last_output["model"],
            object="chat.completion",
            usage=usage,
        )

    def create_completion(
        self,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Iterator[Completion], Completion]:
        params = params or {}
        params.update(kwargs)
        return (
            self._create_completion_stream(params)
            if params.get("stream", False)
            else self._create_completion(params)
        )

    def create_chat_completion(
        self,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Iterator[ChatCompletionChunk], ChatCompletion]:
        params = params or {}
        params.update(kwargs)
        return (
            self._create_chat_completion_stream(params)
            if params.get("stream", False)
            else self._create_chat_completion(params)
        )

    @property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            The stop property of the prompt adapter, or None if it does not exist.
        """
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
