import traceback
from abc import ABC
from typing import (
    Optional,
    Union,
    Dict,
    Iterator,
    Any,
    TYPE_CHECKING,
)

import torch
from fastapi.responses import JSONResponse
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
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

from api.common import model_validate
from api.protocol import ErrorCode
from api.templates import get_template
from api.templates.glm import generate_stream_chatglm, generate_stream_chatglm_v3
from api.templates.minicpm import generate_stream_minicpm_v
from api.templates.stream import generate_stream
from api.templates.utils import get_context_length
from api.utils import create_error_response

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedModel

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)


class HuggingFaceEngine(ABC):
    """ 基于原生 transformers 实现的模型引擎 """
    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        model_name: str,
        template_name: Optional[str] = None,
        max_model_length: Optional[int] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        self.model_name = model_name.lower()
        self.template_name = template_name.lower() if template_name else self.model_name
        self.max_model_length = max_model_length

        if self.max_model_length is None:
            self.max_model_length = get_context_length(self.model.config)

        self.template = get_template(self.template_name, self.tokenizer, self.max_model_length)

        self.generate_stream_func = generate_stream
        if "chatglm3" == self.template_name:
            self.generate_stream_func = generate_stream_chatglm_v3
        elif "chatglm" == self.template_name:
            self.generate_stream_func = generate_stream_chatglm
        elif self.model.config.model_type == "minicpmv":
            self.generate_stream_func = generate_stream_minicpm_v

        logger.info(f"Using {self.model_name} Model for Chat!")
        logger.info(f"Using {self.template} for Chat!")

    def _generate(self, params: Dict[str, Any]) -> Iterator[dict]:
        """
        Generates text based on the given parameters.

        Args:
            params (Dict[str, Any]): A dictionary containing the parameters for text generation.

        Yields:
            Iterator: A dictionary containing the generated text and error code.
        """
        prompt_or_messages = params.get("prompt_or_messages")
        if isinstance(prompt_or_messages, str):
            inputs = self.tokenizer(prompt_or_messages).input_ids
        else:
            if self.model.config.model_type == "minicpmv":
                inputs = prompt_or_messages
            else:
                inputs = self.template.convert_messages_to_ids(
                    prompt_or_messages,
                    tools=params.get("tools"),
                    max_tokens=params.get("max_tokens", 256),
                )
        params.update(dict(inputs=inputs))

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
                logprobs = model_validate(Logprobs, output["logprobs"])

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
            logprobs = model_validate(Logprobs, last_output["logprobs"])

        choice = CompletionChoice(
            index=0,
            text=last_output["text"],
            finish_reason="stop",
            logprobs=logprobs,
        )
        usage = model_validate(CompletionUsage, last_output["usage"])
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
                    _, function_call = self.template.parse_assistant_response(
                        output["text"], params.get("tools") or params.get("functions"),
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
                tool_calls = [model_validate(ChoiceDeltaToolCall, function_call)]
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
                res, function_call = self.template.parse_assistant_response(
                    last_output["text"], params.get("tools") or params.get("functions"),
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
            tool_calls = [model_validate(ChatCompletionMessageToolCall, function_call)]
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
        usage = model_validate(CompletionUsage, last_output["usage"])
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
