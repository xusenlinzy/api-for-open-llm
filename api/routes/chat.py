import json
import secrets
from typing import Generator, Optional, Union, Dict, List, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger

from api.apapter.react import (
    check_function_call,
    build_function_call_messages,
    build_chat_message,
    build_delta_message,
)
from api.config import config
from api.models import GENERATE_MDDEL
from api.routes.utils import check_requests, create_error_response
from api.utils.constants import ErrorCode
from api.utils.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    DeltaMessage,
    UsageInfo,
    Role,
)

chat_router = APIRouter(prefix="/chat")


@chat_router.post("/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    with_function_call = check_function_call(request.messages, functions=request.functions)
    if with_function_call and "qwen" not in config.MODEL_NAME.lower():
        create_error_response(
            ErrorCode.VALIDATION_TYPE_ERROR,
            "Invalid request format: functions only supported by Qwen-7B-Chat",
        )

    messages = request.messages
    if with_function_call:
        if request.functions is None:
            for message in messages:
                if message.functions is not None:
                    request.functions = message.functions
                    break

        messages = build_function_call_messages(
            request.messages,
            request.functions,
            request.function_call,
        )

    gen_params = get_gen_params(
        request.model,
        messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
        with_function_call=with_function_call,
    )

    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    usage = UsageInfo()
    for i in range(request.n):
        content = GENERATE_MDDEL.generate_gate(gen_params)
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])

        finish_reason = "stop"
        if with_function_call:
            message, finish_reason = build_chat_message(content["text"], request.functions)
        else:
            message = ChatMessage(role=Role.ASSISTANT, content=content["text"])

        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=message,
                finish_reason=finish_reason,
            )
        )

        task_usage = UsageInfo.parse_obj(content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


def get_gen_params(
    model_name: str,
    messages: Union[str, List[ChatMessage]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    echo: Optional[bool],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]] = None,
    with_function_call: Optional[bool] = False,
) -> Dict[str, Any]:
    if not max_tokens:
        max_tokens = 1024

    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
        "with_function_call": with_function_call,
    }

    if GENERATE_MDDEL.stop is not None:
        if "token_ids" in GENERATE_MDDEL.stop:
            gen_params["stop_token_ids"] = GENERATE_MDDEL.stop["token_ids"]

        if "strings" in GENERATE_MDDEL.stop:
            gen_params["stop"] = GENERATE_MDDEL.stop["strings"]

    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]

        gen_params["stop"] = gen_params["stop"] + stop if "stop" in gen_params else stop
        gen_params["stop"] = list(set(gen_params["stop"]))

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    _id = f"chatcmpl-{secrets.token_hex(12)}"
    finish_stream_events = []
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role=Role.ASSISTANT),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=_id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        with_function_call = gen_params.get("with_function_call", False)
        found_action_name = False
        for content in GENERATE_MDDEL.generate_stream_gate(gen_params):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text):]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None

            messages = []
            if with_function_call:
                if found_action_name:
                    messages.append(build_delta_message(delta_text, "arguments"))
                    finish_reason = "function_call"
                else:
                    if previous_text.rfind("\nFinal Answer:") > 0:
                        with_function_call = False

                    if previous_text.rfind("\nAction Input:") == -1:
                        continue
                    else:
                        messages.append(build_delta_message(previous_text))
                        pos = previous_text.rfind("\nAction Input:") + len("\nAction Input:")
                        messages.append(build_delta_message(previous_text[pos:], "arguments"))

                        found_action_name = True
                        finish_reason = "function_call"
            else:
                messages = [DeltaMessage(content=delta_text)]
                finish_reason = content.get("finish_reason", "stop")

            chunks = []
            for m in messages:
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=m,
                    finish_reason=finish_reason,
                )
                chunks.append(ChatCompletionStreamResponse(id=_id, choices=[choice_data], model=model_name))

            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.extend(chunks)
                continue

            for chunk in chunks:
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
