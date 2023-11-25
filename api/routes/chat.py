import json
import secrets
import time
from functools import partial
from typing import Generator, Any, Iterator

import anyio
from fastapi import APIRouter, Depends, Request, HTTPException
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaFunctionCall
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.config import SETTINGS
from api.generation.chatglm import process_response_v3
from api.generation.qwen import parse_response
from api.utils.protocol import ChatCompletionCreateParams, Role
from api.utils.request import (
    handle_request,
    create_error_response,
    check_api_key,
    get_engine,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine=Depends(get_engine),
):
    """Creates a completion for the chat message"""
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    request, stop_token_ids = await handle_request(request, engine.stop)
    request.max_tokens = request.max_tokens or 1024

    gen_params = request.model_dump(exclude={"messages"})
    gen_params.update(
        dict(
            prompt=request.messages,
            echo=False,
            stop_token_ids=stop_token_ids,
        )
    )

    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        generator = await run_in_threadpool(chat_completion_stream_generator, engine, gen_params, request)
        first_response = await run_in_threadpool(next, generator)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid, and we can use it to stream the response.
        def iterator() -> Iterator:
            yield first_response
            yield from generator

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
        )

    choices = []
    usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    for i in range(gen_params.get("n", 1)):
        content = await run_in_threadpool(engine.generate_gate, gen_params)
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])

        function_call, finish_reason = None, "stop"
        if request.functions and "chatglm3" in SETTINGS.model_name.lower():
            try:
                function_call = process_response_v3(content["text"], use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        elif request.functions and "qwen" in SETTINGS.model_name.lower():
            res, function_call = parse_response(content["text"])
            content["text"] = res

        if isinstance(function_call, dict) and "arguments" in function_call:
            finish_reason = "function_call"
            function_call = FunctionCall(**function_call)
            message = ChatCompletionMessage(
                role="assistant", content=content["text"], function_call=function_call
            )
        else:
            message = ChatCompletionMessage(role="assistant", content=content["text"].strip())

        choices.append(Choice(index=i, message=message, finish_reason=finish_reason))

        task_usage = CompletionUsage.model_validate(content["usage"])
        for usage_key, usage_value in task_usage.model_dump().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletion(
        id=f"chatcmpl-{secrets.token_hex(12)}", choices=choices, created=int(time.time()),
        model=request.model, object="chat.completion", usage=usage
    )


def chat_completion_stream_generator(
    engine, gen_params, request: ChatCompletionCreateParams,
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    _id = f"chatcmpl-{secrets.token_hex(12)}"
    use_tool = bool(request.functions is not None)
    for i in range(request.n):
        # First chunk with role
        choice = ChunkChoice(index=i, delta=ChoiceDelta(role="assistant", content=""), finish_reason=None)
        chunk = ChatCompletionChunk(
            id=_id, choices=[choice], created=int(time.time()),
            model=request.model, object="chat.completion.chunk",
        )
        yield chunk.model_dump_json()

        previous_text = ""
        for content in engine.generate_stream_gate(gen_params):
            if content["error_code"] != 0:
                yield json.dumps(content, ensure_ascii=False)
                return

            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text):]
            previous_text = decoded_unicode

            finish_reason = content.get("finish_reason", None)
            if len(delta_text) == 0 and finish_reason != "function_call":
                continue

            function_call = None
            if finish_reason == "function_call" and "chatglm3" in SETTINGS.model_name.lower():
                try:
                    function_call = process_response_v3(decoded_unicode, use_tool=use_tool)
                except:
                    logger.warning("Failed to parse tool call")

            elif finish_reason == "function_call" and "qwen" in SETTINGS.model_name.lower():
                _, function_call = parse_response(decoded_unicode)

            if isinstance(function_call, dict) and "arguments" in function_call:
                function_call = ChoiceDeltaFunctionCall(**function_call)
                delta = ChoiceDelta(content=delta_text, function_call=function_call)
            else:
                delta = ChoiceDelta(content=delta_text)

            choice = ChunkChoice(index=i, delta=delta, finish_reason=finish_reason)
            chunk = ChatCompletionChunk(
                id=_id, choices=[choice], created=int(time.time()),
                model=request.model, object="chat.completion.chunk",
            )
            yield chunk.model_dump_json()

        choice = ChunkChoice(index=i, delta=ChoiceDelta(), finish_reason="stop")
        chunk = ChatCompletionChunk(
            id=_id, choices=[choice], created=int(time.time()),
            model=request.model, object="chat.completion.chunk",
        )
        yield chunk.model_dump_json(exclude_none=True)
