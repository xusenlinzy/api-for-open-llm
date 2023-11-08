import asyncio
import json
import secrets
import time
from typing import Generator, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from openai.types.chat import (
    CompletionCreateParams,
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaFunctionCall
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.completion_usage import CompletionUsage

from api.config import config
from api.generation.chatglm import process_response_v3
from api.generation.qwen import parse_response
from api.models import GENERATE_MDDEL
from api.routes.utils import check_requests, create_error_response, check_api_key
from api.utils.protocol import Role

chat_router = APIRouter(prefix="/chat")


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: CompletionCreateParams, raw_request: Request):
    """Creates a completion for the chat message"""
    messages = request.get("messages")
    if (not messages) or messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    # stop settings
    _stop, stop_token_ids = [], []
    if GENERATE_MDDEL.stop is not None:
        stop_token_ids = GENERATE_MDDEL.stop.get("token_ids", [])
        _stop = GENERATE_MDDEL.stop.get("strings", [])

    stop = request.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]

    functions = request.get("functions")
    if "qwen" in config.MODEL_NAME.lower() and functions:
        stop.append("Observation:")
    stop = list(set(_stop + stop))

    gen_params = request.copy()
    gen_params.update(
        dict(
            prompt=messages,
            max_tokens=request.get("max_tokens", 1024),
            echo=False,
            stop=stop,
            stop_token_ids=stop_token_ids,
        )
    )

    logger.debug(f"==== request ====\n{gen_params}")

    stream = gen_params.get("stream")
    if stream:
        generator = chat_completion_stream_generator(gen_params, raw_request)
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    for i in range(gen_params.get("n", 1)):
        content = GENERATE_MDDEL.generate_gate(gen_params)
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])

        function_call, finish_reason = None, "stop"
        if functions and "chatglm3" in config.MODEL_NAME.lower():
            try:
                function_call = process_response_v3(content["text"], use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        elif functions and "qwen" in config.MODEL_NAME.lower():
            res, function_call = parse_response(content["text"])
            content["text"] = res

        if isinstance(function_call, dict) and "arguments" in function_call:
            finish_reason = "function_call"
            function_call = FunctionCall(**function_call)
            message = ChatCompletionMessage(
                role="assistant", content=content["text"], function_call=function_call
            )
        else:
            message = ChatCompletionMessage(role="assistant", content=content["text"])

        choices.append(Choice(index=i, message=message, finish_reason=finish_reason))

        task_usage = CompletionUsage.parse_obj(content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletion(
        id=f"chatcmpl-{secrets.token_hex(12)}", choices=choices, created=int(time.time()),
        model=request.get("model"), object="chat.completion", usage=usage
    )


async def chat_completion_stream_generator(
    gen_params: Dict[str, Any], raw_request: Request
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    _id = f"chatcmpl-{secrets.token_hex(12)}"
    use_tool = bool(gen_params["functions"] is not None)
    for i in range(gen_params["n"]):
        # First chunk with role
        choice = ChunkChoice(index=i, delta=ChoiceDelta(role="assistant"), finish_reason=None)
        chunk = ChatCompletionChunk(
            id=_id, choices=[choice], created=int(time.time()),
            model=gen_params["model"], object="chat.completion.chunk",
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        for content in GENERATE_MDDEL.generate_stream_gate(gen_params):
            if await raw_request.is_disconnected():
                asyncio.current_task().cancel()
                return

            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text):]
            previous_text = decoded_unicode

            finish_reason = content.get("finish_reason", None)
            if len(delta_text) == 0 and finish_reason != "function_call":
                continue

            function_call = None
            if finish_reason == "function_call" and "chatglm3" in config.MODEL_NAME.lower():
                try:
                    function_call = process_response_v3(decoded_unicode, use_tool=use_tool)
                except:
                    logger.warning("Failed to parse tool call")

            elif finish_reason == "function_call" and "qwen" in config.MODEL_NAME.lower():
                _, function_call = parse_response(decoded_unicode)

            if isinstance(function_call, dict) and "arguments" in function_call:
                function_call = ChoiceDeltaFunctionCall(**function_call)
                delta = ChoiceDelta(content=delta_text, role="assistant", function_call=function_call)
            else:
                delta = ChoiceDelta(content=delta_text, role="assistant")

            choice = ChunkChoice(index=i, delta=delta, finish_reason=finish_reason)
            chunk = ChatCompletionChunk(
                id=_id, choices=[choice], created=int(time.time()),
                model=gen_params["model"], object="chat.completion.chunk",
            )
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        choice = ChunkChoice(index=i, delta=ChoiceDelta(), finish_reason="stop")
        chunk = ChatCompletionChunk(
            id=_id, choices=[choice], created=int(time.time()),
            model=gen_params["model"], object="chat.completion.chunk",
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n]\n"
        yield "data: [DONE]\n\n"
