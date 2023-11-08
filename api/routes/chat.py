import asyncio
import json
import secrets
from typing import Generator, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from api.config import config
from api.generation.chatglm import process_response_v3
from api.generation.qwen import parse_response
from api.models import GENERATE_MDDEL
from api.routes.utils import check_requests, create_error_response, check_api_key
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
    FunctionCallResponse,
)

chat_router = APIRouter(prefix="/chat")


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Creates a completion for the chat message"""
    if len(request.messages) < 1 or request.messages[-1].role == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    # stop settings
    stop, stop_token_ids = [], []
    if GENERATE_MDDEL.stop is not None:
        stop_token_ids = GENERATE_MDDEL.stop.get("token_ids", [])
        stop = GENERATE_MDDEL.stop.get("strings", [])

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    if "qwen" in config.MODEL_NAME.lower() and request.functions:
        request.stop.append("Observation:")
    request.stop = list(set(stop + request.stop))

    request.stop_token_ids = request.stop_token_ids or []
    request.stop_token_ids = list(set(stop_token_ids + request.stop_token_ids))

    gen_params = dict(
        model=request.model,
        prompt=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        stop_token_ids=request.stop_token_ids,
        stop=request.stop,
        repetition_penalty=request.repetition_penalty,
        functions=request.functions,
    )

    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        generator = chat_completion_stream_generator(request.model, gen_params, request.n, raw_request)
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    usage = UsageInfo()
    for i in range(request.n):
        content = GENERATE_MDDEL.generate_gate(gen_params)
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])

        function_call, finish_reason = None, "stop"
        if request.functions and "chatglm3" in config.MODEL_NAME.lower():
            try:
                function_call = process_response_v3(content["text"], use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        elif request.functions and "qwen" in config.MODEL_NAME.lower():
            res, function_call = parse_response(content["text"])
            content["text"] = res

        if isinstance(function_call, dict) and "arguments" in function_call:
            finish_reason = "function_call"
            function_call = FunctionCallResponse(**function_call)

        message = ChatMessage(
            role=Role.ASSISTANT,
            content=content["text"],
            function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
        )

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


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int, raw_request: Request
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    _id = f"chatcmpl-{secrets.token_hex(12)}"
    use_tool = bool(gen_params["functions"] is not None)
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
                function_call = FunctionCallResponse(**function_call)

            delta = DeltaMessage(
                content=delta_text,
                role=Role.ASSISTANT,
                function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=delta,
                finish_reason=finish_reason,
            )

            chunk = ChatCompletionStreamResponse(id=_id, choices=[choice_data], model=model_name)
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(),
            finish_reason="stop"
        )
        chunk = ChatCompletionStreamResponse(id=_id, choices=[choice_data], model=model_name)
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n]\n"
        yield "data: [DONE]\n\n"
