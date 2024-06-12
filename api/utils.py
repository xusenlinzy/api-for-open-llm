import json
from threading import Lock
from typing import (
    Optional,
    Union,
    Iterator,
    List,
    AsyncIterator,
)

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from api.common import jsonify, dictify
from api.config import SETTINGS
from api.protocol import (
    ChatCompletionCreateParams,
    CompletionCreateParams,
    ErrorResponse,
    ErrorCode
)

llama_outer_lock = Lock()
llama_inner_lock = Lock()


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):
    if not SETTINGS.api_keys:
        # api_keys not set; allow all
        return None
    if auth is None or (token := auth.credentials) not in SETTINGS.api_keys:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )
    return token


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        dictify(
            ErrorResponse(
                message=message,
                code=code
            )
        ),
        status_code=500
    )


async def check_completion_requests(
    request: Union[CompletionCreateParams, ChatCompletionCreateParams],
    stop: Optional[List[str]] = None,
    stop_token_ids: Optional[List[int]] = None,
    chat: bool = True,
) -> Union[CompletionCreateParams, ChatCompletionCreateParams, JSONResponse]:
    error_check_ret = _check_completion_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    _stop = stop or []
    _stop_token_ids = stop_token_ids or []

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    if chat and (
        "qwen" in SETTINGS.model_name.lower()
        and (request.functions is not None or request.tools is not None)
    ):
        request.stop.append("Observation:")

    request.stop = list(set(_stop + request.stop))
    request.stop_token_ids = request.stop_token_ids or []
    request.stop_token_ids = list(set(_stop_token_ids + request.stop_token_ids))

    request.top_p = max(request.top_p, 1e-5)
    if request.temperature <= 1e-5:
        request.top_p = 1.0

    return request


def _check_completion_requests(request: Union[CompletionCreateParams, ChatCompletionCreateParams]) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is None or isinstance(request.stop, (str, list)):
        return None
    else:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Union[Iterator, AsyncIterator],
):
    async with inner_send_chan:
        try:
            if SETTINGS.engine not in ["vllm", "tgi"]:
                async for chunk in iterate_in_threadpool(iterator):
                    if isinstance(chunk, BaseModel):
                        chunk = jsonify(chunk)
                    elif isinstance(chunk, dict):
                        chunk = json.dumps(chunk, ensure_ascii=False)

                    await inner_send_chan.send(dict(data=chunk))

                    if await request.is_disconnected():
                        raise anyio.get_cancelled_exc_class()()

                    if SETTINGS.interrupt_requests and llama_outer_lock.locked():
                        await inner_send_chan.send(dict(data="[DONE]"))
                        raise anyio.get_cancelled_exc_class()()
            else:
                async for chunk in iterator:
                    chunk = jsonify(chunk)
                    await inner_send_chan.send(dict(data=chunk))
                    if await request.is_disconnected():
                        raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))

        except anyio.get_cancelled_exc_class() as e:
            logger.info("disconnected")
            with anyio.move_on_after(1, shield=True):
                logger.info(f"Disconnected from client (via refresh/close) {request.client}")
                raise e
