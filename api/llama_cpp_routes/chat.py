import json
import secrets
from functools import partial
from typing import Iterator

import anyio
from fastapi import APIRouter, Depends, Request, HTTPException
from loguru import logger
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.utils.protocol import Role, ChatCompletionCreateParams
from api.utils.request import (
    handle_request,
    check_api_key,
    get_llama_cpp_engine,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine=Depends(get_llama_cpp_engine),
):
    logger.info(f"Received chat messages: {request.messages}")

    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    request, stop_token_ids = await handle_request(request, engine.prompt_adapter.stop)
    request.max_tokens = request.max_tokens or 512

    prompt = engine.prompt_adapter.apply_chat_template(request.messages)
    include = {
        "temperature", "temperature", "top_p", "stream", "stop",
        "max_tokens", "presence_penalty", "frequency_penalty", "model"
    }
    kwargs = request.model_dump(include=include)
    iterator_or_completion = await run_in_threadpool(engine.create_completion, prompt, **kwargs)

    _id = f"chatcmpl-{secrets.token_hex(12)}"

    if not request.stream:
        completion = iterator_or_completion
        return {
            "id": _id,
            "object": "chat.completion",
            "created": completion["created"],
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion["choices"][0]["text"],
                    },
                    "finish_reason": completion["choices"][0]["finish_reason"],
                }
            ],
            "usage": completion["usage"],
        }

    def iterator() -> Iterator:
        for i, chunk in enumerate(iterator_or_completion):
            if i == 0:
                yield json.dumps(
                    {
                        "id": _id,
                        "model": request.model,
                        "created": chunk["created"],
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            yield json.dumps(
                {
                    "id": _id,
                    "model": request.model,
                    "created": chunk["created"],
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk["choices"][0]["text"],
                            }
                            if chunk["choices"][0]["finish_reason"] is None else {},
                            "finish_reason": chunk["choices"][0]["finish_reason"],
                        }
                    ],
                },
                ensure_ascii=False,
            )

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
