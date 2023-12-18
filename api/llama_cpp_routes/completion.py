from functools import partial
from typing import Iterator

import anyio
from fastapi import APIRouter, Depends, Request
from loguru import logger
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.core.llama_cpp_engine import LlamaCppEngine
from api.llama_cpp_routes.utils import get_llama_cpp_engine
from api.utils.compat import model_dump
from api.utils.protocol import CompletionCreateParams
from api.utils.request import (
    handle_request,
    check_api_key,
    get_event_publisher,
)

completion_router = APIRouter()


@completion_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_completion(
    request: CompletionCreateParams,
    raw_request: Request,
    engine: LlamaCppEngine = Depends(get_llama_cpp_engine),
):
    if isinstance(request.prompt, list):
        assert len(request.prompt) <= 1
        request.prompt = request.prompt[0] if len(request.prompt) > 0 else ""

    request.max_tokens = request.max_tokens or 256
    request = await handle_request(request, engine.stop)

    include = {
        "temperature",
        "top_p",
        "stream",
        "stop",
        "model",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
    }
    kwargs = model_dump(request, include=include)
    logger.debug(f"==== request ====\n{kwargs}")

    iterator_or_completion = await run_in_threadpool(engine.create_completion, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid, and we can use it to stream the response.
        def iterator() -> Iterator:
            yield first_response
            yield from iterator_or_completion

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
    else:
        return iterator_or_completion
