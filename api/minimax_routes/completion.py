from functools import partial
from typing import Iterator, Dict, Any

import anyio
from fastapi import (
    APIRouter,
    Depends,
    Request,
    HTTPException,
    status,
)
from loguru import logger
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.engine.minimax import MiniMaxEngine
from api.models import LLM_ENGINE
from api.protocol import CompletionCreateParams
from api.utils import (
    check_api_key,
    get_event_publisher,
)

completion_router = APIRouter()


def get_engine():
    yield LLM_ENGINE


@completion_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def create_completion(
    request: CompletionCreateParams,
    raw_request: Request,
    engine: MiniMaxEngine = Depends(get_engine),
):
    """Creates a text completion via MiniMax Cloud API."""
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

    if len(request.prompt) < 1:
        raise HTTPException(status_code=400, detail="Invalid request")

    request.max_tokens = request.max_tokens or 128

    kwargs: Dict[str, Any] = {
        "model": engine.model_name,
        "prompt": request.prompt[0],
        "temperature": engine.clamp_temperature(request.temperature),
        "max_tokens": request.max_tokens,
        "stream": request.stream or False,
    }
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stop:
        kwargs["stop"] = request.stop
    if request.frequency_penalty:
        kwargs["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty:
        kwargs["presence_penalty"] = request.presence_penalty
    if request.n and request.n > 1:
        kwargs["n"] = request.n

    logger.debug(f"==== MiniMax completion request ====\n{kwargs}")

    if request.stream:
        def _sync_stream() -> Iterator:
            stream = engine.client.completions.create(**kwargs)
            for chunk in stream:
                yield chunk

        iterator = _sync_stream()
        first_response = await run_in_threadpool(next, iterator)

        def iterator_with_first() -> Iterator:
            yield first_response
            yield from iterator

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator_with_first(),
            ),
        )
    else:
        response = await run_in_threadpool(
            engine.client.completions.create, **kwargs
        )
        return response
