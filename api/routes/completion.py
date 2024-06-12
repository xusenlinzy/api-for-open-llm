from functools import partial
from typing import Iterator

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

from api.common import dictify
from api.engine.hf import HuggingFaceEngine
from api.models import LLM_ENGINE
from api.protocol import CompletionCreateParams
from api.utils import (
    check_completion_requests,
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
    engine: HuggingFaceEngine = Depends(get_engine),
):
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

    if len(request.prompt) < 1:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await check_completion_requests(
        request,
        engine.template.stop,
        engine.template.stop_token_ids,
        chat=False,
    )
    request.max_tokens = request.max_tokens or 128

    params = dictify(request, exclude={"prompt"})
    params.update(dict(prompt_or_messages=request.prompt[0]))
    logger.debug(f"==== request ====\n{params}")

    iterator_or_completion = await run_in_threadpool(engine.create_completion, params)

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
