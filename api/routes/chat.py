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
from api.protocol import ChatCompletionCreateParams, Role
from api.utils import (
    check_completion_requests,
    check_api_key,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")


def get_engine():
    yield LLM_ENGINE


@chat_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine: HuggingFaceEngine = Depends(get_engine),
):
    """Creates a completion for the chat message"""
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT.value:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await check_completion_requests(
        request,
        engine.template.stop,
        engine.template.stop_token_ids,
    )
    request.max_tokens = request.max_tokens or 1024

    params = dictify(request, exclude={"messages"})
    params.update(dict(prompt_or_messages=request.messages, echo=False))
    logger.debug(f"==== request ====\n{params}")

    iterator_or_completion = await run_in_threadpool(engine.create_chat_completion, params)

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
