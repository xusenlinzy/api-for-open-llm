import json
from functools import partial
from typing import Iterator

import anyio
import llama_cpp
from fastapi import APIRouter, Depends, Request, HTTPException
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.utils.protocol import CompletionCreateParams
from api.utils.request import (
    handle_request,
    check_api_key,
    get_engine,
    get_event_publisher,
)

completion_router = APIRouter()


@completion_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_completion(
    request: CompletionCreateParams,
    raw_request: Request,
    engine=Depends(get_engine),
):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - echo (since the vLLM engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """
    if isinstance(request.prompt, list):
        assert len(request.prompt) <= 1
        request.prompt = request.prompt[0] if len(request.prompt) > 0 else ""

    if request.echo:
        # We do not support echo since the vLLM engine does not
        # currently support getting the logprobs of prompt tokens.
        raise HTTPException(status_code=400, detail="echo is not currently supported")

    if request.suffix:
        # The language models we currently support do not support suffix.
        raise HTTPException(status_code=400, detail="suffix is not currently supported")

    request.max_tokens = request.max_tokens or 256
    request, stop_token_ids = await handle_request(request, engine.prompt_adapter.stop)

    gen_kwargs = request.dict(exclude={"n", "user", "best_of"})
    iterator_or_completion = await run_in_threadpool(engine, **gen_kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.ChatCompletionChunk]:
            yield json.dumps(first_response, ensure_ascii=False)
            for part in iterator_or_completion:
                yield json.dumps(part, ensure_ascii=False)

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
        )
    else:
        return iterator_or_completion
