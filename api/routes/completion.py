import json
import secrets
import time
from functools import partial
from typing import List, Iterator, Generator, Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from openai.types.completion import Completion, CompletionChoice
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.utils.protocol import CompletionCreateParams
from api.utils.request import (
    handle_request,
    create_error_response,
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
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

    if len(request.prompt) < 1:
        raise HTTPException(status_code=400, detail="Invalid request")

    request, stop_token_ids = await handle_request(request, engine.stop, chat=False)
    request.max_tokens = request.max_tokens or 128

    start_time = time.time()
    if request.stream:
        generator = await run_in_threadpool(generate_completion_stream_generator, engine, request, stop_token_ids)
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

    text_completions = []
    for text in request.prompt:
        gen_params = request.model_dump()
        gen_params.update(
            dict(
                prompt=text,
                stop_token_ids=stop_token_ids,
            )
        )
        for i in range(request.n):
            content = await run_in_threadpool(engine.generate_gate, gen_params)
            text_completions.append(content)

    choices = []
    usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    indexes = [i for _ in range(len(request.prompt)) for i in range(request.n)]
    for content, i in zip(text_completions, indexes):
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])

        choices.append(
            CompletionChoice(
                index=i,
                text=content["text"],
                logprobs=content.get("logprobs", None),
                finish_reason="stop",  # TODO: support for length
            )
        )

        task_usage = CompletionUsage.model_validate(content["usage"])
        for usage_key, usage_value in task_usage.model_dump().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    logger.info(f"consume time  = {(time.time() - start_time)}s, response = {str(choices)}")
    return Completion(
        id=f"cmpl-{secrets.token_hex(12)}", choices=choices, created=int(time.time()),
        model=request.model, object="text_completion", usage=usage
    )


def generate_completion_stream_generator(
    engine, request: CompletionCreateParams, stop_token_ids: List[int],
) -> Generator[str, Any, None]:
    _id = f"cmpl-{secrets.token_hex(12)}"
    finish_stream_events = []
    for text in request.prompt:
        for i in range(request.n):
            previous_text = ""
            gen_params = request.model_dump()
            gen_params.update(
                dict(
                    prompt=text,
                    stop_token_ids=stop_token_ids,
                )
            )
            for content in engine.generate_stream_gate(gen_params):
                if content["error_code"] != 0:
                    yield json.dumps(content, ensure_ascii=False)
                    return

                decoded_unicode = content["text"].replace("\ufffd", "")
                delta_text = decoded_unicode[len(previous_text):]
                previous_text = decoded_unicode

                choice = CompletionChoice(
                    index=i,
                    text=delta_text,
                    finish_reason="stop",  # TODO: support for length
                    logprobs=None,
                )
                chunk = Completion(
                    id=_id, choices=[choice], created=int(time.time()),
                    model=request.model, object="text_completion",
                )

                if len(delta_text) == 0:
                    if content.get("finish_reason", None) is not None:
                        finish_stream_events.append(chunk)
                    continue

                yield chunk.model_dump_json()

    for finish_chunk in finish_stream_events:
        yield finish_chunk.model_dump_json(exclude_none=True)
