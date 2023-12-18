import time
import uuid
from functools import partial
from typing import (
    Dict,
    Any,
    AsyncIterator,
)

import anyio
from fastapi import APIRouter, Depends
from fastapi import Request
from loguru import logger
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from text_generation.types import Response, StreamResponse

from api.core.tgi import TGIEngine
from api.models import GENERATE_ENGINE
from api.utils.compat import model_dump
from api.utils.protocol import CompletionCreateParams
from api.utils.request import (
    handle_request,
    get_event_publisher,
    check_api_key
)

completion_router = APIRouter()


def get_engine():
    yield GENERATE_ENGINE


@completion_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_completion(
    request: CompletionCreateParams,
    raw_request: Request,
    engine: TGIEngine = Depends(get_engine),
):
    """ Completion API similar to OpenAI's API. """

    request.max_tokens = request.max_tokens or 128
    request = await handle_request(request, engine.prompt_adapter.stop, chat=False)

    if isinstance(request.prompt, list):
        request.prompt = request.prompt[0]

    request_id: str = f"cmpl-{str(uuid.uuid4())}"
    include = {
        "temperature",
        "best_of",
        "repetition_penalty",
        "typical_p",
        "watermark",
    }
    params = model_dump(request, include=include)
    params.update(
        dict(
            prompt=request.prompt,
            do_sample=request.temperature > 1e-5,
            max_new_tokens=request.max_tokens,
            stop_sequences=request.stop,
            top_p=request.top_p if request.top_p < 1.0 else 0.99,
            return_full_text=request.echo,
        )
    )
    logger.debug(f"==== request ====\n{params}")

    if request.stream:
        generator = engine.generate_stream(**params)
        iterator = create_completion_stream(generator, params, request_id)
        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator,
            ),
        )

    # Non-streaming response
    response: Response = await engine.generate(**params)

    finish_reason = response.details.finish_reason.value
    finish_reason = "length" if finish_reason == "length" else "stop"
    choice = CompletionChoice(
        index=0,
        text=response.generated_text,
        finish_reason=finish_reason,
        logprobs=None,
    )

    num_prompt_tokens = len(response.details.prefill)
    num_generated_tokens = response.details.generated_tokens
    usage = CompletionUsage(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )

    return Completion(
        id=request_id,
        choices=[choice],
        created=int(time.time()),
        model=params.get("model", "llm"),
        object="text_completion",
        usage=usage,
    )


async def create_completion_stream(
    generator: AsyncIterator[StreamResponse], params: Dict[str, Any], request_id: str,
) -> AsyncIterator[Completion]:
    async for output in generator:
        output: StreamResponse
        if output.token.special:
            continue

        choice = CompletionChoice(
            index=0,
            text=output.token.text,
            finish_reason="stop",
            logprobs=None,
        )
        yield Completion(
            id=request_id,
            choices=[choice],
            created=int(time.time()),
            model=params.get("model", "llm"),
            object="text_completion",
        )
