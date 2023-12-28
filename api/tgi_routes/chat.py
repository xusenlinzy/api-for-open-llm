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
from fastapi import HTTPException, Request
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from text_generation.types import StreamResponse, Response

from api.core.tgi import TGIEngine
from api.models import GENERATE_ENGINE
from api.utils.compat import model_dump
from api.utils.protocol import Role, ChatCompletionCreateParams
from api.utils.request import (
    check_api_key,
    handle_request,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")


def get_engine():
    yield GENERATE_ENGINE


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine: TGIEngine = Depends(get_engine),
):
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await handle_request(request, engine.prompt_adapter.stop)
    request.max_tokens = request.max_tokens or 512

    prompt = engine.apply_chat_template(request.messages)
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
            prompt=prompt,
            do_sample=request.temperature > 1e-5,
            max_new_tokens=request.max_tokens,
            stop_sequences=request.stop,
            top_p=request.top_p if request.top_p < 1.0 else 0.99,
        )
    )
    logger.debug(f"==== request ====\n{params}")

    request_id: str = f"chatcmpl-{str(uuid.uuid4())}"

    if request.stream:
        generator = engine.generate_stream(**params)
        iterator = create_chat_completion_stream(generator, params, request_id)
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

    response: Response = await engine.generate(**params)
    finish_reason = response.details.finish_reason.value
    finish_reason = "length" if finish_reason == "length" else "stop"

    message = ChatCompletionMessage(role="assistant", content=response.generated_text)

    choice = Choice(
        index=0,
        message=message,
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
    return ChatCompletion(
        id=request_id,
        choices=[choice],
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        usage=usage,
    )


async def create_chat_completion_stream(
    generator: AsyncIterator[StreamResponse], params: Dict[str, Any], request_id: str
) -> AsyncIterator[ChatCompletionChunk]:
    # First chunk with role
    choice = ChunkChoice(
        index=0,
        delta=ChoiceDelta(role="assistant", content=""),
        finish_reason=None,
        logprobs=None,
    )
    yield ChatCompletionChunk(
        id=request_id,
        choices=[choice],
        created=int(time.time()),
        model=params.get("model", "llm"),
        object="chat.completion.chunk",
    )
    async for output in generator:
        output: StreamResponse
        if output.token.special:
            continue

        choice = ChunkChoice(
            index=0,
            delta=ChoiceDelta(content=output.token.text),
            finish_reason=None,
            logprobs=None,
        )
        yield ChatCompletionChunk(
            id=request_id,
            choices=[choice],
            created=int(time.time()),
            model=params.get("model", "llm"),
            object="chat.completion.chunk",
        )

    choice = ChunkChoice(
        index=0,
        delta=ChoiceDelta(),
        finish_reason="stop",
        logprobs=None,
    )
    yield ChatCompletionChunk(
        id=request_id,
        choices=[choice],
        created=int(time.time()),
        model=params.get("model", "llm"),
        object="chat.completion.chunk",
    )
