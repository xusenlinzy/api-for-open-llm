import asyncio
import json
import secrets
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from openai.types.completion import Completion, CompletionChoice
from openai.types.completion_usage import CompletionUsage

from api.models import GENERATE_MDDEL
from api.routes.utils import check_requests, create_error_response, check_api_key
from api.utils.protocol import CompletionCreateParams

completion_router = APIRouter()


@completion_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionCreateParams, raw_request: Request):
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    start_time = time.time()
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

    if len(request.prompt) < 1:
        raise HTTPException(status_code=400, detail="Invalid request")

    # stop settings
    _stop, stop_token_ids = [], []
    if GENERATE_MDDEL.stop is not None:
        stop_token_ids = GENERATE_MDDEL.stop.get("token_ids", [])
        _stop = GENERATE_MDDEL.stop.get("strings", [])

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]
    request.stop = list(set(_stop + request.stop))

    if request.stream:
        generator = generate_completion_stream_generator(request, raw_request, stop_token_ids)
        return StreamingResponse(generator, media_type="text/event-stream")

    text_completions = []
    for text in request.prompt:
        gen_params = request.dict()
        gen_params.update(
            dict(
                prompt=text,
                max_tokens=request.max_tokens or 256,
                stop_token_ids=stop_token_ids,
            )
        )
        for i in range(request.n):
            content = GENERATE_MDDEL.generate_gate(gen_params)
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

        task_usage = CompletionUsage.parse_obj(content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    logger.info(f"consume time  = {(time.time() - start_time)}s, response = {str(choices)}")
    return Completion(
        id=f"cmpl-{secrets.token_hex(12)}", choices=choices, created=int(time.time()),
        model=request.model, object="text_completion", usage=usage
    )


async def generate_completion_stream_generator(
    request: CompletionCreateParams, raw_request: Request, stop_token_ids: List[int],
):
    _id = f"cmpl-{secrets.token_hex(12)}"
    finish_stream_events = []
    for text in request.prompt:
        for i in range(request.n):
            previous_text = ""
            gen_params = request.dict()
            gen_params.update(
                dict(
                    prompt=text,
                    max_tokens=request.max_tokens or 256,
                    stop_token_ids=stop_token_ids,
                )
            )
            for content in GENERATE_MDDEL.generate_stream_gate(gen_params):
                if await raw_request.is_disconnected():
                    asyncio.current_task().cancel()
                    return

                if content["error_code"] != 0:
                    yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                decoded_unicode = content["text"].replace("\ufffd", "")
                delta_text = decoded_unicode[len(previous_text):]
                previous_text = decoded_unicode

                choice = CompletionChoice(index=i, text=delta_text, finish_reason="stop")  # TODO: support for length
                chunk = Completion(
                    id=_id, choices=[choice], created=int(time.time()),
                    model=request.model, object="text_completion",
                )

                if len(delta_text) == 0:
                    if content.get("finish_reason", None) is not None:
                        finish_stream_events.append(chunk)
                    continue

                yield f"data: {chunk.json(ensure_ascii=False)}\n\n"

    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
