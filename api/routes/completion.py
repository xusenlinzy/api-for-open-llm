import asyncio
import json
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from openai.types.completion import Completion, CompletionChoice
from openai.types.completion_create_params import CompletionCreateParams
from openai.types.completion_usage import CompletionUsage

from api.models import GENERATE_MDDEL
from api.routes.utils import check_requests, create_error_response, check_api_key
from api.utils.protocol import (
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)

completion_router = APIRouter()


@completion_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionCreateParams, raw_request: Request):
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    start_time = time.time()
    prompt = request.get("prompt")
    if isinstance(prompt, str):
        if len(prompt) < 1:
            raise HTTPException(status_code=400, detail="Invalid request")
        prompt = [prompt]

    # stop settings
    _stop, stop_token_ids = [], []
    if GENERATE_MDDEL.stop is not None:
        stop_token_ids = GENERATE_MDDEL.stop.get("token_ids", [])
        _stop = GENERATE_MDDEL.stop.get("strings", [])

    stop = request.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    stop = list(set(_stop + stop))

    stream = request.get("stream")
    if stream:
        generator = generate_completion_stream_generator(request, raw_request)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        text_completions = []
        for text in prompt:
            gen_params = request.copy()
            gen_params.update(
                dict(
                    prompt=text,
                    max_tokens=request.get("max_tokens", 1024),
                    stop=stop,
                    stop_token_ids=stop_token_ids,
                )
            )
            for i in range(request.get("n")):
                content = GENERATE_MDDEL.generate_gate(gen_params)
                text_completions.append(content)

        choices = []
        usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        for i, content in enumerate(text_completions):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])

            choices.append(
                CompletionChoice(
                    index=i,
                    text=content["text"],
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )

            task_usage = CompletionUsage.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        logger.info(f"consume time  = {(time.time() - start_time)}s, response = {str(choices)}")
        return Completion(
            id=f"cmpl-{secrets.token_hex(12)}", choices=choices, created=int(time.time()),
            model=request.get("model"), object="text_completion", usage=usage
        )


async def generate_completion_stream_generator(request: CompletionRequest, raw_request: Request):
    model_name = request.model
    _id = f"cmpl-{secrets.token_hex(12)}"
    finish_stream_events = []

    for text in request.prompt:
        for i in range(request.n):
            previous_text = ""
            payload = dict(
                model=request.model,
                prompt=text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens or 1024,
                echo=request.echo,
                stream=request.stream,
                stop_token_ids=request.stop_token_ids,
                stop=request.stop,
                infilling=request.infilling,
                suffix_first=request.suffix_first,
                repetition_penalty=request.repetition_penalty,
            )

            for content in GENERATE_MDDEL.generate_stream_gate(payload):
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

                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text=delta_text,
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", None),
                )
                chunk = CompletionStreamResponse(
                    id=_id, object="text_completion", choices=[choice_data], model=model_name
                )
                if len(delta_text) == 0:
                    if content.get("finish_reason", None) is not None:
                        finish_stream_events.append(chunk)
                    continue

                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
