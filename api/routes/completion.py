import json
import secrets
import time
from typing import Optional, Union, Dict, List, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger

from api.models import GENERATE_MDDEL
from api.routes.utils import check_requests, create_error_response
from api.utils.protocol import (
    ChatMessage,
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    UsageInfo,
)
from api.utils.protocol import CompletionResponse, CompletionResponseChoice

completion_router = APIRouter()


@completion_router.post("/completions")
async def create_completion(request: CompletionRequest):
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret
    start_time = time.time()
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

    if request.stream:
        generator = generate_completion_stream_generator(request)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        text_completions = []
        for text in request.prompt:
            gen_params = get_gen_params(
                request.model,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                echo=request.echo,
                stream=request.stream,
                stop=request.stop,
            )
            for i in range(request.n):
                content = GENERATE_MDDEL.generate_gate(gen_params)
                text_completions.append(content)

        choices = []
        usage = UsageInfo()
        for i, content in enumerate(text_completions):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])

            choices.append(
                CompletionResponseChoice(
                    index=i,
                    text=content["text"],
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )

            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        logger.info(f"consume time  = {(time.time() - start_time)}s, response = {str(choices)}")
        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
        )


def get_gen_params(
    model_name: str,
    messages: Union[str, List[ChatMessage]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    echo: Optional[bool],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]] = None,
    with_function_call: Optional[bool] = False,
) -> Dict[str, Any]:
    if not max_tokens:
        max_tokens = 1024

    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
        "with_function_call": with_function_call,
    }

    if GENERATE_MDDEL.stop is not None:
        if "token_ids" in GENERATE_MDDEL.stop:
            gen_params["stop_token_ids"] = GENERATE_MDDEL.stop["token_ids"]

        if "strings" in GENERATE_MDDEL.stop:
            gen_params["stop"] = GENERATE_MDDEL.stop["strings"]

    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]

        gen_params["stop"] = gen_params["stop"] + stop if "stop" in gen_params else stop
        gen_params["stop"] = list(set(gen_params["stop"]))

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


async def generate_completion_stream_generator(request: CompletionRequest):
    model_name = request.model
    _id = f"cmpl-{secrets.token_hex(12)}"
    finish_stream_events = []

    for text in request.prompt:
        for i in range(request.n):
            previous_text = ""
            payload = get_gen_params(
                request.model,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                echo=request.echo,
                stream=request.stream,
                stop=request.stop,
            )

            for content in GENERATE_MDDEL.generate_stream_gate(payload):
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
