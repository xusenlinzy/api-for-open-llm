import time
import uuid
from functools import partial
from typing import (
    List,
    Dict,
    Any,
    AsyncIterator,
    Optional,
)

import anyio
from fastapi import APIRouter, Depends
from fastapi import HTTPException, Request
from loguru import logger
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice, Logprobs
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from vllm.outputs import RequestOutput

from api.models import GENERATE_ENGINE
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
    if request.echo:
        # We do not support echo since the vLLM engine does not
        # currently support getting the logprobs of prompt tokens.
        raise HTTPException(status_code=400, detail="echo is not currently supported")

    if request.suffix:
        # The language models we currently support do not support suffix.
        raise HTTPException(status_code=400, detail="suffix is not currently supported")

    request.max_tokens = request.max_tokens or 128
    request, stop_token_ids = await handle_request(request, engine.prompt_adapter.stop, chat=False)

    if isinstance(request.prompt, list):
        request.prompt = request.prompt[0]

    params = request.model_dump()
    params.update(dict(stop_token_ids=stop_token_ids, prompt_or_messages=request.prompt))
    logger.debug(f"==== request ====\n{params}")

    request_id: str = f"cmpl-{str(uuid.uuid4())}"
    generator = engine.generate(params, request_id)

    if request.stream:
        iterator = create_completion_stream(generator, params, request_id, engine.tokenizer)
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
    else:
        # Non-streaming response
        final_res: RequestOutput = None
        async for res in generator:
            if raw_request is not None:
                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await engine.model.abort(request_id)
                    return
            final_res = res

        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            output.text = output.text.replace("�", "")
            logprobs = None
            if params.get("logprobs", None) is not None:
                logprobs = create_logprobs(engine.tokenizer, output.token_ids, output.logprobs)

            choice = CompletionChoice(
                index=output.index,
                text=output.text,
                finish_reason=output.finish_reason,
                logprobs=logprobs,
            )
            choices.append(choice)

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return Completion(
            id=request_id,
            choices=choices,
            created=int(time.time()),
            model=params.get("model", "llm"),
            object="text_completion",
            usage=usage,
        )


def create_logprobs(
    tokenizer,
    token_ids: List[int],
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None,
    num_output_top_logprobs: Optional[int] = None,
    initial_text_offset: int = 0,
) -> Logprobs:
    logprobs = Logprobs(text_offset=[], token_logprobs=[], tokens=[], top_logprobs=None)
    last_token_len = 0
    if num_output_top_logprobs:
        logprobs.top_logprobs = []

    for i, token_id in enumerate(token_ids):
        step_top_logprobs = top_logprobs[i]
        if step_top_logprobs is not None:
            token_logprob = step_top_logprobs[token_id]
        else:
            token_logprob = None

        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(token_logprob)
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
        last_token_len = len(token)

        if num_output_top_logprobs:
            logprobs.top_logprobs.append(
                {
                    tokenizer.convert_ids_to_tokens(i): p
                    for i, p in step_top_logprobs.items()
                }
                if step_top_logprobs else None
            )
    return logprobs


async def create_completion_stream(
    generator: AsyncIterator, params: Dict[str, Any], request_id: str, tokenizer,
    ) -> AsyncIterator:
    n = params.get("n", 1)
    previous_texts = [""] * n
    previous_num_tokens = [0] * n
    async for res in generator:
        res: RequestOutput
        for output in res.outputs:
            i = output.index
            output.text = output.text.replace("�", "")
            delta_text = output.text[len(previous_texts[i]):]

            if params.get("logprobs", None) is not None:
                logprobs = create_logprobs(
                    tokenizer,
                    output.token_ids[previous_num_tokens[i]:],
                    output.logprobs[previous_num_tokens[i]:],
                    len(previous_texts[i])
                )
            else:
                logprobs = None

            previous_texts[i] = output.text
            previous_num_tokens[i] = len(output.token_ids)

            choice = CompletionChoice(
                index=i,
                text=delta_text,
                finish_reason="stop",
                logprobs=logprobs,
            )
            yield Completion(
                id=request_id,
                choices=[choice],
                created=int(time.time()),
                model=params.get("model", "llm"),
                object="text_completion",
            )

            if output.finish_reason is not None:
                if params.get("logprobs", None) is not None:
                    logprobs = Logprobs(
                        text_offset=[], token_logprobs=[], tokens=[], top_logprobs=[]
                    )
                else:
                    logprobs = None

                choice = CompletionChoice(
                    index=i,
                    text=delta_text,
                    finish_reason="stop",
                    logprobs=logprobs,
                )
                yield Completion(
                    id=request_id,
                    choices=[choice],
                    created=int(time.time()),
                    model=params.get("model", "llm"),
                    object="text_completion",
                    )
