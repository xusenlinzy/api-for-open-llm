import time
import traceback
import uuid
from functools import partial
from typing import AsyncIterator, Tuple

import anyio
import vllm
from fastapi import APIRouter, Depends
from fastapi import Request
from loguru import logger
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice, Logprobs
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import merge_async_iterators

from api.common import dictify
from api.engine.vllm_engine import VllmEngine
from api.models import LLM_ENGINE
from api.protocol import CompletionCreateParams
from api.utils import (
    check_completion_requests,
    get_event_publisher,
    check_api_key,
)

completion_router = APIRouter()
vllm_version = vllm.__version__


def get_engine():
    yield LLM_ENGINE


def parse_prompt_format(prompt) -> Tuple[bool, list]:
    # get the prompt, openai supports the following
    # "a string, array of strings, array of tokens, or array of token arrays."
    prompt_is_tokens = False
    prompts = [prompt]  # case 1: a string
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")
        elif isinstance(prompt[0], str):
            prompt_is_tokens = False
            prompts = prompt  # case 2: array of strings
        elif isinstance(prompt[0], int):
            prompt_is_tokens = True
            prompts = [prompt]  # case 3: array of tokens
        elif isinstance(prompt[0], list) and isinstance(prompt[0][0], int):
            prompt_is_tokens = True
            prompts = prompt  # case 4: array of token arrays
        else:
            raise ValueError(
                "prompt must be a string, array of strings, array of tokens, or array of token arrays"
            )
    return prompt_is_tokens, prompts


@completion_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_completion(
    request: CompletionCreateParams,
    raw_request: Request,
    engine: VllmEngine = Depends(get_engine),
):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.
    """
    request.max_tokens = request.max_tokens or 128
    request = await check_completion_requests(
        request,
        engine.template.stop,
        engine.template.stop_token_ids,
        chat=False,
    )

    if isinstance(request.prompt, list):
        request.prompt = request.prompt[0]

    params = dictify(request, exclude={"prompt"})
    params.update(dict(prompt_or_messages=request.prompt))
    logger.debug(f"==== request ====\n{params}")

    request_id: str = f"cmpl-{str(uuid.uuid4())}"
    # Schedule the request and get the result generator.
    generators = []
    num_prompts = 1
    try:
        include = {
            "n",
            "presence_penalty",
            "frequency_penalty",
            "temperature",
            "top_p",
            "repetition_penalty",
            "min_p",
            "best_of",
            "ignore_eos",
            "use_beam_search",
            "skip_special_tokens",
            "spaces_between_special_tokens",
        }
        kwargs = dictify(request, include=include)
        sampling_params = SamplingParams(
            stop=request.stop or [],
            stop_token_ids=request.stop_token_ids or [],
            max_tokens=request.max_tokens,
            **kwargs,
        )
        lora_request = None

        try:
            from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor

            if vllm_version >= "0.4.2":
                decoding_config = await engine.model.get_decoding_config()
                guided_decode_logits_processor = (
                    await get_guided_decoding_logits_processor(
                        request.guided_decoding_backend or decoding_config.guided_decoding_backend,
                        request,
                        engine.tokenizer,
                    )
                )
            else:
                guided_decode_logits_processor = (
                    await get_guided_decoding_logits_processor(
                        request,
                        engine.tokenizer,
                    )
                )
            if guided_decode_logits_processor:
                sampling_params.logits_processors = sampling_params.logits_processors or []
                sampling_params.logits_processors.append(guided_decode_logits_processor)
        except ImportError:
            pass

        prompt_is_tokens, prompts = parse_prompt_format(request.prompt)
        num_prompts = len(prompts)

        for i, prompt in enumerate(prompts):
            if prompt_is_tokens:
                input_ids = prompt
            else:
                input_ids = engine.tokenizer(prompt).input_ids

            if vllm_version >= "0.4.3":
                generator = engine.model.generate(
                    {
                        "prompt": prompt if isinstance(prompt, str) else None,
                        "prompt_token_ids": input_ids,
                    },
                    sampling_params,
                    request_id,
                    lora_request,
                )
            else:
                generator = engine.model.generate(
                    prompt,
                    sampling_params,
                    f"{request_id}-{i}",
                    prompt_token_ids=input_ids,
                    lora_request=lora_request
                )

            generators.append(generator)
    except ValueError as e:
        traceback.print_exc()

    result_generator: AsyncIterator[Tuple[int, RequestOutput]] = merge_async_iterators(*generators)

    if request.stream:
        iterator = create_completion_stream(
            engine, result_generator, request, request_id, num_prompts
        )
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
        final_res_batch = [None] * num_prompts
        async for i, res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.model.abort(f"{request_id}-{i}")
            final_res_batch[i] = res

        choices = []
        num_prompt_tokens = 0
        num_generated_tokens = 0
        for final_res in final_res_batch:
            final_res: RequestOutput
            prompt_token_ids = final_res.prompt_token_ids
            prompt_logprobs = final_res.prompt_logprobs
            prompt_text = final_res.prompt

            for output in final_res.outputs:
                if request.echo and request.max_tokens == 0:
                    token_ids = prompt_token_ids
                    top_logprobs = prompt_logprobs
                    output_text = prompt_text
                elif request.echo and request.max_tokens > 0:
                    token_ids = prompt_token_ids + output.token_ids
                    top_logprobs = (prompt_logprobs + output.logprobs if request.logprobs else None)
                    output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    top_logprobs = output.logprobs
                    output_text = output.text

                if request.logprobs is not None:
                    logprobs = engine.create_completion_logprobs(
                        token_ids=token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=request.logprobs,
                    )
                else:
                    logprobs = None

                choice = CompletionChoice(
                    index=len(choices),
                    text=output_text,
                    finish_reason=output.finish_reason,
                    logprobs=logprobs,
                )
                choices.append(choice)

                num_prompt_tokens += len(prompt_token_ids)
                num_generated_tokens += sum(len(output.token_ids) for output in final_res.outputs)

        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return Completion(
            id=request_id,
            choices=choices,
            created=int(time.time()),
            model=request.model,
            object="text_completion",
            usage=usage,
        )


async def create_completion_stream(
    engine: VllmEngine,
    generator: AsyncIterator,
    request: CompletionCreateParams,
    request_id: str,
    num_prompts: int,
) -> AsyncIterator:
    previous_texts = [""] * request.n * num_prompts
    previous_num_tokens = [0] * request.n * num_prompts
    has_echoed = [False] * request.n * num_prompts
    try:
        async for prompt_idx, res in generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index + prompt_idx * request.n
                output.text = output.text.replace("�", "")

                if request.echo and request.max_tokens == 0:
                    # only return the prompt
                    delta_text = res.prompt
                    delta_token_ids = res.prompt_token_ids
                    top_logprobs = res.prompt_logprobs
                    has_echoed[i] = True
                elif request.echo and request.max_tokens > 0 and not has_echoed[i]:
                    # echo the prompt and first token
                    delta_text = res.prompt + output.text
                    delta_token_ids = res.prompt_token_ids + output.token_ids
                    top_logprobs = res.prompt_logprobs + (output.logprobs or [])
                    has_echoed[i] = True
                else:
                    # return just the delta
                    delta_text = output.text[len(previous_texts[i]):]
                    delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                    top_logprobs = output.logprobs[previous_num_tokens[i]:] if output.logprobs else None

                if request.logprobs is not None:
                    assert top_logprobs is not None, (
                        "top_logprobs must be provided when logprobs "
                        "is requested")
                    logprobs = engine.create_completion_logprobs(
                        token_ids=delta_token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=request.logprobs,
                        initial_text_offset=len(previous_texts[i]),
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
                    model=request.model,
                    object="text_completion",
                )

                if output.finish_reason is not None:
                    if request.logprobs is not None:
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
                        model=request.model,
                        object="text_completion",
                    )
    except:
        traceback.print_exc()
