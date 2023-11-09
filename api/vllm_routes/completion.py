import secrets
import time

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.completion import Completion, CompletionChoice
from openai.types.completion_usage import CompletionUsage
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from api.config import config
from api.models import VLLM_ENGINE
from api.routes.utils import check_api_key
from api.utils.protocol import CompletionCreateParams
from api.vllm_routes.utils import get_model_inputs

completion_router = APIRouter()


@completion_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionCreateParams, raw_request: Request):
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

    request.max_tokens = request.max_tokens or 256
    request_id = f"cmpl-{secrets.token_hex(12)}"

    token_ids, error_check_ret = await get_model_inputs(request, request.prompt, config.MODEL_NAME.lower())
    if error_check_ret is not None:
        return error_check_ret

    # stop settings
    _stop, stop_token_ids = [], []
    if VLLM_ENGINE.prompt_adapter.stop is not None:
        stop_token_ids = VLLM_ENGINE.prompt_adapter.stop.get("token_ids", [])
        _stop = VLLM_ENGINE.prompt_adapter.stop.get("strings", [])

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]
    request.stop = list(set(_stop + request.stop))

    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stop_token_ids=stop_token_ids,
            max_tokens=request.max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result_generator = VLLM_ENGINE.generate(
        request.prompt if isinstance(request.prompt, str) else None, sampling_params, request_id, token_ids,
    )

    # Streaming response
    if request.stream:
        generator = generate_completion_stream_generator(result_generator, request, request_id)
        return StreamingResponse(generator, media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await VLLM_ENGINE.abort(request_id)
            return
        final_res = res
        
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        output.text = output.text.replace("�", "")
        choice = CompletionChoice(
            index=output.index, text=output.text, finish_reason=output.finish_reason,
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
        id=request_id, choices=choices, created=int(time.time()),
        model=request.model, object="text_completion", usage=usage
    )


async def generate_completion_stream_generator(
    result_generator, request: CompletionCreateParams, request_id: str
):
    previous_texts = [""] * request.n
    previous_num_tokens = [0] * request.n
    async for res in result_generator:
        res: RequestOutput
        for output in res.outputs:
            i = output.index
            output.text = output.text.replace("�", "")
            delta_text = output.text[len(previous_texts[i]):]
            previous_texts[i] = output.text
            previous_num_tokens[i] = len(output.token_ids)

            choice = CompletionChoice(index=i, text=delta_text, finish_reason="stop")  # TODO: support for length
            chunk = Completion(
                id=request_id, choices=[choice], created=int(time.time()),
                model=request.model, object="text_completion",
            )
            yield f"data: {chunk.json(ensure_ascii=False)}\n\n"

            if output.finish_reason is not None:
                choice = CompletionChoice(index=i, text=delta_text, finish_reason="stop")  # TODO: support for length
                chunk = Completion(
                    id=request_id, choices=[choice], created=int(time.time()),
                    model=request.model, object="text_completion",
                )
                yield f"data: {chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
