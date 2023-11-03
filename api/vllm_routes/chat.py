import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from api.config import config
from api.generation.chatglm import process_response_v3
from api.generation.qwen import parse_response
from api.models import VLLM_ENGINE
from api.routes.utils import check_api_key
from api.utils.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
    Role,
    FunctionCallResponse,
)
from api.vllm_routes.utils import get_gen_prompt, get_model_inputs

chat_router = APIRouter(prefix="/chat")


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    logger.info(f"Received chat messages: {request.messages}")

    if len(request.messages) < 1 or request.messages[-1].role == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    prompt = await get_gen_prompt(request, config.MODEL_NAME.lower())
    request.max_tokens = request.max_tokens or 512
    token_ids, error_check_ret = await get_model_inputs(request, prompt, config.MODEL_NAME.lower())
    if error_check_ret is not None:
        return error_check_ret

    # stop settings
    stop, stop_token_ids = [], []
    if VLLM_ENGINE.prompt_adapter.stop is not None:
        stop_token_ids = VLLM_ENGINE.prompt_adapter.stop.get("token_ids", [])
        stop = VLLM_ENGINE.prompt_adapter.stop.get("strings", [])

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]
    request.stop = list(set(stop + request.stop))

    request.stop_token_ids = request.stop_token_ids or []
    request.stop_token_ids = list(set(stop_token_ids + request.stop_token_ids))

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result_generator = VLLM_ENGINE.generate(
        prompt if isinstance(prompt, str) else None,
        sampling_params,
        request_id,
        token_ids,
    )

    def create_stream_response_json(
        index: int,
        delta: DeltaMessage,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=delta,
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role=Role.ASSISTANT),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                choices=[choice_data],
                model=model_name
            )
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                output.text = output.text.replace("�", "")  # TODO: fix qwen decode
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                delta = DeltaMessage(content=delta_text, role=Role.ASSISTANT)
                response_json = create_stream_response_json(index=i, delta=delta, finish_reason=output.finish_reason)
                yield f"data: {response_json}\n\n"

                if output.finish_reason is not None:
                    response_json = create_stream_response_json(
                        index=i,
                        delta=DeltaMessage(content="", role=Role.ASSISTANT),
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"

        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(), media_type="text/event-stream")

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
        output.text = output.text.replace("�", "")  # TODO: fix qwen decode

        finish_reason = output.finish_reason
        function_call = None
        if request.functions and "chatglm3" in config.MODEL_NAME.lower():
            try:
                function_call = process_response_v3(output.text, use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        elif request.functions and "qwen" in config.MODEL_NAME.lower():
            res, function_call = parse_response(output.text)
            output.text = res

        if isinstance(function_call, dict):
            finish_reason = "function_call"
            function_call = FunctionCallResponse(**function_call)

        message = ChatMessage(
            role=Role.ASSISTANT,
            content=output.text,
            function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
        )

        choices.append(
            ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                finish_reason=finish_reason,
            )
        )

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )

    return ChatCompletionResponse(
        id=request_id, created=created_time, model=model_name, choices=choices, usage=usage
    )
