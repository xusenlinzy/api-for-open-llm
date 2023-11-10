import secrets
import time
from typing import Any
from typing import Generator

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.completion_usage import CompletionUsage
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from api.config import config
from api.generation.chatglm import process_response_v3
from api.generation.qwen import parse_response
from api.models import VLLM_ENGINE
from api.routes.utils import check_api_key
from api.utils.protocol import Role, ChatCompletionCreateParams
from api.vllm_routes.utils import get_gen_prompt, get_model_inputs

chat_router = APIRouter(prefix="/chat")


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionCreateParams, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    logger.info(f"Received chat messages: {request.messages}")

    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    prompt = await get_gen_prompt(request, config.MODEL_NAME.lower())
    request.max_tokens = request.max_tokens or 512
    token_ids, error_check_ret = await get_model_inputs(request, prompt, config.MODEL_NAME.lower())
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

    if "qwen" in config.MODEL_NAME.lower() and request.functions:
        request.stop.append("Observation:")
    request.stop = list(set(_stop + request.stop))

    request_id = f"chatcmpl-{secrets.token_hex(12)}"
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
        prompt if isinstance(prompt, str) else None, sampling_params, request_id, token_ids,
    )

    # Streaming response
    if request.stream:
        generator = chat_completion_stream_generator(result_generator, request, request_id)
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

        if isinstance(function_call, dict) and "arguments" in function_call:
            function_call = FunctionCall(**function_call)
            message = ChatCompletionMessage(
                role="assistant", content=output.text, function_call=function_call
            )
            finish_reason = "function_call"
        else:
            message = ChatCompletionMessage(role="assistant", content=output.text)

        choices.append(Choice(index=output.index, message=message, finish_reason=finish_reason))

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = CompletionUsage(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    return ChatCompletion(
        id=request_id, choices=choices, created=int(time.time()),
        model=request.model, object="chat.completion", usage=usage
    )


async def chat_completion_stream_generator(
    result_generator, request: ChatCompletionCreateParams, request_id: str
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    n = request.n
    for i in range(n):
        # First chunk with role
        choice = ChunkChoice(index=i, delta=ChoiceDelta(role="assistant", content=""), finish_reason=None)
        chunk = ChatCompletionChunk(
            id=request_id, choices=[choice], created=int(time.time()),
            model=request.model, object="chat.completion.chunk",
        )
        yield f"data: {chunk.json(ensure_ascii=False)}\n\n"

        previous_texts = [""] * n
        previous_num_tokens = [0] * n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                output.text = output.text.replace("�", "")
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                choice = ChunkChoice(index=i, delta=ChoiceDelta(content=delta_text), finish_reason=output.finish_reason)
                chunk = ChatCompletionChunk(
                    id=request_id, choices=[choice], created=int(time.time()),
                    model=request.model, object="chat.completion.chunk",
                )
                yield f"data: {chunk.json(ensure_ascii=False)}\n\n"

                if output.finish_reason is not None:
                    choice = ChunkChoice(index=i, delta=ChoiceDelta(), finish_reason="stop")
                    chunk = ChatCompletionChunk(
                        id=request_id, choices=[choice], created=int(time.time()),
                        model=request.model, object="chat.completion.chunk",
                    )
                    yield f"data: {chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"
