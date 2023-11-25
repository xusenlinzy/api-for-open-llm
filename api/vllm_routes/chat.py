import secrets
import time
from functools import partial
from typing import AsyncGenerator

import anyio
from fastapi import APIRouter, Depends, Request, HTTPException
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
from sse_starlette import EventSourceResponse
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from api.config import SETTINGS
from api.generation.chatglm import process_response_v3
from api.generation.qwen import parse_response
from api.utils.protocol import Role, ChatCompletionCreateParams
from api.utils.request import check_api_key
from api.utils.request import (
    handle_request,
    get_engine,
    get_event_publisher,
)
from api.vllm_routes.utils import get_gen_prompt, get_model_inputs

chat_router = APIRouter(prefix="/chat")


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine=Depends(get_engine),
):
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

    prompt = await get_gen_prompt(engine, request, SETTINGS.model_name.lower())
    request, stop_token_ids = await handle_request(request, engine.prompt_adapter.stop)

    request.max_tokens = request.max_tokens or 512
    token_ids, error_check_ret = await get_model_inputs(engine, request, prompt, SETTINGS.model_name.lower())
    if error_check_ret is not None:
        return error_check_ret

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

    result_generator = engine.generate(
        prompt if isinstance(prompt, str) else None, sampling_params, request_id, token_ids,
    )

    # Streaming response
    if request.stream:
        generator = chat_completion_stream_generator(result_generator, request, request_id)
        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=generator,
            ),
        )

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return
        final_res = res

    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        output.text = output.text.replace("�", "")

        finish_reason = output.finish_reason
        function_call = None
        if request.functions and "chatglm3" in SETTINGS.model_name.lower():
            try:
                function_call = process_response_v3(output.text, use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        elif request.functions and "qwen" in SETTINGS.model_name.lower():
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
) -> AsyncGenerator:
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
        yield chunk.model_dump_json()

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
                yield chunk.model_dump_json()

                if output.finish_reason is not None:
                    choice = ChunkChoice(index=i, delta=ChoiceDelta(), finish_reason="stop")
                    chunk = ChatCompletionChunk(
                        id=request_id, choices=[choice], created=int(time.time()),
                        model=request.model, object="chat.completion.chunk",
                    )
                    yield chunk.model_dump_json(exclude_none=True)
