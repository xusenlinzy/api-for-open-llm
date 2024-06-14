import time
import traceback
import uuid
from functools import partial
from typing import AsyncIterator

import anyio
import vllm
from fastapi import APIRouter, Depends, status
from fastapi import HTTPException, Request
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from api.common import dictify, model_validate
from api.engine.vllm_engine import VllmEngine
from api.models import LLM_ENGINE
from api.protocol import Role, ChatCompletionCreateParams
from api.utils import (
    check_api_key,
    check_completion_requests,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")
vllm_version = vllm.__version__


def get_engine():
    yield LLM_ENGINE


@chat_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine: VllmEngine = Depends(get_engine),
):
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT.value:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await check_completion_requests(
        request,
        engine.template.stop,
        engine.template.stop_token_ids,
    )
    request.max_tokens = request.max_tokens or 512

    if request.best_of < request.n:
        request.best_of = request.n

    params = dictify(request, exclude={"messages"})
    params.update(dict(prompt_or_messages=request.messages, echo=False))
    logger.debug(f"==== request ====\n{params}")

    request_id: str = f"chatcmpl-{str(uuid.uuid4())}"
    token_ids = engine.template.convert_messages_to_ids(
        messages=request.messages,
        tools=request.tools,
        max_tokens=request.max_tokens,
    )

    result_generator = None
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

        # Todo: support for lora
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

        if vllm_version >= "0.4.3":
            result_generator = engine.model.generate(
                {
                    "prompt": None,
                    "prompt_token_ids": token_ids,
                },
                sampling_params,
                request_id,
                lora_request,
            )
        else:
            result_generator = engine.model.generate(
                None,
                sampling_params,
                request_id,
                token_ids,
                lora_request,
            )

    except ValueError as e:
        traceback.print_exc()

    if request.stream:
        iterator = create_chat_completion_stream(result_generator, request, request_id, engine)
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
        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                await engine.model.abort(request_id)
                return
            final_res = res

        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            output.text = output.text.replace("�", "")

            finish_reason = output.finish_reason
            function_call = None
            if request.functions or request.tools:
                try:
                    res, function_call = engine.template.parse_assistant_response(
                        output.text, request.tools or request.functions,
                    )
                    output.text = res
                except Exception as e:
                    traceback.print_exc()
                    logger.warning("Failed to parse tool call")

            if isinstance(function_call, dict) and "arguments" in function_call:
                function_call = FunctionCall(**function_call)
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    function_call=function_call
                )
                finish_reason = "function_call"
            elif isinstance(function_call, dict) and "function" in function_call:
                finish_reason = "tool_calls"
                tool_calls = [model_validate(ChatCompletionMessageToolCall, function_call)]
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    tool_calls=tool_calls,
                )
            else:
                message = ChatCompletionMessage(role="assistant", content=output.text.strip())

            choices.append(
                Choice(
                    index=output.index,
                    message=message,
                    finish_reason=finish_reason,
                )
            )

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        return ChatCompletion(
            id=request_id,
            choices=choices,
            created=int(time.time()),
            model=request.model,
            object="chat.completion",
            usage=usage,
        )


async def create_chat_completion_stream(
    generator: AsyncIterator,
    request: ChatCompletionCreateParams,
    request_id: str,
    engine: VllmEngine,
) -> AsyncIterator:
    for i in range(request.n):
        # First chunk with role
        choice = ChunkChoice(
            index=i,
            delta=ChoiceDelta(role="assistant", content=""),
            finish_reason=None,
            logprobs=None,
        )
        yield ChatCompletionChunk(
            id=request_id,
            choices=[choice],
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
        )

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                output.text = output.text.replace("�", "")

                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                finish_reason = output.finish_reason
                delta = None

                if finish_reason is None:
                    delta = ChoiceDelta(content=delta_text)
                elif request.functions or request.tools:
                    call_info = None
                    try:
                        res, call_info = engine.template.parse_assistant_response(
                            output.text, request.tools or request.functions,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        logger.warning("Failed to parse tool call")

                    if isinstance(call_info, dict) and "arguments" in call_info:
                        finish_reason = "function_call"
                        function_call = ChoiceDeltaFunctionCall(**call_info)
                        delta = ChoiceDelta(
                            role="assistant",
                            content=delta_text,
                            function_call=function_call
                        )
                    elif isinstance(call_info, dict) and "function" in call_info:
                        finish_reason = "tool_calls"
                        call_info["index"] = 0
                        tool_calls = [model_validate(ChoiceDeltaToolCall, call_info)]
                        delta = ChoiceDelta(
                            role="assistant",
                            content=delta_text,
                            tool_calls=tool_calls,
                        )
                
                choice = ChunkChoice(
                    index=i,
                    delta=delta or ChoiceDelta(content=delta_text),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
                yield ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=int(time.time()),
                    model=request.model,
                    object="chat.completion.chunk",
                )
