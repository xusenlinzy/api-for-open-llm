import time
from http import HTTPStatus
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from api.apapter.react import (
    check_function_call,
    build_function_call_messages,
    build_chat_message,
    build_delta_message,
)
from api.config import config
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
)
from api.vllm_routes.utils import create_error_response, get_gen_prompt, get_model_inputs

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

    if len(request.messages) < 1 or request.messages[-1].role != Role.USER:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "Invalid request: messages is empty"
        )

    with_function_call = check_function_call(request.messages, functions=request.functions)
    if with_function_call and "qwen" not in config.MODEL_NAME.lower():
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "Invalid request format: functions only supported by Qwen-7B-Chat",
        )

    if with_function_call:
        if request.functions is None:
            for message in request.messages:
                if message.functions is not None:
                    request.functions = message.functions
                    break

        request.messages = build_function_call_messages(
            request.messages,
            request.functions,
            request.function_call,
        )

    prompt = await get_gen_prompt(request, config.MODEL_NAME.lower())
    request.max_tokens = request.max_tokens or 512
    token_ids, error_check_ret = await get_model_inputs(request, prompt, config.MODEL_NAME.lower())
    if error_check_ret is not None:
        return error_check_ret

    # stop settings
    stop = []
    if VLLM_ENGINE.prompt_adapter.stop is not None:
        stop = VLLM_ENGINE.prompt_adapter.stop.get("strings", [])

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]
    request.stop = list(set(stop + request.stop))

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = VLLM_ENGINE.generate(
        prompt if isinstance(prompt, str) else None,
        sampling_params,
        request_id,
        token_ids,
    )

    async def abort_request() -> None:
        await VLLM_ENGINE.abort(request_id)

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
        found_action_name = False
        with_function_call = request.functions is not None
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                output.text = output.text.replace("�", "")  # TODO: fix qwen decode
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                msgs = []
                if with_function_call:
                    if found_action_name:
                        if previous_texts[i].rfind("\nObserv") > 0:
                            break
                        msgs.append(build_delta_message(delta_text, "arguments"))
                        finish_reason = "function_call"
                    else:
                        if previous_texts[i].rfind("\nFinal Answer:") > 0:
                            with_function_call = False

                        if previous_texts[i].rfind("\nAction Input:") == -1:
                            continue
                        else:
                            msgs.append(build_delta_message(previous_texts[i]))
                            pos = previous_texts[i].rfind("\nAction Input:") + len("\nAction Input:")
                            msgs.append(build_delta_message(previous_texts[i][pos:], "arguments"))

                            found_action_name = True
                            finish_reason = "function_call"
                else:
                    msgs = [DeltaMessage(content=delta_text)]
                    finish_reason = output.finish_reason

                for m in msgs:
                    response_json = create_stream_response_json(index=i, delta=m, finish_reason=finish_reason)
                    yield f"data: {response_json}\n\n"

                if output.finish_reason is not None:
                    response_json = create_stream_response_json(
                        index=i,
                        delta=DeltaMessage(content=""),
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"

        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(
            completion_stream_generator(),
            media_type="text/event-stream",
            background=background_tasks,
        )

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST, "Client disconnected")
        final_res = res

    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        output.text = output.text.replace("�", "")  # TODO: fix qwen decode

        finish_reason = output.finish_reason
        if with_function_call:
            message, finish_reason = build_chat_message(output.text, request.functions)
        else:
            message = ChatMessage(role=Role.ASSISTANT, content=output.text)

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
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming, but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(), media_type="text/event-stream")

    return response
