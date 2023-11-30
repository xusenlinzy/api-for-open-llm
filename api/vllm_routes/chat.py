import time
import traceback
import uuid
from functools import partial
from typing import (
    Dict,
    Any,
    AsyncIterator,
)

import anyio
from fastapi import APIRouter, Depends
from fastapi import HTTPException, Request
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
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from vllm.outputs import RequestOutput

from api.models import GENERATE_ENGINE
from api.utils.protocol import Role, ChatCompletionCreateParams
from api.utils.request import check_api_key
from api.utils.request import (
    handle_request,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")


def get_engine():
    yield GENERATE_ENGINE


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine=Depends(get_engine),
):
    logger.info(f"Received chat messages: {request.messages}")

    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    request, stop_token_ids = await handle_request(request, engine.prompt_adapter.stop)
    request.max_tokens = request.max_tokens or 512

    params = request.model_dump(exclude={"messages"})
    params.update(
        dict(
            prompt_or_messages=request.messages,
            echo=False,
            stop_token_ids=stop_token_ids,
        )
    )
    logger.debug(f"==== request ====\n{params}")

    request_id: str = f"chatcmpl-{str(uuid.uuid4())}"
    generator = engine.generate(params, request_id)

    if request.stream:
        iterator = create_chat_completion_stream(generator, params, request_id)
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
        functions = params.get("functions", None)
        tools = params.get("tools", None)
        for output in final_res.outputs:
            output.text = output.text.replace("�", "")

            finish_reason = output.finish_reason
            function_call = None
            if functions or tools:
                try:
                    res, function_call = engine.prompt_adapter.parse_assistant_response(
                        output.text, functions, tools,
                    )
                    output.text = res
                except:
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
                tool_calls = [ChatCompletionMessageToolCall.model_validate(function_call)]
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    tool_calls=tool_calls,
                )
            else:
                message = ChatCompletionMessage(role="assistant", content=output.text)

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


async def create_chat_completion_stream(generator: AsyncIterator, params: Dict[str, Any], request_id: str) -> AsyncIterator:
    n = params.get("n", 1)
    for i in range(n):
        # First chunk with role
        choice = ChunkChoice(
            index=i,
            delta=ChoiceDelta(role="assistant", content=""),
            finish_reason=None,
        )
        yield ChatCompletionChunk(
            id=request_id,
            choices=[choice],
            created=int(time.time()),
            model=params.get("model", "llm"),
            object="chat.completion.chunk",
        )

        previous_texts = [""] * n
        previous_num_tokens = [0] * n
        async for res in generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                output.text = output.text.replace("�", "")

                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                choice = ChunkChoice(
                    index=i,
                    delta=ChoiceDelta(content=delta_text),
                    finish_reason=output.finish_reason,
                )
                yield ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=int(time.time()),
                    model=params.get("model", "llm"),
                    object="chat.completion.chunk",
                )

                if output.finish_reason is not None:
                    choice = ChunkChoice(
                        index=i,
                        delta=ChoiceDelta(),
                        finish_reason="stop",
                    )
                    yield ChatCompletionChunk(
                        id=request_id,
                        choices=[choice],
                        created=int(time.time()),
                        model=params.get("model", "llm"),
                        object="chat.completion.chunk",
                    )
