import argparse
import asyncio
import concurrent.futures
import json
import traceback
import warnings

import openai
import uvicorn
from backoff import on_exception, expo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from api.utils.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    ModelPermission,
    EmbeddingsRequest,
)

warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
openai.api_key = "xxx"

# 根据模型名称选择不同的模型接口
MODEL_LIST = {
    "chatglm": {
        "api_base": "http://192.168.x.xx:8000/v1",
        "model_names":  # 如果模型名称在下面的列表中，则使用上面的 api_base
        [
            "chatglm",
            "chatglm-6b",
            # "gpt-3.5-turbo"  # 对于 ChatGPT-Next-Web 和 dify 等应用，指定模型名称为 gpt-3.5-turbo，因此需要加上
        ]
    },
    "chatglm2": {
        "api_base": "http://192.168.x.xx:8001/v1",
        "model_names":
        [
            "chatglm2",
            "chatglm2-6b",
        ]
    },
    "internlm": {
        "api_base": "http://192.168.x.xx:8002/v1",
        "model_names":
        [
            "internlm",
            "internlm-chat-7b"
        ]
    },
    "baichuan-13b": {
        "api_base": "http://192.168.x.xx:8003/v1",
        "model_names":
        [
            "baichuan",
            "baichuan-chat-13b"
        ]
    }
}

MODEL_NAME_MAP = {name: m for m, v in MODEL_LIST.items() for name in v["model_names"]}


@on_exception(expo, openai.error.RateLimitError, max_tries=5)
def _chat_completions_create(params):
    return openai.ChatCompletion.create(**params)


@on_exception(expo, openai.error.RateLimitError, max_tries=5)
def _completions_create(params):
    return openai.Completion.create(**params)


@on_exception(expo, openai.error.RateLimitError, max_tries=5)
def _embeddings_create(params):
    return openai.Embedding.create(**params)


async def _chat_completions_create_async(params):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                executor, _chat_completions_create, params
            )
        except:
            err = traceback.format_exc()
            logger.error(err)
            return None
    return result


async def _completions_create_async(params):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                executor, _completions_create, params
            )
        except:
            err = traceback.format_exc()
            logger.error(err)
            return None
    return result


async def _embeddings_create_async(params):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                executor, _embeddings_create, params
            )
        except:
            err = traceback.format_exc()
            logger.error(err)
            return None
    return result


@app.get("/v1/models")
async def show_available_models():
    model_cards = []
    model_list = MODEL_LIST.keys()
    for m in model_list:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def create_chat_completions(request: ChatCompletionRequest):
    assert request.model in MODEL_NAME_MAP.keys(), f"Model {request.model} not launched!"
    model_name = MODEL_NAME_MAP[request.model]
    openai.api_base = MODEL_LIST[model_name]["api_base"]
    res = await _chat_completions_create_async(request.dict(exclude_none=True))

    async def chat_generator():
        if res is None:
            yield "WebServerError: SomethingWrongInOpenaiGptApi"
            return

        for openai_object in res:
            yield f"{json.dumps(openai_object.to_dict_recursive(), ensure_ascii=False, separators=(',', ':'))}"

        yield "[DONE]"

    if request.stream:
        return EventSourceResponse(chat_generator(), media_type="text/event-stream")
    else:
        return res.to_dict_recursive() if res else {}


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    assert request.model in MODEL_NAME_MAP.keys(), f"Model {request.model} not launched!"
    model_name = MODEL_NAME_MAP[request.model]
    openai.api_base = MODEL_LIST[model_name]["api_base"]
    res = await _completions_create_async(request.dict(exclude_none=True))

    async def completion_generator():
        if res is None:
            yield "WebServerError: SomethingWrongInOpenaiGptApi"
            return

        for openai_object in res:
            yield f"{json.dumps(openai_object.to_dict_recursive(), ensure_ascii=False, separators=(',', ':'))}"

        yield "[DONE]"

    if request.stream:
        return EventSourceResponse(completion_generator(), media_type="text/event-stream")
    else:
        return res.to_dict_recursive() if res else {}


@app.post("/v1/embeddings")
@app.post("/v1/engines/{model_name}/embeddings")
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name

    assert request.model in MODEL_NAME_MAP.keys(), f"Model {request.model} not launched!"
    model_name = MODEL_NAME_MAP[request.model]
    openai.api_base = MODEL_LIST[model_name]["api_base"]
    res = await _embeddings_create_async(request.dict(exclude_none=True))
    return res if res else {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple API server for ChatGPT')
    parser.add_argument('--host', '-H', type=str, help='host name', default='0.0.0.0')
    parser.add_argument('--port', '-P', type=int, help='port number', default=9009)

    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
