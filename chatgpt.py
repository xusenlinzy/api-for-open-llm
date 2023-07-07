import argparse
import asyncio
import concurrent.futures
import json
import traceback
import warnings
from typing import List, Optional, Dict, Union, Any

import openai
import uvicorn
from backoff import on_exception, expo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 2048
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None


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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    res = await _chat_completions_create_async(request.dict(exclude_none=True))

    async def chat_generator():
        if res is None:
            yield "WebServerError: SomethingWrongInOpenaiGptApi"
            return

        for openai_object in res:
            yield json.dumps(openai_object.to_dict_recursive(), ensure_ascii=False, separators=(",", ":"))

        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(chat_generator, media_type="text/event-stream")
    else:
        return res.to_dict_recursive() if res else {}


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    res = await _completions_create_async(request.dict(exclude_none=True))

    async def completion_generator():
        if res is None:
            yield "WebServerError: SomethingWrongInOpenaiGptApi"
            return

        for openai_object in res:
            yield json.dumps(openai_object.to_dict_recursive(), ensure_ascii=False, separators=(",", ":"))

        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(completion_generator, media_type="text/event-stream")
    else:
        return res.to_dict_recursive() if res else {}


@app.post("/v1/embeddings")
@app.post("/v1/engines/{model_name}/embeddings")
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name

    res = await _embeddings_create_async(request.dict(exclude_none=True))
    return res if res else {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple API server for ChatGPT')
    parser.add_argument('--api_key', type=str, help='API KEY', default=None, required=True)
    parser.add_argument('--host', '-H', type=str, help='host name', default='0.0.0.0')
    parser.add_argument('--port', '-P', type=int, help='port number', default=80)

    args = parser.parse_args()
    openai.api_key = args.api_key

    uvicorn.run(app, host=args.host, port=args.port)
