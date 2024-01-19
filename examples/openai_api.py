from typing import Optional

from fastapi import Depends, HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI
from sse_starlette import EventSourceResponse

from api.utils.protocol import ChatCompletionCreateParams, CompletionCreateParams, EmbeddingCreateParams

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEYS = None  # 此处设置允许访问接口的api_key列表
# 此处设置模型和接口地址的对应关系
MODEL_LIST = {
    "chat":
        {
            # 模型名称
            "qwen-7b-chat": {
                "addtional_names": ["gpt-3.5-turbo"],  # 其他允许访问该模型的名称，比如chatgpt-next-web使用gpt-3.5-turbo，则需要加入到此处
                "base_url": "http://192.168.20.59:7891/v1",  # 实际访问该模型的接口地址
                "api_key": "xxx"
            },
            # 模型名称
            "baichuan2-13b": {
                "addtional_names": [],  # 其他允许访问该模型的名称
                "base_url": "http://192.168.20.44:7860/v1",  # 实际访问该模型的接口地址
                "api_key": "xxx"
            },
            # 需要增加其他模型，仿照上面的例子添加即可
        },
    "completion":
        {
            "sqlcoder": {
                "addtional_names": [],  # 其他允许访问该模型的名称
                "base_url": "http://192.168.20.59:7892/v1",  # 实际访问该模型的接口地址
                "api_key": "xxx"
            },
            # 需要增加其他模型，仿照上面的例子添加即可
        },
    "embedding":
        {
            "base_url": "http://192.168.20.59:8001/v1",  # 实际访问该模型的接口地址
            "api_key": "xxx",  # api_key
        },
}

CHAT_MODEL_MAP = {am: name for name, detail in MODEL_LIST["chat"].items() for am in (detail["addtional_names"] + [name])}
COMPLETION_MODEL_MAP = {am: name for name, detail in MODEL_LIST["completion"].items() for am in (detail["addtional_names"] + [name])}


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):
    if API_KEYS is not None:
        if auth is None or (token := auth.credentials) not in API_KEYS:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionCreateParams):
    if request.model not in CHAT_MODEL_MAP:
        raise HTTPException(status_code=404, detail="Invalid model")

    model = CHAT_MODEL_MAP[request.model]
    client = AsyncOpenAI(
        api_key=MODEL_LIST["chat"][model]["api_key"],
        base_url=MODEL_LIST["chat"][model]["base_url"],
    )

    params = request.dict(
        exclude_none=True,
        include={
            "messages",
            "model",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
            "stream",
        }
    )
    response = await client.chat.completions.create(**params)

    async def chat_completion_stream_generator():
        async for chunk in response:
            yield chunk.json()
        yield "[DONE]"

    if request.stream:
        return EventSourceResponse(chat_completion_stream_generator())

    return response


@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionCreateParams):
    if request.model not in COMPLETION_MODEL_MAP:
        raise HTTPException(status_code=404, detail="Invalid model")

    model = COMPLETION_MODEL_MAP[request.model]
    client = AsyncOpenAI(
        api_key=MODEL_LIST["completion"][model]["api_key"],
        base_url=MODEL_LIST["completion"][model]["base_url"],
    )

    params = request.dict(
        exclude_none=True,
        include={
            "prompt",
            "model",
            "best_of",
            "echo",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "suffix",
            "temperature",
            "top_p",
            "user",
            "stream",
        }
    )
    response = await client.completions.create(**params)

    async def generate_completion_stream_generator():
        async for chunk in response:
            yield chunk.json()
        yield "[DONE]"

    if request.stream:
        return EventSourceResponse(generate_completion_stream_generator())

    return response


@app.post("/v1/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(request: EmbeddingCreateParams):
    client = AsyncOpenAI(
        api_key=MODEL_LIST["embedding"]["api_key"],
        base_url=MODEL_LIST["embedding"]["base_url"],
    )
    embeddings = await client.embeddings.create(**request.dict(exclude_none=True))
    return embeddings


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9009, log_level="info")
