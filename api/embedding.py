import sys

sys.path.insert(0, '.')

import argparse
import json

import fastapi
import tiktoken
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from api.protocol import (
    UsageInfo,
    EmbeddingsResponse,
    EmbeddingsRequest,
)

app = fastapi.FastAPI()


@app.post("/v1/embeddings")
@app.post("/v1/engines/{model_name}/embeddings")
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name

    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]
    elif isinstance(inputs, list):
        if isinstance(inputs[0], int):
            decoding = tiktoken.model.encoding_for_model(request.model)
            inputs = [decoding.decode(inputs)]
        elif isinstance(inputs[0], list):
            decoding = tiktoken.model.encoding_for_model(request.model)
            inputs = [decoding.decode(text) for text in inputs]

    # https://huggingface.co/BAAI/bge-large-zh
    if embed_client is not None:
        if "bge" in args.embedding_name.lower():
            instruction = ""
            if "zh" in args.embedding_name.lower():
                instruction = "为这个句子生成表示以用于检索相关文章："
            elif "en" in args.embedding_name.lower():
                instruction = "Represent this sentence for searching relevant passages: "
            inputs = [instruction + q for q in inputs]

    data, token_num = [], 0
    batches = [
        inputs[i: min(i + 1024, len(inputs))]
        for i in range(0, len(inputs), 1024)
    ]
    for num_batch, batch in enumerate(batches):
        embedding = {
            "embedding": embed_client.encode(batch, normalize_embeddings=True).tolist(),
            "token_num": sum([len(i) for i in batch]),
        }

        data += [
            {
                "object": "embedding",
                "embedding": emb,
                "index": num_batch * 1024 + i,
            }
            for i, emb in enumerate(embedding["embedding"])
        ]
        token_num += embedding["token_num"]

    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenAI Compatible RESTful API server."
    )
    # fastapi related
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="host name"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="port number"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )

    # model related
    parser.add_argument(
        '--embedding_name', help='embedding model name or path', required=True, default=None
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    embed_client = SentenceTransformer(args.embedding_name)

    uvicorn.run(app, host=args.host, port=args.port)
