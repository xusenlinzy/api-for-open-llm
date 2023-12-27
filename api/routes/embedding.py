import asyncio
import base64
from typing import Union

import numpy as np
import tiktoken
from fastapi import APIRouter, Depends
from openai import AsyncOpenAI
from openai.types.create_embedding_response import Usage
from sentence_transformers import SentenceTransformer

from api.config import SETTINGS
from api.models import EMBEDDED_MODEL
from api.utils.protocol import EmbeddingCreateParams, Embedding, CreateEmbeddingResponse
from api.utils.request import check_api_key

embedding_router = APIRouter()


def get_embedding_engine():
    yield EMBEDDED_MODEL


@embedding_router.post("/embeddings", dependencies=[Depends(check_api_key)])
@embedding_router.post("/engines/{model_name}/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(
    request: EmbeddingCreateParams,
    model_name: str = None,
    client: Union[SentenceTransformer, AsyncOpenAI] = Depends(get_embedding_engine),
):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name

    request.input = request.input
    if isinstance(request.input, str):
        request.input = [request.input]
    elif isinstance(request.input, list):
        if isinstance(request.input[0], int):
            decoding = tiktoken.model.encoding_for_model(request.model)
            request.input = [decoding.decode(request.input)]
        elif isinstance(request.input[0], list):
            decoding = tiktoken.model.encoding_for_model(request.model)
            request.input = [decoding.decode(text) for text in request.input]

    data, total_tokens = [], 0

    # support for tei: https://github.com/huggingface/text-embeddings-inference
    if isinstance(client, AsyncOpenAI):
        global_batch_size = SETTINGS.max_concurrent_requests * SETTINGS.max_client_batch_size
        for i in range(0, len(request.input), global_batch_size):
            tasks = []
            texts = request.input[i: i + global_batch_size]
            for j in range(0, len(texts), SETTINGS.max_client_batch_size):
                tasks.append(
                    client.embeddings.create(
                        input=[text[:510] for text in texts[j: j + SETTINGS.max_client_batch_size]],
                        model=request.model,
                    )
                )
            res = await asyncio.gather(*tasks)

            vecs = np.asarray([e.embedding for r in res for e in r.data])
            bs, dim = vecs.shape
            if SETTINGS.embedding_size > dim:
                zeros = np.zeros((bs, SETTINGS.embedding_size - dim))
                vecs = np.c_[vecs, zeros]

            if request.encoding_format == "base64":
                vecs = [base64.b64encode(v.tobytes()).decode("utf-8") for v in vecs]
            else:
                vecs = vecs.tolist()

            data.extend(
                Embedding(
                    index=i * global_batch_size + j,
                    object="embedding",
                    embedding=embed
                )
                for j, embed in enumerate(vecs)
            )
            total_tokens += sum(r.usage.total_tokens for r in res)
    else:
        batches = [request.input[i: i + 1024] for i in range(0, len(request.input), 1024)]
        for num_batch, batch in enumerate(batches):
            token_num = sum(len(i) for i in batch)
            vecs = client.encode(batch, normalize_embeddings=True)

            bs, dim = vecs.shape
            if SETTINGS.embedding_size > dim:
                zeros = np.zeros((bs, SETTINGS.embedding_size - dim))
                vecs = np.c_[vecs, zeros]

            if request.encoding_format == "base64":
                vecs = [base64.b64encode(v.tobytes()).decode("utf-8") for v in vecs]
            else:
                vecs = vecs.tolist()

            data.extend(
                Embedding(
                    index=num_batch * 1024 + i,
                    object="embedding",
                    embedding=embedding,
                )
                for i, embedding in enumerate(vecs)
            )
            total_tokens += token_num

    return CreateEmbeddingResponse(
        data=data,
        model=request.model,
        object="list",
        usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
