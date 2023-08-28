import base64

import numpy as np
import tiktoken
from fastapi import APIRouter, Depends

from api.config import config
from api.models import EMBEDDED_MODEL
from api.routes.utils import check_api_key
from api.utils.protocol import (
    UsageInfo,
    EmbeddingsResponse,
    EmbeddingsRequest,
)

embedding_router = APIRouter()


@embedding_router.post("/embeddings", dependencies=[Depends(check_api_key)])
@embedding_router.post("/engines/{model_name}/embeddings", dependencies=[Depends(check_api_key)])
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
    if EMBEDDED_MODEL is not None:
        if "bge" in config.EMBEDDING_NAME.lower():
            instruction = ""
            if "zh" in config.EMBEDDING_NAME.lower():
                instruction = "为这个句子生成表示以用于检索相关文章："
            elif "en" in config.EMBEDDING_NAME.lower():
                instruction = "Represent this sentence for searching relevant passages: "
            inputs = [instruction + q for q in inputs]

    data, token_num = [], 0
    batches = [
        inputs[i: min(i + 1024, len(inputs))]
        for i in range(0, len(inputs), 1024)
    ]
    for num_batch, batch in enumerate(batches):
        token_num = sum([len(i) for i in batch])
        vecs = EMBEDDED_MODEL.encode(batch, normalize_embeddings=True)

        bs, dim = vecs.shape
        if config.EMBEDDING_SIZE is not None and config.EMBEDDING_SIZE > dim:
            zeros = np.zeros((bs, config.EMBEDDING_SIZE - dim))
            vecs = np.c_[vecs, zeros]

        if request.encoding_format == "base64":
            vecs = [base64.b64encode(v.tobytes()).decode("utf-8") for v in vecs]
        else:
            vecs = vecs.tolist()

        data += [
            {
                "object": "embedding",
                "embedding": emb,
                "index": num_batch * 1024 + i,
            }
            for i, emb in enumerate(vecs)
        ]
        token_num += token_num

    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)
