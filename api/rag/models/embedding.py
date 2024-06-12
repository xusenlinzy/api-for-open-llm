import base64
import os
from abc import ABC
from typing import (
    List,
    Literal,
    Optional,
    Sequence,
)

import numpy as np
from openai.types.create_embedding_response import Usage
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings

from api.protocol import CreateEmbeddingResponse, Embedding


class BaseEmbedding(ABC):
    def embed(
        self,
        texts: Sequence[str],
        model: Optional[str] = "bce",
        encoding_format: Literal["float", "base64"] = "float",
    ) -> CreateEmbeddingResponse:
        ...


class RAGEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = None,
    ) -> None:
        self.client = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=True,
        )

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = "bce",
        encoding_format: Literal["float", "base64"] = "float",
        dimensions: Optional[int] = -1,
    ) -> CreateEmbeddingResponse:
        dim = self.client.get_sentence_embedding_dimension()
        use_matryoshka = bool(0 < dimensions < dim)

        data, total_tokens = [], 0
        batches = [texts[i: i + 1024] for i in range(0, len(texts), 1024)]
        for num_batch, batch in enumerate(batches):
            vecs = self.client.encode(
                batch,
                batch_size=int(os.getenv("batch_size", 32)),
                normalize_embeddings=False if use_matryoshka else True,
                convert_to_tensor=True if use_matryoshka else False,
            )

            bs = vecs.shape[0]
            if dimensions > dim:
                zeros = np.zeros((bs, dimensions - dim))
                vecs = np.c_[vecs, zeros]
            elif 0 < dimensions < dim:
                vecs = vecs[..., :dimensions]  # Shrink the embedding dimensions
                vecs = normalize_embeddings(vecs).cpu().numpy()

            if encoding_format == "base64":
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
            total_tokens += sum(len(i) for i in batch)

        return CreateEmbeddingResponse(
            data=data,
            model=model,
            object="list",
            usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )
