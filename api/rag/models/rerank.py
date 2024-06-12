from abc import ABC, abstractmethod
from typing import (
    List,
    Dict,
    Any,
    Optional,
)

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from api.protocol import (
    DocumentObj,
    Document,
    RerankResponse,
)


class BaseReranker(ABC):
    @abstractmethod
    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = 256,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = False,
    ) -> Dict[str, Any]:
        ...


class RAGReranker(BaseReranker):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = None,
    ) -> None:
        self.client = CrossEncoder(
            model_name_or_path,
            device=device,
            trust_remote_code=True,
        )
        logger.info(f"Loading from `{model_name_or_path}`.")

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = 256,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = False,
        **kwargs: Any,
    ) -> Optional[RerankResponse]:
        results = self.client.rank(
            query=query,
            documents=documents,
            top_k=top_n,
            return_documents=True,
            batch_size=batch_size,
            **kwargs,
        )

        if return_documents:
            docs = [
                DocumentObj(
                    index=int(res["corpus_id"]),
                    relevance_score=float(res["score"]),
                    document=Document(text=res["text"]),
                )
                for res in results
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(res["corpus_id"]),
                    relevance_score=float(res["score"]),
                    document=None,
                )
                for res in results
            ]
        return RerankResponse(results=docs)
