from fastapi import APIRouter, Depends, status

from api.models import RERANK_MODEL
from api.protocol import RerankRequest
from api.rag import RAGReranker
from api.utils import check_api_key

rerank_router = APIRouter()


def get_embedding_engine():
    yield RERANK_MODEL


@rerank_router.post(
    "/rerank",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def create_rerank(request: RerankRequest, client: RAGReranker = Depends(get_embedding_engine)):
    return client.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
        return_documents=request.return_documents,
    )
