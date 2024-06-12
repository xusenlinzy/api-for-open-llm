import tiktoken
from fastapi import APIRouter, Depends, status

from api.config import SETTINGS
from api.models import EMBEDDING_MODEL
from api.protocol import EmbeddingCreateParams
from api.rag import RAGEmbedding
from api.utils import check_api_key

embedding_router = APIRouter()


def get_embedding_engine():
    yield EMBEDDING_MODEL


@embedding_router.post(
    "/embeddings",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
@embedding_router.post(
    "/engines/{model_name}/embeddings",
)
async def create_embeddings(
    request: EmbeddingCreateParams,
    model_name: str = None,
    client: RAGEmbedding = Depends(get_embedding_engine),
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

    request.dimensions = request.dimensions or getattr(SETTINGS, "embedding_size", -1)

    return client.embed(
        texts=request.input,
        model=request.model,
        encoding_format=request.encoding_format,
        dimensions=request.dimensions,
    )
