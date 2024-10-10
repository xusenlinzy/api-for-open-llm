from api.config import SETTINGS
from api.models import (
    app,
    EMBEDDING_MODEL,
    LLM_ENGINE,
    RERANK_MODEL,
)


prefix = SETTINGS.api_prefix

if EMBEDDING_MODEL is not None:
    from api.routes.embedding import embedding_router

    app.include_router(embedding_router, prefix=prefix, tags=["Embedding"])

    try:
        from api.routes.file import file_router

        app.include_router(file_router, prefix=prefix, tags=["File"])
    except ImportError:
        pass

if RERANK_MODEL is not None:
    from api.routes.rerank import rerank_router

    app.include_router(rerank_router, prefix=prefix, tags=["Rerank"])


if LLM_ENGINE is not None:
    from api.routes import model_router

    app.include_router(model_router, prefix=prefix, tags=["Model"])

    if SETTINGS.engine == "vllm":
        from api.vllm_routes import chat_router as chat_router
        from api.vllm_routes import completion_router as completion_router

    else:
        from api.routes.chat import chat_router as chat_router
        from api.routes.completion import completion_router as completion_router

    app.include_router(chat_router, prefix=prefix, tags=["Chat Completion"])
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")
