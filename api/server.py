from api.config import SETTINGS
from api.models import app, EMBEDDED_MODEL, GENERATE_ENGINE
from api.routes import model_router


prefix = SETTINGS.api_prefix
app.include_router(model_router, prefix=prefix, tags=["Model"])

if EMBEDDED_MODEL is not None:
    from api.routes.embedding import embedding_router

    app.include_router(embedding_router, prefix=prefix, tags=["Embedding"])


if GENERATE_ENGINE is not None:
    if SETTINGS.engine == "vllm":
        from api.vllm_routes import chat_router as chat_router
        from api.vllm_routes import completion_router as completion_router

    elif SETTINGS.engine == "llama.cpp":
        from api.llama_cpp_routes import chat_router as chat_router
        from api.llama_cpp_routes import completion_router as completion_router

    else:
        from api.routes.chat import chat_router as chat_router
        from api.routes.completion import completion_router as completion_router

    app.include_router(chat_router, prefix=prefix, tags=["Chat"])
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")
