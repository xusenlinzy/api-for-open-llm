from api.config import config
from api.models import EMBEDDED_MODEL, GENERATE_MDDEL, app, VLLM_ENGINE
from api.routes import model_router


prefix = config.API_PREFIX
app.include_router(model_router, prefix=prefix, tags=["Model"])

if EMBEDDED_MODEL is not None:
    from api.routes import embedding_router

    app.include_router(embedding_router, prefix=prefix, tags=["Embedding"])

if GENERATE_MDDEL is not None:
    from api.routes import chat_router, completion_router

    app.include_router(chat_router, prefix=prefix, tags=["Chat"])
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])

elif VLLM_ENGINE is not None:
    from api.vllm_routes import chat_router, completion_router

    app.include_router(chat_router, prefix=prefix, tags=["Chat"])
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
