import sys

sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import config
from api.models import EMBEDDED_MODEL, GENERATE_MDDEL
from api.routes import chat_router, completion_router, embedding_router, model_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


prefix = config.API_PREFIX
app.include_router(model_router, prefix=prefix, tags=["model"])
if EMBEDDED_MODEL is not None:
    app.include_router(embedding_router, prefix=prefix, tags=["Embedding"])
if GENERATE_MDDEL is not None:
    app.include_router(chat_router, prefix=prefix, tags=["Chat"])
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
