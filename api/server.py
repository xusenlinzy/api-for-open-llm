import sys

sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import config
from api.routes import chat_router, completion_router, embedding_router
from api.utils.protocol import (
    ModelCard,
    ModelList,
    ModelPermission,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(f"{config.API_PREFIX}/models")
async def show_available_models():
    model_cards = []
    model_list = [config.MODEL_NAME]
    for m in model_list:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)

prefix = config.API_PREFIX
app.include_router(embedding_router, prefix=prefix, tags=["Embedding"])
if not config.ONLY_EMBEDDING:
    app.include_router(chat_router, prefix=prefix, tags=["Chat"])
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
