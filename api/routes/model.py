from fastapi import APIRouter

from api.config import config
from api.utils.protocol import (
    ModelCard,
    ModelList,
    ModelPermission,
)

model_router = APIRouter()


@model_router.get("/models")
async def show_available_models():
    return ModelList(data=[ModelCard(id=config.MODEL_NAME, root=config.MODEL_NAME, permission=[ModelPermission()])])
