from fastapi import APIRouter, Depends

from api.config import config
from api.routes.utils import check_api_key
from api.utils.protocol import (
    ModelCard,
    ModelList,
    ModelPermission,
)

model_router = APIRouter()


@model_router.get("/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    return ModelList(
        data=[
            ModelCard(
                id=config.MODEL_NAME,
                root=config.MODEL_NAME,
                permission=[ModelPermission()]
            )
        ]
    )
