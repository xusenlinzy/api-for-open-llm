import time
from typing import List

from fastapi import APIRouter, Depends
from openai.types.model import Model
from pydantic import BaseModel

from api.config import SETTINGS
from api.models import GENERATE_ENGINE
from api.utils.request import check_api_key

model_router = APIRouter()


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model] = []


available_models = ModelList(
    data=[
        Model(
            id=SETTINGS.model_name or "",
            object="model",
            created=int(time.time()),
            owned_by="open"
        )
    ]
)


@model_router.get("/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    return await GENERATE_ENGINE.show_available_models() if SETTINGS.engine == "vllm" else available_models


@model_router.get("/models/{model}", dependencies=[Depends(check_api_key)])
async def retrieve_model():
    return ModelList.data[0]
