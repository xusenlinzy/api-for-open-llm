from fastapi import APIRouter, HTTPException, Depends

from api.ft_routes.utils import FineTuneRepo, FineTuneWorker
from api.routes.utils import check_api_key
from api.utils.protocol import (
    CreateFineTuneRequest,
    FineTune,
    ListFineTuneEventsResponse,
    ListFineTunesResponse,
)

finetune_router = APIRouter(prefix="/fine_tuning/jobs")


@finetune_router.post("", response_model=FineTune, dependencies=[Depends(check_api_key)])
async def create_finetune(request: CreateFineTuneRequest):
    return FineTuneWorker.train(request)


@finetune_router.get("", response_model=ListFineTunesResponse, dependencies=[Depends(check_api_key)])
async def list_finetunes():
    return ListFineTunesResponse(data=FineTuneRepo.get_all())


@finetune_router.get("/{finetune_id}", response_model=FineTune, dependencies=[Depends(check_api_key)])
async def retrieve_finetune(finetune_id: str):
    finetune = FineTuneRepo.get(finetune_id)
    if finetune:
        return finetune
    else:
        raise HTTPException(status_code=404, detail=f"Fine-tune {finetune_id} not found!")


@finetune_router.post("/{finetune_id}/cancel", response_model=FineTune, dependencies=[Depends(check_api_key)])
async def cancel_finetune(finetune_id: str):
    if FineTuneRepo.get(finetune_id) is None:
        raise HTTPException(status_code=404, detail=f"Fine-tune {finetune_id} not found!")

    try:
        return FineTuneWorker.cancel(finetune_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Fine-tune {finetune_id} not found!")


@finetune_router.get("/{finetune_id}/events", response_model=ListFineTuneEventsResponse, dependencies=[Depends(check_api_key)])
async def list_finetune_events(finetune_id: str):
    finetune = await retrieve_finetune(finetune_id)
    return ListFineTuneEventsResponse(data=finetune.events)
