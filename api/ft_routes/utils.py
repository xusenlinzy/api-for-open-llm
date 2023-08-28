import secrets

from pydantic import BaseModel

from api.config import config
from api.utils.protocol import (
    CreateFineTuneRequest,
    FineTune,
    FineTuneEvent,
    FineTuneHyperparams,
    File,
)

FINETUNES_REPO = {}

WORKERS = {}


class FineTuneRepo(BaseModel):

    @staticmethod
    def get(finetune_id: str):
        return FINETUNES_REPO.get(finetune_id)

    @staticmethod
    def add(fine_tune: FineTune):
        FINETUNES_REPO[fine_tune.id] = fine_tune

    @staticmethod
    def get_all():
        return list(FINETUNES_REPO.values())


class FineTuneWorker(BaseModel):

    training_id: str
    fine_tune: FineTune

    @staticmethod
    def train(params: CreateFineTuneRequest):
        finetune_id = "ft-" + secrets.token_hex(12)

        # TODO: do something
        finetune = FineTune(
            id=finetune_id,
            model=config.MODEL_NAME,
            events=[
                FineTuneEvent(
                    object="fine-tune-event",
                    level="info",
                    message="Job enqueued. Waiting for jobs ahead to complete. Queue number: 0."
                )
            ],
            hyperparams=FineTuneHyperparams(
                batch_size=4,
                learning_rate_multiplier=0.1,
                n_epochs=4,
                prompt_loss_weight=0.1,
            ),
            organization_id="open-llm",
            result_files=[],
            status="pending",
            validation_files=[],
            training_files=[
                File(
                    id=params.training_file,
                    object="file",
                    bytes=1547276,
                    created_at=1610062281,
                    filename="my-data-train.jsonl",
                    purpose="fine-tune-train"
                )
            ]
        )

        FineTuneRepo.add(finetune)
        worker = FineTuneWorker(training_id='1', fine_tune=finetune)

        WORKERS[finetune.id] = worker

        return finetune

    @staticmethod
    def cancel(finetune_id: str):
        worker = WORKERS.pop(finetune_id)
        if not worker:
            raise Exception(f"Worker {id} not found!")

        # TODO: do something
        finetune = worker.fine_tune
        finetune.status = "cancelled"

        return finetune
