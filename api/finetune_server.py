from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import config
from api.ft_routes import file_router, finetune_router
from api.routes import model_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


prefix = config.API_PREFIX
app.include_router(model_router, prefix=prefix, tags=["Model"])
app.include_router(file_router, prefix=prefix, tags=["File"])
app.include_router(finetune_router, prefix=prefix, tags=["Finetune"])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
