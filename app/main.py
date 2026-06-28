import logging

from fastapi import Depends, FastAPI

from app.auth import require_bearer_token
from app.config import Settings, get_settings
from app.embeddings import EmbeddingsService
from app.schemas import EmbeddingsRequest, EmbeddingsResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Voyage-Compatible Embedding Gateway",
    version="0.1.0",
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(
    request: EmbeddingsRequest,
    _: str = Depends(require_bearer_token),
    settings: Settings = Depends(get_settings),
) -> EmbeddingsResponse:
    service = EmbeddingsService(settings)
    return await service.create_embeddings(request)
