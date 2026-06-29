import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request

from app.auth import require_bearer_token
from app.batching_client import GatewayBatchingClient
from app.config import Settings, get_settings
from app.embeddings import EmbeddingsService
from app.schemas import EmbeddingsRequest, EmbeddingsResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    batching_client: GatewayBatchingClient | None = None

    if settings.enable_query_batching or settings.enable_document_batching:
        batching_client = GatewayBatchingClient(settings)
        await batching_client.start()

    app.state.batching_client = batching_client
    try:
        yield
    finally:
        if batching_client is not None:
            await batching_client.stop()


app = FastAPI(
    title="Voyage-Compatible Embedding Gateway",
    version="0.3.1",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(
    request: Request,
    body: EmbeddingsRequest,
    _: str = Depends(require_bearer_token),
    settings: Settings = Depends(get_settings),
) -> EmbeddingsResponse:
    service = EmbeddingsService(
        settings,
        batching_client=request.app.state.batching_client,
    )
    return await service.create_embeddings(body)
