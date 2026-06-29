import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response

from app.auth import require_bearer_token
from app.autoscaling import recommend_replicas
from app.batching_client import GatewayBatchingClient
from app.config import Settings, get_settings
from app.embeddings import EmbeddingsService
from app.metrics import (
    AUTOSCALING_ESTIMATED_DRAIN_TIME_SECONDS,
    AUTOSCALING_RECOMMENDED_REPLICAS,
    QUEUE_ITEMS,
    QUEUE_OLDEST_ITEM_AGE_SECONDS,
    QUEUE_TOKEN_BACKLOG,
    prometheus_content,
    prometheus_content_type,
)
from app.queue_inspection import QueueStats, inspect_embedding_queue
from app.redis_batch_queue import RedisBatchQueue
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
    version="0.3.2",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=prometheus_content(), media_type=prometheus_content_type())


@app.get("/debug/queues")
async def debug_queues(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    try:
        stats = await _collect_queue_stats(settings)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to inspect Redis queues: {exc}",
        ) from exc

    for item in stats.values():
        _set_queue_metrics(item)

    return {
        "model": settings.queue_inspection_model,
        "queues": {
            workload: {
                "items": item.items,
                "token_backlog": item.token_backlog,
                "oldest_item_age_ms": item.oldest_item_age_ms,
                "newest_item_age_ms": item.newest_item_age_ms,
                "malformed_items": item.malformed_items,
            }
            for workload, item in stats.items()
        },
    }


@app.get("/debug/autoscaling")
async def debug_autoscaling(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    try:
        queue_stats = await _collect_queue_stats(settings)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to inspect Redis queues: {exc}",
        ) from exc

    recommendations = {
        workload: _recommend_for_workload(settings, stats)
        for workload, stats in queue_stats.items()
    }

    for stats in queue_stats.values():
        _set_queue_metrics(stats)
    for recommendation in recommendations.values():
        AUTOSCALING_RECOMMENDED_REPLICAS.labels(
            model=recommendation.model,
            workload=recommendation.workload,
        ).set(recommendation.recommended_replicas)
        AUTOSCALING_ESTIMATED_DRAIN_TIME_SECONDS.labels(
            model=recommendation.model,
            workload=recommendation.workload,
        ).set(recommendation.estimated_drain_time_seconds)

    return {
        "model": settings.queue_inspection_model,
        "workloads": {
            workload: recommendation.model_dump()
            for workload, recommendation in recommendations.items()
        },
    }


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


async def _collect_queue_stats(settings: Settings) -> dict[str, QueueStats]:
    queue = RedisBatchQueue.from_url(settings.redis_url, settings.redis_key_prefix)
    try:
        await queue.ping()
        return {
            workload: await inspect_embedding_queue(
                queue,
                logical_model=settings.queue_inspection_model,
                workload=workload,
                max_scan_items=settings.queue_inspection_max_items,
            )
            for workload in ("query", "document")
        }
    finally:
        await queue.close()


def _set_queue_metrics(stats: QueueStats) -> None:
    labels = {"model": stats.model, "workload": stats.workload}
    QUEUE_ITEMS.labels(**labels).set(stats.items)
    QUEUE_TOKEN_BACKLOG.labels(**labels).set(stats.token_backlog)
    QUEUE_OLDEST_ITEM_AGE_SECONDS.labels(**labels).set(stats.oldest_item_age_ms / 1000)


def _recommend_for_workload(settings: Settings, stats: QueueStats):
    if stats.workload == "document":
        return recommend_replicas(
            model=stats.model,
            workload=stats.workload,
            token_backlog=stats.token_backlog,
            tokens_per_second_per_replica=settings.document_tokens_per_second_per_replica,
            current_replicas=settings.document_current_replicas,
            target_drain_time_seconds=settings.document_target_drain_time_seconds,
            min_replicas=settings.document_autoscale_min_replicas,
            max_replicas=settings.document_autoscale_max_replicas,
        )

    return recommend_replicas(
        model=stats.model,
        workload=stats.workload,
        token_backlog=stats.token_backlog,
        tokens_per_second_per_replica=settings.query_tokens_per_second_per_replica,
        current_replicas=settings.query_current_replicas,
        target_drain_time_seconds=settings.query_target_drain_time_seconds,
        min_replicas=settings.query_autoscale_min_replicas,
        max_replicas=settings.query_autoscale_max_replicas,
    )
