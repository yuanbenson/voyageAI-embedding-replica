from __future__ import annotations

import asyncio
import logging

from prometheus_client import start_http_server

from app.autoscaling import recommend_replicas
from app.config import Settings, get_settings
from app.metrics import (
    AUTOSCALING_ESTIMATED_DRAIN_TIME_SECONDS,
    AUTOSCALING_RECOMMENDED_REPLICAS,
    QUEUE_ITEMS,
    QUEUE_OLDEST_ITEM_AGE_SECONDS,
    QUEUE_TOKEN_BACKLOG,
)
from app.queue_inspection import QueueStats, inspect_embedding_queue
from app.redis_batch_queue import RedisBatchQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


async def run_metrics_exporter() -> None:
    settings = get_settings()
    start_http_server(settings.metrics_exporter_port)
    logger.info(
        "metrics_exporter_started model=%s port=%d refresh_seconds=%.3f",
        settings.queue_inspection_model,
        settings.metrics_exporter_port,
        settings.metrics_exporter_refresh_seconds,
    )

    while True:
        try:
            await refresh_metrics(settings)
        except Exception:
            logger.exception("metrics_exporter_refresh_failed")
        await asyncio.sleep(settings.metrics_exporter_refresh_seconds)


async def refresh_metrics(settings: Settings) -> None:
    queue = RedisBatchQueue.from_url(settings.redis_url, settings.redis_key_prefix)
    try:
        await queue.ping()
        for workload in ("query", "document"):
            stats = await inspect_embedding_queue(
                queue,
                logical_model=settings.queue_inspection_model,
                workload=workload,
                max_scan_items=settings.queue_inspection_max_items,
            )
            _set_queue_metrics(stats)
            _set_autoscaling_metrics(settings, stats)
    finally:
        await queue.close()


def _set_queue_metrics(stats: QueueStats) -> None:
    labels = {"model": stats.model, "workload": stats.workload}
    QUEUE_ITEMS.labels(**labels).set(stats.items)
    QUEUE_TOKEN_BACKLOG.labels(**labels).set(stats.token_backlog)
    QUEUE_OLDEST_ITEM_AGE_SECONDS.labels(**labels).set(stats.oldest_item_age_ms / 1000)


def _set_autoscaling_metrics(settings: Settings, stats: QueueStats) -> None:
    if stats.workload == "document":
        recommendation = recommend_replicas(
            model=stats.model,
            workload=stats.workload,
            token_backlog=stats.token_backlog,
            tokens_per_second_per_replica=settings.document_tokens_per_second_per_replica,
            current_replicas=settings.document_current_replicas,
            target_drain_time_seconds=settings.document_target_drain_time_seconds,
            min_replicas=settings.document_autoscale_min_replicas,
            max_replicas=settings.document_autoscale_max_replicas,
        )
    else:
        recommendation = recommend_replicas(
            model=stats.model,
            workload=stats.workload,
            token_backlog=stats.token_backlog,
            tokens_per_second_per_replica=settings.query_tokens_per_second_per_replica,
            current_replicas=settings.query_current_replicas,
            target_drain_time_seconds=settings.query_target_drain_time_seconds,
            min_replicas=settings.query_autoscale_min_replicas,
            max_replicas=settings.query_autoscale_max_replicas,
        )

    labels = {"model": recommendation.model, "workload": recommendation.workload}
    AUTOSCALING_RECOMMENDED_REPLICAS.labels(**labels).set(recommendation.recommended_replicas)
    AUTOSCALING_ESTIMATED_DRAIN_TIME_SECONDS.labels(**labels).set(
        recommendation.estimated_drain_time_seconds
    )


if __name__ == "__main__":
    asyncio.run(run_metrics_exporter())
