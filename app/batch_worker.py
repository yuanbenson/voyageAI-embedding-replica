from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from fastapi import HTTPException

from app.config import Settings, get_settings
from app.model_registry import resolve_model_route
from app.queue_models import EmbeddingResultItem, EmbeddingWorkItem
from app.redis_batch_queue import RedisBatchQueue, current_time_ms
from app.response_normalization import normalize_vllm_embedding_data
from app.vllm_client import VllmClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


async def run_batch_worker() -> None:
    settings = get_settings()
    route = resolve_model_route(settings.batch_worker_model, settings)
    queue = RedisBatchQueue.from_url(settings.redis_url, settings.redis_key_prefix)
    vllm = VllmClient(settings)

    await queue.ping()
    target_tokens = _target_tokens(settings)
    max_wait_ms = _max_wait_ms(settings)
    max_items = _max_items(settings)
    logger.info(
        "batch_worker_started",
        extra={
            "model": settings.batch_worker_model,
            "workload": settings.batch_worker_workload,
            "target_tokens": target_tokens,
            "max_wait_ms": max_wait_ms,
            "max_items": max_items,
        },
    )

    try:
        while True:
            await _wait_for_batch_window(
                queue,
                settings,
                settings.batch_worker_model,
                settings.batch_worker_workload,
            )

            items = await queue.claim_embedding_batch(
                logical_model=settings.batch_worker_model,
                workload=settings.batch_worker_workload,
                max_tokens=target_tokens,
                max_items=max_items,
            )

            if not items:
                await asyncio.sleep(0.005)
                continue

            await _process_batch(queue=queue, vllm=vllm, route=route, items=items, settings=settings)
    finally:
        await queue.close()


async def _wait_for_batch_window(
    queue: RedisBatchQueue,
    settings: Settings,
    logical_model: str,
    workload: str,
) -> None:
    oldest = await queue.peek_oldest_embedding_work(
        logical_model=logical_model,
        workload=workload,
    )
    if oldest is None:
        return

    now_ms = current_time_ms()
    if oldest.deadline_ms < now_ms:
        # Let the claim script drop expired head items immediately.
        return

    age_ms = max(0, now_ms - oldest.created_at_ms)
    remaining_wait_ms = _max_wait_ms(settings) - age_ms
    if remaining_wait_ms > 0:
        await asyncio.sleep(remaining_wait_ms / 1000)


async def _process_batch(
    *,
    queue: RedisBatchQueue,
    vllm: VllmClient,
    route,
    items: list[EmbeddingWorkItem],
    settings: Settings,
) -> None:
    workload = settings.batch_worker_workload
    batch_request_id = f"batch-{uuid4()}"
    batch_size = len(items)
    batch_tokens = sum(item.token_count for item in items)
    oldest_wait_ms = max(0, current_time_ms() - min(item.created_at_ms for item in items))

    logger.info(
        "claimed_%s_batch model=%s batch_size=%d batch_tokens=%d target_tokens=%d oldest_wait_ms=%d",
        workload,
        settings.batch_worker_model,
        batch_size,
        batch_tokens,
        _target_tokens(settings),
        oldest_wait_ms,
    )

    inputs = [item.input_text for item in items]
    try:
        vllm_response = await vllm.embed(
            route=route,
            inputs=inputs,
            request_id=batch_request_id,
        )
        data = normalize_vllm_embedding_data(vllm_response, expected_count=len(items))
    except HTTPException as exc:
        logger.exception(
            "%s_batch_failed_http_exception",
            workload,
            extra={"batch_request_id": batch_request_id, "batch_size": len(items)},
        )
        await _publish_batch_error(queue, items, str(exc.detail), exc.status_code, settings)
        return
    except Exception as exc:
        logger.exception(
            "%s_batch_failed",
            workload,
            extra={"batch_request_id": batch_request_id, "batch_size": len(items)},
        )
        await _publish_batch_error(queue, items, str(exc), 502, settings)
        return

    for item, embedding_data in zip(items, data, strict=True):
        await queue.publish_embedding_result(
            reply_to=item.reply_to,
            result=EmbeddingResultItem(
                request_id=item.request_id,
                parent_request_id=item.parent_request_id,
                input_index=item.input_index,
                ok=True,
                embedding=embedding_data.embedding,
                index=item.input_index,
            ),
            ttl_seconds=settings.result_queue_ttl_seconds,
        )

    logger.info(
        "completed_%s_batch model=%s batch_size=%d batch_tokens=%d",
        workload,
        settings.batch_worker_model,
        batch_size,
        batch_tokens,
    )


async def _publish_batch_error(
    queue: RedisBatchQueue,
    items: list[EmbeddingWorkItem],
    error: str,
    status_code: int,
    settings: Settings,
) -> None:
    for item in items:
        await queue.publish_embedding_result(
            reply_to=item.reply_to,
            result=EmbeddingResultItem(
                request_id=item.request_id,
                parent_request_id=item.parent_request_id,
                input_index=item.input_index,
                ok=False,
                error=error,
                status_code=status_code,
            ),
            ttl_seconds=settings.result_queue_ttl_seconds,
        )


def _target_tokens(settings: Settings) -> int:
    if settings.batch_worker_workload == "document":
        return settings.document_batch_target_tokens
    return settings.query_batch_target_tokens


def _max_wait_ms(settings: Settings) -> int:
    if settings.batch_worker_workload == "document":
        return settings.document_max_wait_ms
    return settings.query_max_wait_ms


def _max_items(settings: Settings) -> int:
    if settings.batch_worker_workload == "document":
        return settings.document_batch_max_items
    return settings.query_batch_max_items


def main() -> None:
    asyncio.run(run_batch_worker())


if __name__ == "__main__":
    main()
