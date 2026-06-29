from __future__ import annotations

from pydantic import BaseModel, Field

from app.queue_models import EmbeddingWorkItem
from app.redis_batch_queue import RedisBatchQueue, current_time_ms


class QueueStats(BaseModel):
    model: str
    workload: str
    items: int = Field(ge=0)
    token_backlog: int = Field(ge=0)
    oldest_item_age_ms: int = Field(ge=0)
    newest_item_age_ms: int = Field(ge=0)
    malformed_items: int = Field(ge=0)


async def inspect_embedding_queue(
    queue: RedisBatchQueue,
    *,
    logical_model: str,
    workload: str,
    max_scan_items: int | None = None,
) -> QueueStats:
    raw_items = await queue.list_embedding_work_raw(
        logical_model=logical_model,
        workload=workload,
        max_items=max_scan_items,
    )
    now_ms = current_time_ms()
    token_backlog = 0
    created_times: list[int] = []
    malformed_items = 0

    for raw in raw_items:
        try:
            item = EmbeddingWorkItem.model_validate_json(raw)
        except Exception:
            malformed_items += 1
            continue
        token_backlog += item.token_count
        created_times.append(item.created_at_ms)

    if created_times:
        oldest_item_age_ms = max(0, now_ms - min(created_times))
        newest_item_age_ms = max(0, now_ms - max(created_times))
    else:
        oldest_item_age_ms = 0
        newest_item_age_ms = 0

    return QueueStats(
        model=logical_model,
        workload=workload,
        items=len(raw_items),
        token_backlog=token_backlog,
        oldest_item_age_ms=oldest_item_age_ms,
        newest_item_age_ms=newest_item_age_ms,
        malformed_items=malformed_items,
    )


def queue_stats_to_dict(stats: QueueStats) -> dict[str, int | str]:
    return stats.model_dump()
