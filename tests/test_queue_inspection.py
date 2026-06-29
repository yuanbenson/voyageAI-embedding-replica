import pytest

from app.queue_inspection import inspect_embedding_queue
from app.queue_models import EmbeddingWorkItem


class FakeQueue:
    def __init__(self, raw_items: list[str]) -> None:
        self.raw_items = raw_items

    async def list_embedding_work_raw(self, **kwargs) -> list[str]:
        return self.raw_items


@pytest.mark.asyncio
async def test_queue_stats_sum_tokens_and_age(monkeypatch) -> None:
    monkeypatch.setattr("app.queue_inspection.current_time_ms", lambda: 2_000)
    items = [
        EmbeddingWorkItem(
            request_id="r1",
            reply_to="gateway",
            logical_model="voyage-4-nano",
            backend_model="voyageai/voyage-4-nano",
            lane="nano",
            input_text="hello",
            token_count=10,
            created_at_ms=1_000,
            deadline_ms=10_000,
        ).model_dump_json(),
        EmbeddingWorkItem(
            request_id="r2",
            reply_to="gateway",
            logical_model="voyage-4-nano",
            backend_model="voyageai/voyage-4-nano",
            lane="nano",
            input_text="world",
            token_count=20,
            created_at_ms=1_500,
            deadline_ms=10_000,
        ).model_dump_json(),
    ]

    stats = await inspect_embedding_queue(
        FakeQueue(items),
        logical_model="voyage-4-nano",
        workload="query",
    )

    assert stats.items == 2
    assert stats.token_backlog == 30
    assert stats.oldest_item_age_ms == 1_000
    assert stats.newest_item_age_ms == 500
    assert stats.malformed_items == 0


@pytest.mark.asyncio
async def test_queue_stats_count_malformed_items(monkeypatch) -> None:
    monkeypatch.setattr("app.queue_inspection.current_time_ms", lambda: 2_000)

    stats = await inspect_embedding_queue(
        FakeQueue(["not json"]),
        logical_model="voyage-4-nano",
        workload="document",
    )

    assert stats.items == 1
    assert stats.token_backlog == 0
    assert stats.malformed_items == 1
