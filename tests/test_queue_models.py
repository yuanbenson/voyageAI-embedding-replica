from app.queue_models import EmbeddingResultItem, EmbeddingWorkItem


def test_embedding_work_item_round_trip() -> None:
    item = EmbeddingWorkItem(
        request_id="r1:0",
        parent_request_id="r1",
        input_index=0,
        input_count=2,
        reply_to="gateway-1",
        logical_model="voyage-4-nano",
        backend_model="voyageai/voyage-4-nano",
        lane="nano",
        input_text="hello",
        token_count=3,
        created_at_ms=1000,
        deadline_ms=2000,
    )

    assert EmbeddingWorkItem.model_validate_json(item.model_dump_json()) == item


def test_embedding_result_item_round_trip() -> None:
    item = EmbeddingResultItem(
        request_id="r1:0",
        parent_request_id="r1",
        input_index=0,
        ok=True,
        embedding=[0.1, 0.2, 0.3],
        index=0,
    )

    assert EmbeddingResultItem.model_validate_json(item.model_dump_json()) == item
