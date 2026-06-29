from app.redis_keys import result_queue_key, work_queue_key


def test_query_work_queue_key() -> None:
    assert (
        work_queue_key("voyage-replica", "voyage-4-nano", "query")
        == "voyage-replica:work:embed:voyage-4-nano:query"
    )


def test_document_work_queue_key() -> None:
    assert (
        work_queue_key("voyage-replica", "voyage-4-nano", "document")
        == "voyage-replica:work:embed:voyage-4-nano:document"
    )


def test_result_queue_key() -> None:
    assert (
        result_queue_key("voyage-replica", "embedding-api-abc123")
        == "voyage-replica:results:embedding-api-abc123"
    )
