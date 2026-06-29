from app.workload import WorkloadClass, classify_embedding_request


def test_single_short_query_is_query_workload() -> None:
    assert (
        classify_embedding_request(
            input_count=1,
            total_tokens=10,
            input_type="query",
            query_max_tokens=512,
        )
        == WorkloadClass.QUERY
    )


def test_multi_input_query_is_document_workload() -> None:
    assert (
        classify_embedding_request(
            input_count=2,
            total_tokens=10,
            input_type="query",
            query_max_tokens=512,
        )
        == WorkloadClass.DOCUMENT
    )


def test_long_single_query_is_document_workload() -> None:
    assert (
        classify_embedding_request(
            input_count=1,
            total_tokens=513,
            input_type="query",
            query_max_tokens=512,
        )
        == WorkloadClass.DOCUMENT
    )


def test_short_document_is_document_workload() -> None:
    assert (
        classify_embedding_request(
            input_count=1,
            total_tokens=10,
            input_type="document",
            query_max_tokens=512,
        )
        == WorkloadClass.DOCUMENT
    )
