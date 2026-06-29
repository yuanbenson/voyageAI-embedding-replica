from app.metrics import GATEWAY_REQUESTS_TOTAL, prometheus_content


def test_prometheus_content_contains_gateway_metric() -> None:
    GATEWAY_REQUESTS_TOTAL.labels(
        model="voyage-4-nano",
        input_type="query",
        path="query_batch",
        status="success",
    ).inc()

    body = prometheus_content().decode("utf-8")

    assert "voyage_gateway_requests_total" in body
    assert 'model="voyage-4-nano"' in body
