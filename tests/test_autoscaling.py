from app.autoscaling import recommend_replicas


def test_recommends_min_replicas_when_backlog_is_empty() -> None:
    recommendation = recommend_replicas(
        model="voyage-4-nano",
        workload="query",
        token_backlog=0,
        tokens_per_second_per_replica=8000,
        current_replicas=1,
        target_drain_time_seconds=0.5,
        min_replicas=1,
        max_replicas=4,
    )

    assert recommendation.estimated_drain_time_seconds == 0
    assert recommendation.recommended_replicas == 1


def test_recommends_more_replicas_for_large_backlog() -> None:
    recommendation = recommend_replicas(
        model="voyage-4-nano",
        workload="document",
        token_backlog=32_000,
        tokens_per_second_per_replica=8_000,
        current_replicas=1,
        target_drain_time_seconds=2.0,
        min_replicas=1,
        max_replicas=8,
    )

    assert recommendation.estimated_drain_time_seconds == 4.0
    assert recommendation.recommended_replicas == 2


def test_recommendation_respects_max_replicas() -> None:
    recommendation = recommend_replicas(
        model="voyage-4-nano",
        workload="document",
        token_backlog=1_000_000,
        tokens_per_second_per_replica=8_000,
        current_replicas=1,
        target_drain_time_seconds=2.0,
        min_replicas=1,
        max_replicas=8,
    )

    assert recommendation.recommended_replicas == 8
