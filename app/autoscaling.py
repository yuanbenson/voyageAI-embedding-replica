from __future__ import annotations

import math

from pydantic import BaseModel, Field


class AutoscalingRecommendation(BaseModel):
    model: str
    workload: str
    token_backlog: int = Field(ge=0)
    estimated_tokens_per_second_per_replica: float = Field(ge=0)
    current_replicas: int = Field(ge=0)
    target_drain_time_seconds: float = Field(gt=0)
    estimated_drain_time_seconds: float = Field(ge=0)
    recommended_replicas: int = Field(ge=0)
    min_replicas: int = Field(ge=0)
    max_replicas: int = Field(ge=0)


def recommend_replicas(
    *,
    model: str,
    workload: str,
    token_backlog: int,
    tokens_per_second_per_replica: float,
    current_replicas: int,
    target_drain_time_seconds: float,
    min_replicas: int,
    max_replicas: int,
) -> AutoscalingRecommendation:
    if max_replicas < min_replicas:
        raise ValueError("max_replicas must be >= min_replicas")

    safe_per_replica = max(tokens_per_second_per_replica, 0.0)
    total_current_tps = safe_per_replica * max(current_replicas, 0)
    if token_backlog <= 0:
        estimated_drain_time_seconds = 0.0
        raw_recommended = min_replicas
    elif total_current_tps <= 0:
        estimated_drain_time_seconds = float("inf")
        raw_recommended = max_replicas
    else:
        estimated_drain_time_seconds = token_backlog / total_current_tps
        raw_recommended = math.ceil(
            token_backlog / max(target_drain_time_seconds * safe_per_replica, 1e-9)
        )

    recommended = min(max_replicas, max(min_replicas, raw_recommended))
    # Pydantic cannot serialize inf cleanly in JSON. Preserve a large sentinel.
    if math.isinf(estimated_drain_time_seconds):
        estimated_drain_time_seconds = 1e30

    return AutoscalingRecommendation(
        model=model,
        workload=workload,
        token_backlog=max(0, token_backlog),
        estimated_tokens_per_second_per_replica=safe_per_replica,
        current_replicas=max(0, current_replicas),
        target_drain_time_seconds=target_drain_time_seconds,
        estimated_drain_time_seconds=estimated_drain_time_seconds,
        recommended_replicas=recommended,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )
