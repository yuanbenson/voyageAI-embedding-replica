from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    local_api_keys: set[str] = Field(default_factory=lambda: {"local-dev-key"})

    voyage_tokenizer_model: str = "voyageai/voyage-4-nano"

    vllm_nano_embeddings_url: str = "http://vllm-nano:8000/v1/embeddings"
    vllm_large_shim_embeddings_url: str = "http://vllm-large-shim:8000/v1/embeddings"
    vllm_api_key: str | None = None

    request_timeout_seconds: float = 60.0
    model_context_tokens: int = 32_000
    max_inputs: int = 1_000

    # Phase 3: Redis-backed token-count batching.
    redis_url: str = "redis://redis:6379/0"
    redis_key_prefix: str = "voyage-replica"

    enable_query_batching: bool = False
    query_max_tokens: int = 512
    query_batch_target_tokens: int = 512
    query_max_wait_ms: int = 10
    query_batch_max_items: int = 128

    # Phase 3B: separate synchronous document/indexing lane.
    enable_document_batching: bool = False
    document_batch_target_tokens: int = 2048
    document_max_wait_ms: int = 50
    document_batch_max_items: int = 128

    # This is the gateway instance/process that owns the synchronous HTTP connection.
    # In Kubernetes, set this from metadata.name. Locally, the batching client will
    # derive a unique id when this is left at the default.
    gateway_instance_id: str = "local-gateway"
    result_queue_ttl_seconds: int = 120

    # Phase 3C: observability and autoscaling readiness.
    queue_inspection_model: str = "voyage-4-nano"
    queue_inspection_max_items: int = 10_000

    query_tokens_per_second_per_replica: float = 8_000.0
    document_tokens_per_second_per_replica: float = 8_000.0
    query_target_drain_time_seconds: float = 0.5
    document_target_drain_time_seconds: float = 2.0
    query_autoscale_min_replicas: int = 1
    query_autoscale_max_replicas: int = 4
    document_autoscale_min_replicas: int = 1
    document_autoscale_max_replicas: int = 8
    query_current_replicas: int = 1
    document_current_replicas: int = 1

    # One batch worker process can serve either the query lane or document lane.
    # Kubernetes runs separate Deployments with different BATCH_WORKER_WORKLOAD values.
    batch_worker_model: str = "voyage-4-nano"
    batch_worker_workload: str = "query"

    @field_validator("local_api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, value: object) -> set[str]:
        if isinstance(value, str):
            return {item.strip() for item in value.split(",") if item.strip()}
        if isinstance(value, set):
            return value
        if isinstance(value, list):
            return {str(item).strip() for item in value if str(item).strip()}
        return {"local-dev-key"}

    @field_validator("batch_worker_workload")
    @classmethod
    def validate_batch_worker_workload(cls, value: str) -> str:
        if value not in {"query", "document"}:
            raise ValueError("batch_worker_workload must be 'query' or 'document'")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
