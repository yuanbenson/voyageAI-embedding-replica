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

    # Phase 3: Redis-backed token-count query batching.
    redis_url: str = "redis://redis:6379/0"
    redis_key_prefix: str = "voyage-replica"

    enable_query_batching: bool = False
    query_max_tokens: int = 512
    query_batch_target_tokens: int = 512
    query_max_wait_ms: int = 10
    query_batch_max_items: int = 128

    # This is the gateway instance/process that owns the synchronous HTTP connection.
    # In Kubernetes, set this from metadata.name. Locally, the batching client will
    # derive a unique id when this is left at the default.
    gateway_instance_id: str = "local-gateway"
    result_queue_ttl_seconds: int = 120

    # Phase 3A runs one query batch worker for the nano lane. Document traffic and
    # alias traffic remain on the direct vLLM path until Phase 3B+.
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
