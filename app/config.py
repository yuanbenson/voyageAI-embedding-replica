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
