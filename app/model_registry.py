from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class ModelRoute:
    logical_model: str
    backend_model: str
    embeddings_url: str
    lane: str
    is_alias: bool


def resolve_model_route(model: str, settings: Settings) -> ModelRoute:
    if model == "voyage-4-nano":
        return ModelRoute(
            logical_model="voyage-4-nano",
            backend_model="voyageai/voyage-4-nano",
            embeddings_url=settings.vllm_nano_embeddings_url,
            lane="nano",
            is_alias=False,
        )

    if model == "voyage-4-large":
        return ModelRoute(
            logical_model="voyage-4-large",
            backend_model="voyageai/voyage-4-nano",
            embeddings_url=settings.vllm_large_shim_embeddings_url,
            lane="large-shim",
            is_alias=True,
        )

    # Pydantic should catch this first. This is defensive.
    raise ValueError(f"Unsupported model: {model}")
