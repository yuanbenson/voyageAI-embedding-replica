import logging
from uuid import uuid4

from fastapi import HTTPException, status

from app.config import Settings
from app.model_registry import resolve_model_route
from app.schemas import EmbeddingData, EmbeddingsRequest, EmbeddingsResponse, Usage
from app.tokenization import (
    apply_input_type_prefixes,
    count_tokens,
    get_tokenizer,
    truncate_to_context_length,
)
from app.vllm_client import VllmClient


logger = logging.getLogger(__name__)


class EmbeddingsService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._vllm = VllmClient(settings)

    async def create_embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        request_id = str(uuid4())
        route = resolve_model_route(request.model, self._settings)

        if request.output_dimension is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "output_dimension is not supported by the local vLLM backend in Phase 2. "
                    "Omit output_dimension to use the backend default embedding size."
                ),
            )

        raw_inputs = request.normalized_inputs()
        if len(raw_inputs) > self._settings.max_inputs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"input list length must be <= {self._settings.max_inputs}",
            )

        tokenizer = get_tokenizer(self._settings.voyage_tokenizer_model)

        # Voyage-compatible gateway behavior:
        # The public API accepts input_type=query/document/null.
        # vLLM does not know this Voyage-specific field, so the gateway applies
        # the transparent retrieval prefixes before calling the internal worker.
        model_inputs = apply_input_type_prefixes(raw_inputs, request.input_type)

        if request.truncation:
            model_inputs = truncate_to_context_length(
                model_inputs,
                tokenizer,
                self._settings.model_context_tokens,
            )
        else:
            for text in model_inputs:
                token_count = count_tokens([text], tokenizer)
                if token_count > self._settings.model_context_tokens:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            "input exceeds model context length and truncation=false; "
                            f"input_tokens={token_count}, "
                            f"context_length={self._settings.model_context_tokens}"
                        ),
                    )

        total_tokens = count_tokens(model_inputs, tokenizer)

        if route.is_alias:
            logger.info(
                "logical_model_alias_route",
                extra={
                    "request_id": request_id,
                    "logical_model": route.logical_model,
                    "backend_model": route.backend_model,
                    "lane": route.lane,
                    "total_tokens": total_tokens,
                },
            )

        vllm_response = await self._vllm.embed(
            route=route,
            inputs=model_inputs,
            request_id=request_id,
        )

        data = _normalize_vllm_embedding_data(vllm_response, expected_count=len(model_inputs))

        return EmbeddingsResponse(
            data=data,
            model=request.model,
            usage=Usage(total_tokens=total_tokens),
        )


def _normalize_vllm_embedding_data(vllm_response: dict, expected_count: int) -> list[EmbeddingData]:
    raw_data = vllm_response.get("data")
    if not isinstance(raw_data, list):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Internal vLLM response missing data list",
        )

    if len(raw_data) != expected_count:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Internal vLLM response data length mismatch; "
                f"expected={expected_count}, actual={len(raw_data)}"
            ),
        )

    normalized: list[EmbeddingData] = []

    for default_index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Internal vLLM response data item must be an object",
            )

        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Internal vLLM response data item missing embedding list",
            )

        # Preserve vLLM index if present; otherwise fall back to response order.
        index = item.get("index", default_index)

        normalized.append(
            EmbeddingData(
                embedding=embedding,
                index=index,
            )
        )

    normalized.sort(key=lambda item: item.index)
    return normalized
