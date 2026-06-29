import logging
from uuid import uuid4

from fastapi import HTTPException, status

from app.batching_client import GatewayBatchingClient
from app.config import Settings
from app.model_registry import resolve_model_route
from app.response_normalization import normalize_vllm_embedding_data
from app.schemas import EmbeddingsRequest, EmbeddingsResponse, Usage
from app.tokenization import (
    apply_input_type_prefixes,
    count_tokens,
    get_tokenizer,
    truncate_to_context_length,
)
from app.vllm_client import VllmClient
from app.workload import WorkloadClass, classify_embedding_request

logger = logging.getLogger(__name__)


class EmbeddingsService:
    def __init__(
        self,
        settings: Settings,
        batching_client: GatewayBatchingClient | None = None,
    ):
        self._settings = settings
        self._vllm = VllmClient(settings)
        self._batching_client = batching_client

    async def create_embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        request_id = str(uuid4())
        route = resolve_model_route(request.model, self._settings)

        if request.output_dimension is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "output_dimension is not supported by the local vLLM backend in Phase 3. "
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

        # Voyage AI-compatible gateway behavior:
        # The public API accepts input_type=query/document/null. vLLM does not know
        # this Voyage AI-specific field, so the gateway applies retrieval prefixes
        # before calling the internal worker.
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

        per_input_tokens = [count_tokens([text], tokenizer) for text in model_inputs]
        total_tokens = sum(per_input_tokens)

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

        workload = classify_embedding_request(
            input_count=len(model_inputs),
            total_tokens=total_tokens,
            input_type=request.input_type,
            query_max_tokens=self._settings.query_max_tokens,
        )

        if self._should_use_query_batching(route_logical_model=route.logical_model, workload=workload):
            logger.info(
                "enqueue_query_embedding_request",
                extra={
                    "request_id": request_id,
                    "logical_model": route.logical_model,
                    "backend_model": route.backend_model,
                    "lane": route.lane,
                    "total_tokens": total_tokens,
                },
            )
            data = await self._batching_client.embed_query(  # type: ignore[union-attr]
                request_id=request_id,
                route=route,
                input_text=model_inputs[0],
                token_count=total_tokens,
            )
        elif self._should_use_document_batching(
            route_logical_model=route.logical_model,
            workload=workload,
        ):
            logger.info(
                "enqueue_document_embedding_request",
                extra={
                    "request_id": request_id,
                    "logical_model": route.logical_model,
                    "backend_model": route.backend_model,
                    "lane": route.lane,
                    "workload": workload.value,
                    "total_tokens": total_tokens,
                    "input_count": len(model_inputs),
                },
            )
            data = await self._batching_client.embed_documents(  # type: ignore[union-attr]
                request_id=request_id,
                route=route,
                inputs=model_inputs,
                token_counts=per_input_tokens,
            )
        else:
            logger.info(
                "direct_embeddings_request",
                extra={
                    "request_id": request_id,
                    "logical_model": route.logical_model,
                    "backend_model": route.backend_model,
                    "lane": route.lane,
                    "workload": workload.value,
                    "total_tokens": total_tokens,
                    "input_count": len(model_inputs),
                },
            )
            vllm_response = await self._vllm.embed(
                route=route,
                inputs=model_inputs,
                request_id=request_id,
            )
            data = normalize_vllm_embedding_data(vllm_response, expected_count=len(model_inputs))

        return EmbeddingsResponse(
            data=data,
            model=request.model,
            usage=Usage(total_tokens=total_tokens),
        )

    def _should_use_query_batching(self, *, route_logical_model: str, workload: WorkloadClass) -> bool:
        return (
            self._settings.enable_query_batching
            and self._batching_client is not None
            and workload == WorkloadClass.QUERY
            and route_logical_model == self._settings.batch_worker_model
        )

    def _should_use_document_batching(
        self,
        *,
        route_logical_model: str,
        workload: WorkloadClass,
    ) -> bool:
        return (
            self._settings.enable_document_batching
            and self._batching_client is not None
            and workload == WorkloadClass.DOCUMENT
            and route_logical_model == self._settings.batch_worker_model
        )
