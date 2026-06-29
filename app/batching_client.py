from __future__ import annotations

import asyncio
import logging
import os
import time
import socket
import uuid
from typing import NoReturn

from fastapi import HTTPException, status

from app.config import Settings
from app.metrics import (
    GATEWAY_PENDING_DOCUMENT_CHILDREN,
    GATEWAY_PENDING_REQUESTS,
    GATEWAY_REASSEMBLY_LATENCY_SECONDS,
)
from app.model_registry import ModelRoute
from app.queue_models import EmbeddingResultItem, EmbeddingWorkItem
from app.redis_batch_queue import RedisBatchQueue, current_time_ms
from app.schemas import EmbeddingData

logger = logging.getLogger(__name__)


class GatewayBatchingClient:
    """Gateway-side async request/result bridge for Redis-backed batching.

    The gateway owns the synchronous HTTP connection. It pushes work into a
    per-model/per-workload Redis queue, stores in-memory Futures by child
    request_id, and has one background listener consuming results for this
    gateway instance.

    Phase 3A used this for one-result query requests. Phase 3B extends the same
    bridge to document/multi-input requests by decomposing one parent HTTP
    request into child work items and reassembling the child results in input
    order.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self.gateway_id = _resolve_gateway_instance_id(settings.gateway_instance_id)
        self._queue = RedisBatchQueue.from_url(settings.redis_url, settings.redis_key_prefix)
        self._pending: dict[str, asyncio.Future[EmbeddingResultItem]] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        await self._queue.ping()
        self._listener_task = asyncio.create_task(self._listen_for_results())
        logger.info("gateway_batching_client_started", extra={"gateway_id": self.gateway_id})

    async def stop(self) -> None:
        self._closed = True
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        for request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(
                    RuntimeError(f"gateway batching client stopped before {request_id} completed")
                )
        self._pending.clear()
        await self._queue.close()
        logger.info("gateway_batching_client_stopped", extra={"gateway_id": self.gateway_id})

    async def embed_query(
        self,
        *,
        request_id: str,
        route: ModelRoute,
        input_text: str,
        token_count: int,
    ) -> list[EmbeddingData]:
        results = await self._enqueue_and_wait_for_children(
            parent_request_id=request_id,
            route=route,
            workload="query",
            inputs=[input_text],
            token_counts=[token_count],
        )
        return [EmbeddingData(embedding=results[0].embedding or [], index=0)]

    async def embed_documents(
        self,
        *,
        request_id: str,
        route: ModelRoute,
        inputs: list[str],
        token_counts: list[int],
    ) -> list[EmbeddingData]:
        if len(inputs) != len(token_counts):
            raise ValueError("inputs and token_counts must have the same length")

        results = await self._enqueue_and_wait_for_children(
            parent_request_id=request_id,
            route=route,
            workload="document",
            inputs=inputs,
            token_counts=token_counts,
        )

        return [
            EmbeddingData(embedding=result.embedding or [], index=index)
            for index, result in enumerate(results)
        ]

    async def _enqueue_and_wait_for_children(
        self,
        *,
        parent_request_id: str,
        route: ModelRoute,
        workload: str,
        inputs: list[str],
        token_counts: list[int],
    ) -> list[EmbeddingResultItem]:
        if self._closed:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batching client is shutting down",
            )

        start_time = time.perf_counter()
        now_ms = current_time_ms()
        timeout_ms = int(self._settings.request_timeout_seconds * 1000)
        child_request_ids = [
            parent_request_id if len(inputs) == 1 else f"{parent_request_id}:{index}"
            for index in range(len(inputs))
        ]

        futures: list[asyncio.Future[EmbeddingResultItem]] = []
        try:
            for index, (child_request_id, input_text, token_count) in enumerate(
                zip(child_request_ids, inputs, token_counts, strict=True)
            ):
                future: asyncio.Future[EmbeddingResultItem] = (
                    asyncio.get_running_loop().create_future()
                )
                self._pending[child_request_id] = future
                futures.append(future)
                _set_pending_metrics(workload, route.logical_model, self._pending)

                item = EmbeddingWorkItem(
                    request_id=child_request_id,
                    parent_request_id=parent_request_id,
                    input_index=index,
                    input_count=len(inputs),
                    reply_to=self.gateway_id,
                    logical_model=route.logical_model,
                    backend_model=route.backend_model,
                    lane=route.lane,
                    input_text=input_text,
                    token_count=token_count,
                    created_at_ms=now_ms,
                    deadline_ms=now_ms + timeout_ms,
                )
                await self._queue.enqueue_embedding_work(
                    logical_model=route.logical_model,
                    workload=workload,
                    item=item,
                )

            logger.info(
                "queued_%s_embedding_request",
                workload,
                extra={
                    "parent_request_id": parent_request_id,
                    "logical_model": route.logical_model,
                    "input_count": len(inputs),
                    "total_tokens": sum(token_counts),
                },
            )

            results = await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=self._settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Timed out waiting for batched {workload} embedding result",
            ) from exc
        finally:
            for child_request_id in child_request_ids:
                self._pending.pop(child_request_id, None)
            _set_pending_metrics(workload, route.logical_model, self._pending)

        GATEWAY_REASSEMBLY_LATENCY_SECONDS.labels(
            model=route.logical_model,
            workload=workload,
        ).observe(time.perf_counter() - start_time)

        results_by_index: dict[int, EmbeddingResultItem] = {}
        for result in results:
            if not result.ok:
                raise HTTPException(
                    status_code=result.status_code or status.HTTP_502_BAD_GATEWAY,
                    detail=result.error or f"Batched {workload} embedding worker failed",
                )
            if result.embedding is None:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Batched {workload} embedding worker returned no embedding",
                )
            results_by_index[result.input_index] = result

        missing = [index for index in range(len(inputs)) if index not in results_by_index]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Batched {workload} embedding response missing indexes: {missing}",
            )

        ordered = [results_by_index[index] for index in range(len(inputs))]
        logger.info(
            "completed_%s_embedding_request",
            workload,
            extra={
                "parent_request_id": parent_request_id,
                "input_count": len(inputs),
                "total_tokens": sum(token_counts),
            },
        )
        return ordered

    async def _listen_for_results(self) -> None:
        while not self._closed:
            try:
                result = await self._queue.blpop_embedding_result(
                    gateway_id=self.gateway_id,
                    timeout_seconds=1,
                )
                if result is None:
                    continue

                future = self._pending.get(result.request_id)
                if future is None:
                    logger.warning(
                        "batched_result_without_pending_request",
                        extra={
                            "gateway_id": self.gateway_id,
                            "request_id": result.request_id,
                            "parent_request_id": result.parent_request_id,
                            "input_index": result.input_index,
                        },
                    )
                    continue

                if not future.done():
                    future.set_result(result)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "gateway_result_listener_error",
                    extra={"gateway_id": self.gateway_id},
                )
                await asyncio.sleep(0.1)


def _set_pending_metrics(
    workload: str,
    logical_model: str,
    pending: dict[str, asyncio.Future[EmbeddingResultItem]],
) -> None:
    GATEWAY_PENDING_REQUESTS.labels(workload=workload).set(len(pending))
    if workload == "document":
        GATEWAY_PENDING_DOCUMENT_CHILDREN.labels(model=logical_model).set(len(pending))


def _resolve_gateway_instance_id(configured: str) -> str:
    if configured and configured != "local-gateway":
        return configured
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def unreachable(message: str) -> NoReturn:
    raise AssertionError(message)
