from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from typing import NoReturn

from fastapi import HTTPException, status

from app.config import Settings
from app.model_registry import ModelRoute
from app.queue_models import EmbeddingResultItem, EmbeddingWorkItem
from app.redis_batch_queue import RedisBatchQueue, current_time_ms
from app.schemas import EmbeddingData

logger = logging.getLogger(__name__)


class GatewayBatchingClient:
    """Gateway-side async request/result bridge for Phase 3 query batching.

    The gateway owns the synchronous HTTP connection. It pushes work into a
    per-model Redis queue, stores an in-memory Future by request_id, and has one
    background listener consuming results for this gateway instance.
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
        if self._closed:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Query batching client is shutting down",
            )

        now_ms = current_time_ms()
        timeout_ms = int(self._settings.request_timeout_seconds * 1000)
        future: asyncio.Future[EmbeddingResultItem] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        item = EmbeddingWorkItem(
            request_id=request_id,
            reply_to=self.gateway_id,
            logical_model=route.logical_model,
            backend_model=route.backend_model,
            lane=route.lane,
            input_text=input_text,
            token_count=token_count,
            created_at_ms=now_ms,
            deadline_ms=now_ms + timeout_ms,
        )

        try:
            await self._queue.enqueue_embedding_work(
                logical_model=route.logical_model,
                workload="query",
                item=item,
            )
            result = await asyncio.wait_for(
                future,
                timeout=self._settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timed out waiting for batched embedding result",
            ) from exc
        finally:
            self._pending.pop(request_id, None)

        if not result.ok:
            raise HTTPException(
                status_code=result.status_code or status.HTTP_502_BAD_GATEWAY,
                detail=result.error or "Batched embedding worker failed",
            )

        if result.embedding is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Batched embedding worker returned no embedding",
            )

        return [EmbeddingData(embedding=result.embedding, index=result.index)]

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
                        },
                    )
                    continue

                if not future.done():
                    future.set_result(result)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("gateway_result_listener_error", extra={"gateway_id": self.gateway_id})
                await asyncio.sleep(0.1)


def _resolve_gateway_instance_id(configured: str) -> str:
    if configured and configured != "local-gateway":
        return configured
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def unreachable(message: str) -> NoReturn:
    raise AssertionError(message)
