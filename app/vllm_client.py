import logging
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.config import Settings
from app.model_registry import ModelRoute


logger = logging.getLogger(__name__)


class VllmClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def embed(
        self,
        *,
        route: ModelRoute,
        inputs: list[str],
        request_id: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": route.backend_model,
            "input": inputs,
        }

        headers = {"x-request-id": request_id}
        if self._settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.vllm_api_key}"

        logger.info(
            "forwarding_embeddings_request",
            extra={
                "request_id": request_id,
                "logical_model": route.logical_model,
                "backend_model": route.backend_model,
                "lane": route.lane,
                "is_alias": route.is_alias,
                "input_count": len(inputs),
            },
        )

        try:
            async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
                response = await client.post(route.embeddings_url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timed out calling internal vLLM worker",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed calling internal vLLM worker: {exc}",
            ) from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "message": "Internal vLLM worker returned an error",
                    "status_code": response.status_code,
                    "body": response.text[:2000],
                },
            )

        return response.json()
