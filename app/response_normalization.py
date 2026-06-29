from fastapi import HTTPException, status

from app.schemas import EmbeddingData


def normalize_vllm_embedding_data(vllm_response: dict, expected_count: int) -> list[EmbeddingData]:
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
