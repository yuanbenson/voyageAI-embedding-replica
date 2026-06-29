import pytest
from fastapi import HTTPException

from app.config import Settings
from app.embeddings import EmbeddingsService
from app.schemas import EmbeddingData, EmbeddingsRequest


class FakeBatchingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def embed_query(self, **kwargs):
        self.calls.append(kwargs)
        return [EmbeddingData(embedding=[1.0, 2.0], index=0)]


class FakeVllmClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs.get("inputs", [])
        return {
            "data": [
                {"embedding": [3.0, 4.0], "index": index}
                for index, _ in enumerate(inputs)
            ]
        }


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [text.split() for text in texts]}

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr("app.embeddings.get_tokenizer", lambda _: FakeTokenizer())


def make_settings(enable_query_batching: bool) -> Settings:
    return Settings(
        local_api_keys=["local-dev-key"],
        enable_query_batching=enable_query_batching,
        query_max_tokens=512,
        vllm_nano_embeddings_url="http://vllm-nano:8000/v1/embeddings",
        vllm_large_shim_embeddings_url="http://vllm-large-shim:8000/v1/embeddings",
    )


@pytest.mark.asyncio
async def test_short_query_uses_batching_client() -> None:
    batching_client = FakeBatchingClient()
    service = EmbeddingsService(make_settings(True), batching_client=batching_client)
    fake_vllm = FakeVllmClient()
    service._vllm = fake_vllm

    response = await service.create_embeddings(
        EmbeddingsRequest(input="hello", model="voyage-4-nano", input_type="query")
    )

    assert response.data[0].embedding == [1.0, 2.0]
    assert len(batching_client.calls) == 1
    assert len(fake_vllm.calls) == 0


@pytest.mark.asyncio
async def test_document_request_uses_direct_vllm_path() -> None:
    batching_client = FakeBatchingClient()
    service = EmbeddingsService(make_settings(True), batching_client=batching_client)
    fake_vllm = FakeVllmClient()
    service._vllm = fake_vllm

    response = await service.create_embeddings(
        EmbeddingsRequest(input="hello", model="voyage-4-nano", input_type="document")
    )

    assert response.data[0].embedding == [3.0, 4.0]
    assert len(batching_client.calls) == 0
    assert len(fake_vllm.calls) == 1


@pytest.mark.asyncio
async def test_multi_input_query_uses_direct_vllm_path() -> None:
    batching_client = FakeBatchingClient()
    service = EmbeddingsService(make_settings(True), batching_client=batching_client)
    fake_vllm = FakeVllmClient()
    service._vllm = fake_vllm

    response = await service.create_embeddings(
        EmbeddingsRequest(input=["hello", "world"], model="voyage-4-nano", input_type="query")
    )

    assert len(response.data) == 2
    assert len(batching_client.calls) == 0
    assert len(fake_vllm.calls) == 1


@pytest.mark.asyncio
async def test_output_dimension_returns_clean_400() -> None:
    service = EmbeddingsService(make_settings(True), batching_client=FakeBatchingClient())

    with pytest.raises(HTTPException) as exc_info:
        await service.create_embeddings(
            EmbeddingsRequest(
                input="hello",
                model="voyage-4-nano",
                input_type="query",
                output_dimension=256,
            )
        )

    assert exc_info.value.status_code == 400
