import pytest
from fastapi import HTTPException

from app.config import Settings
from app.embeddings import EmbeddingsService
from app.schemas import EmbeddingData, EmbeddingsRequest


class FakeBatchingClient:
    def __init__(self) -> None:
        self.query_calls: list[dict] = []
        self.document_calls: list[dict] = []

    async def embed_query(self, **kwargs):
        self.query_calls.append(kwargs)
        return [EmbeddingData(embedding=[1.0, 2.0], index=0)]

    async def embed_documents(self, **kwargs):
        self.document_calls.append(kwargs)
        inputs = kwargs.get("inputs", [])
        # Return obviously ordered results so tests catch reassembly mistakes.
        return [
            EmbeddingData(embedding=[float(index), float(index + 100)], index=index)
            for index, _ in enumerate(inputs)
        ]

    @property
    def calls(self) -> list[dict]:
        return self.query_calls + self.document_calls


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


def make_settings(
    enable_query_batching: bool,
    *,
    enable_document_batching: bool = False,
) -> Settings:
    return Settings(
        local_api_keys=["local-dev-key"],
        enable_query_batching=enable_query_batching,
        enable_document_batching=enable_document_batching,
        query_max_tokens=512,
        vllm_nano_embeddings_url="http://vllm-nano:8000/v1/embeddings",
        vllm_large_shim_embeddings_url="http://vllm-large-shim:8000/v1/embeddings",
    )


@pytest.mark.asyncio
async def test_short_query_uses_query_batching_client() -> None:
    batching_client = FakeBatchingClient()
    service = EmbeddingsService(make_settings(True), batching_client=batching_client)
    fake_vllm = FakeVllmClient()
    service._vllm = fake_vllm

    response = await service.create_embeddings(
        EmbeddingsRequest(input="hello", model="voyage-4-nano", input_type="query")
    )

    assert response.data[0].embedding == [1.0, 2.0]
    assert len(batching_client.query_calls) == 1
    assert len(batching_client.document_calls) == 0
    assert len(fake_vllm.calls) == 0


@pytest.mark.asyncio
async def test_document_request_uses_direct_vllm_path_when_document_batching_disabled() -> None:
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
async def test_document_request_uses_document_batching_client_when_enabled() -> None:
    batching_client = FakeBatchingClient()
    service = EmbeddingsService(
        make_settings(True, enable_document_batching=True),
        batching_client=batching_client,
    )
    fake_vllm = FakeVllmClient()
    service._vllm = fake_vllm

    response = await service.create_embeddings(
        EmbeddingsRequest(input="hello", model="voyage-4-nano", input_type="document")
    )

    assert response.data[0].embedding == [0.0, 100.0]
    assert len(batching_client.query_calls) == 0
    assert len(batching_client.document_calls) == 1
    assert len(fake_vllm.calls) == 0


@pytest.mark.asyncio
async def test_multi_input_query_uses_document_batching_lane_when_enabled() -> None:
    batching_client = FakeBatchingClient()
    service = EmbeddingsService(
        make_settings(True, enable_document_batching=True),
        batching_client=batching_client,
    )
    fake_vllm = FakeVllmClient()
    service._vllm = fake_vllm

    response = await service.create_embeddings(
        EmbeddingsRequest(input=["hello", "world"], model="voyage-4-nano", input_type="query")
    )

    assert [item.index for item in response.data] == [0, 1]
    assert response.data[0].embedding == [0.0, 100.0]
    assert response.data[1].embedding == [1.0, 101.0]
    assert len(batching_client.query_calls) == 0
    assert len(batching_client.document_calls) == 1
    assert len(fake_vllm.calls) == 0


@pytest.mark.asyncio
async def test_large_shim_remains_direct_even_when_document_batching_enabled() -> None:
    batching_client = FakeBatchingClient()
    service = EmbeddingsService(
        make_settings(True, enable_document_batching=True),
        batching_client=batching_client,
    )
    fake_vllm = FakeVllmClient()
    service._vllm = fake_vllm

    response = await service.create_embeddings(
        EmbeddingsRequest(input=["hello", "world"], model="voyage-4-large", input_type="document")
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
