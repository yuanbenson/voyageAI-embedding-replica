import pytest
from pydantic import ValidationError

from app.schemas import EmbeddingsRequest


def test_accepts_single_string_input() -> None:
    request = EmbeddingsRequest(input="hello", model="voyage-4-nano")
    assert request.normalized_inputs() == ["hello"]


def test_accepts_list_input() -> None:
    request = EmbeddingsRequest(input=["a", "b"], model="voyage-4-large")
    assert request.normalized_inputs() == ["a", "b"]


def test_rejects_empty_list() -> None:
    with pytest.raises(ValidationError):
        EmbeddingsRequest(input=[], model="voyage-4-nano")


def test_rejects_bad_output_dimension() -> None:
    with pytest.raises(ValidationError):
        EmbeddingsRequest(input="hello", model="voyage-4-nano", output_dimension=123)
