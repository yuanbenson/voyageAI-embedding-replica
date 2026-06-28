from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


InputType = Literal["query", "document"]
LogicalModel = Literal["voyage-4-nano", "voyage-4-large"]


class EmbeddingsRequest(BaseModel):
    input: str | list[str] = Field(..., description="A string or list of strings to embed.")
    model: LogicalModel
    input_type: InputType | None = None
    truncation: bool = True
    output_dimension: int | None = None

    # Explicitly present but intentionally restricted in Phase 1.
    output_dtype: Literal["float"] = "float"
    encoding_format: Literal["float"] = "float"

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            if value == "":
                raise ValueError("input string must not be empty")
            return value

        if not value:
            raise ValueError("input list must not be empty")

        for item in value:
            if not isinstance(item, str):
                raise ValueError("all input items must be strings")
            if item == "":
                raise ValueError("input strings must not be empty")

        return value

    @field_validator("output_dimension")
    @classmethod
    def validate_output_dimension(cls, value: int | None) -> int | None:
        if value is None:
            return value

        supported = {256, 512, 1024, 2048}
        if value not in supported:
            raise ValueError(f"output_dimension must be one of {sorted(supported)}")
        return value

    @model_validator(mode="after")
    def validate_phase1_constraints(self) -> "EmbeddingsRequest":
        # This method is intentionally here so future unsupported fields can be rejected
        # in one place as we expand toward rerank / contextualized / batch APIs.
        return self

    def normalized_inputs(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class Usage(BaseModel):
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage
