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
        if value <= 0:
            raise ValueError("output_dimension must be positive")
        return value

    @model_validator(mode="after")
    def validate_phase3_constraints(self) -> "EmbeddingsRequest":
        # Unsupported fields are rejected in the service layer where we can return
        # a clear Voyage-compatible 400 instead of a Pydantic 422.
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
