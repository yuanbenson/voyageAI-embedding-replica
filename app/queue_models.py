from pydantic import BaseModel, Field


class EmbeddingWorkItem(BaseModel):
    request_id: str
    reply_to: str
    logical_model: str
    backend_model: str
    lane: str
    input_text: str
    token_count: int = Field(ge=0)
    created_at_ms: int
    deadline_ms: int


class EmbeddingResultItem(BaseModel):
    request_id: str
    ok: bool
    embedding: list[float] | None = None
    index: int = 0
    error: str | None = None
    status_code: int | None = None
