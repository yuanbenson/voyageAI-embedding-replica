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

    # Phase 3B: document/multi-input requests are decomposed into child items.
    # Query items keep the default shape where request_id is also the parent id
    # and input_index is 0.
    parent_request_id: str | None = None
    input_index: int = Field(default=0, ge=0)
    input_count: int = Field(default=1, ge=1)


class EmbeddingResultItem(BaseModel):
    request_id: str
    ok: bool
    embedding: list[float] | None = None
    index: int = 0
    error: str | None = None
    status_code: int | None = None

    # Echoed for document-lane reassembly and logging. The gateway still matches
    # pending futures by request_id so these fields are optional metadata rather
    # than required routing keys.
    parent_request_id: str | None = None
    input_index: int = Field(default=0, ge=0)
