from enum import Enum

from app.schemas import InputType


class WorkloadClass(str, Enum):
    QUERY = "query"
    DOCUMENT = "document"


def classify_embedding_request(
    *,
    input_count: int,
    total_tokens: int,
    input_type: InputType | None,
    query_max_tokens: int,
) -> WorkloadClass:
    """Classify an embedding request for Phase 3 routing.

    The Voyage platform talk defines query traffic as single-example, short-token
    retrieval-path traffic. For this prototype, we additionally require
    input_type="query" so short document-indexing calls do not enter the query
    latency lane by accident.
    """

    if input_count == 1 and total_tokens <= query_max_tokens and input_type == "query":
        return WorkloadClass.QUERY
    return WorkloadClass.DOCUMENT
