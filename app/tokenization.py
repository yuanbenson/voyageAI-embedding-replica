from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.schemas import InputType


QUERY_PREFIX = "Represent the query for retrieving supporting documents: "
DOCUMENT_PREFIX = "Represent the document for retrieval: "


def apply_input_type_prefix(text: str, input_type: InputType | None) -> str:
    if input_type == "query":
        return QUERY_PREFIX + text
    if input_type == "document":
        return DOCUMENT_PREFIX + text
    return text


def apply_input_type_prefixes(texts: list[str], input_type: InputType | None) -> list[str]:
    return [apply_input_type_prefix(text, input_type) for text in texts]


@lru_cache
def get_tokenizer(tokenizer_model: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(tokenizer_model)


def count_tokens(texts: list[str], tokenizer: PreTrainedTokenizerBase) -> int:
    # We intentionally use the Voyage/Hugging Face tokenizer at the gateway edge.
    # This deterministic count becomes the basis for future token-aware queues.
    encoded = tokenizer(texts, add_special_tokens=False)
    return sum(len(input_ids) for input_ids in encoded["input_ids"])


def truncate_to_context_length(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens_per_input: int,
) -> list[str]:
    truncated: list[str] = []

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens_per_input:
            truncated.append(text)
            continue

        clipped = token_ids[:max_tokens_per_input]
        truncated.append(tokenizer.decode(clipped, skip_special_tokens=True))

    return truncated
