from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest

# Gateway request-level metrics.
GATEWAY_REQUESTS_TOTAL = Counter(
    "voyage_gateway_requests_total",
    "Total embedding gateway requests.",
    ["model", "input_type", "path", "status"],
)
GATEWAY_ERRORS_TOTAL = Counter(
    "voyage_gateway_errors_total",
    "Total embedding gateway errors.",
    ["model", "error_type"],
)
GATEWAY_BATCHABLE_REQUESTS_TOTAL = Counter(
    "voyage_gateway_batchable_requests_total",
    "Total requests routed through a Redis batching lane.",
    ["model", "workload"],
)
GATEWAY_DIRECT_REQUESTS_TOTAL = Counter(
    "voyage_gateway_direct_requests_total",
    "Total requests routed directly to the model server.",
    ["model", "reason"],
)
GATEWAY_REQUEST_LATENCY_SECONDS = Histogram(
    "voyage_gateway_request_latency_seconds",
    "End-to-end gateway request latency in seconds.",
    ["model", "input_type", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
GATEWAY_TOKEN_COUNT = Histogram(
    "voyage_gateway_token_count",
    "Input token count observed by the gateway.",
    ["model", "input_type", "workload"],
    buckets=(1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
)
GATEWAY_PENDING_REQUESTS = Gauge(
    "voyage_gateway_pending_requests",
    "Number of child requests currently waiting for batched results in this gateway process.",
    ["workload"],
)
GATEWAY_PENDING_DOCUMENT_CHILDREN = Gauge(
    "voyage_gateway_pending_document_children",
    "Number of document child work items currently waiting for batched results "
    "in this gateway process.",
    ["model"],
)
GATEWAY_REASSEMBLY_LATENCY_SECONDS = Histogram(
    "voyage_gateway_reassembly_latency_seconds",
    "Time spent waiting for and reassembling batched child results at the gateway.",
    ["model", "workload"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

# Queue metrics. These are set by /debug/queues and /debug/autoscaling calls for now.
QUEUE_ITEMS = Gauge(
    "voyage_queue_items",
    "Number of queued work items by model and workload.",
    ["model", "workload"],
)
QUEUE_TOKEN_BACKLOG = Gauge(
    "voyage_queue_token_backlog",
    "Queued token backlog by model and workload.",
    ["model", "workload"],
)
QUEUE_OLDEST_ITEM_AGE_SECONDS = Gauge(
    "voyage_queue_oldest_item_age_seconds",
    "Age in seconds of the oldest queued work item by model and workload.",
    ["model", "workload"],
)

# Worker batch metrics.
WORKER_CLAIMED_BATCHES_TOTAL = Counter(
    "voyage_worker_claimed_batches_total",
    "Total token-budgeted batches claimed by workers.",
    ["model", "workload"],
)
WORKER_COMPLETED_BATCHES_TOTAL = Counter(
    "voyage_worker_completed_batches_total",
    "Total token-budgeted batches completed by workers.",
    ["model", "workload"],
)
WORKER_FAILED_BATCHES_TOTAL = Counter(
    "voyage_worker_failed_batches_total",
    "Total token-budgeted batches that failed in workers.",
    ["model", "workload", "error_type"],
)
WORKER_CLAIMED_ITEMS_TOTAL = Counter(
    "voyage_worker_claimed_items_total",
    "Total work items claimed by batch workers.",
    ["model", "workload"],
)
WORKER_CLAIMED_TOKENS_TOTAL = Counter(
    "voyage_worker_claimed_tokens_total",
    "Total input tokens claimed by batch workers.",
    ["model", "workload"],
)
WORKER_BATCH_SIZE = Histogram(
    "voyage_worker_batch_size",
    "Number of work items in a claimed worker batch.",
    ["model", "workload"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)
WORKER_BATCH_TOKENS = Histogram(
    "voyage_worker_batch_tokens",
    "Total tokens in a claimed worker batch.",
    ["model", "workload"],
    buckets=(1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
)
WORKER_VLLM_LATENCY_SECONDS = Histogram(
    "voyage_worker_vllm_latency_seconds",
    "Model-server call latency observed by batch workers.",
    ["model", "workload"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
WORKER_BATCH_PROCESSING_LATENCY_SECONDS = Histogram(
    "voyage_worker_batch_processing_latency_seconds",
    "End-to-end worker batch processing latency in seconds.",
    ["model", "workload"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
WORKER_OLDEST_ITEM_WAIT_SECONDS = Histogram(
    "voyage_worker_oldest_item_wait_seconds",
    "Wait time of the oldest item in a claimed batch.",
    ["model", "workload"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
WORKER_LAST_BATCH_TOKENS = Gauge(
    "voyage_worker_last_batch_tokens",
    "Token count in the most recently completed worker batch.",
    ["model", "workload"],
)
WORKER_LAST_BATCH_SIZE = Gauge(
    "voyage_worker_last_batch_size",
    "Item count in the most recently completed worker batch.",
    ["model", "workload"],
)

AUTOSCALING_RECOMMENDED_REPLICAS = Gauge(
    "voyage_autoscaling_recommended_replicas",
    "Advisory recommended model replica count by model and workload.",
    ["model", "workload"],
)
AUTOSCALING_ESTIMATED_DRAIN_TIME_SECONDS = Gauge(
    "voyage_autoscaling_estimated_drain_time_seconds",
    "Advisory estimated queue drain time in seconds by model and workload.",
    ["model", "workload"],
)


def prometheus_content() -> bytes:
    return generate_latest()


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST


def normalize_label(value: object | None) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text if text else "none"
