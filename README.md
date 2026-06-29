# Voyage AI-Compatible Embedding Gateway — Phase 3

This repo is a self-hosted, Voyage AI-compatible embedding service prototype.

It exposes a Voyage AI-compatible embeddings endpoint:

```http
POST /v1/embeddings
Authorization: Bearer <local-api-key>
Content-Type: application/json
```

The gateway accepts Voyage AI-style embedding requests, validates and normalizes them, applies query/document prompt handling, counts tokens, and routes them to the appropriate internal serving path.

For short query requests, Phase 3 uses Redis-backed token-count batching before calling vLLM. For document, multi-input, or long requests, the gateway falls back to the direct vLLM path.

## Phase 3 scope

Implemented in this overlay:

- `POST /v1/embeddings`
- Bearer auth
- Pydantic request validation
- `input` normalization: string or list of strings
- `input_type`: `null`, `query`, or `document`
- Voyage-style prompt prefixing for `query` and `document`
- Token counting at the gateway using the `voyageai/voyage-4-nano` tokenizer
- Logical model routing:
  - `voyage-4-nano`
  - `voyage-4-large`
- Both logical models route to backend model `voyageai/voyage-4-nano` (as `voyage-4-large` is proprietary, while `voyageai/voyage-4-nano` is an open-weight model).
- Internal vLLM call
- Voyage-compatible response shaping
- Redis-backed token-count batching for short query embedding requests
- CPU-only batch worker that pulls up to 512 tokens and calls vLLM once per microbatch
- Per-gateway result queues for synchronous HTTP response handoff

Deferred:

- Document/indexing queue path
- KEDA autoscaling from queue token metrics
- KServe
- Reranking
- Large-request unbatching/fanout
- Batch jobs with 12-hour completion window
- Contextualized embeddings
- Multimodal embeddings
- Prometheus/OpenTelemetry dashboards
- Runtime-level CUDA graph / kernel fusion experiments

## Architecture

This project implements a Voyage AI-compatible embedding gateway backed by vLLM. Phase 3 adds Redis-based token-count batching for short, latency-sensitive query embedding requests.

```mermaid
flowchart TB
    client["Client / SDK<br/>POST /v1/embeddings"]

    subgraph gatewayLayer["Gateway layer"]
        gateway["embedding-api<br/>FastAPI gateway"]
        auth["Auth + schema validation"]
        tokenize["Tokenize + count tokens"]
        classify{"Workload classifier"}
    end

    subgraph queueLayer["Queue and batching layer"]
        redisWork["Redis work queue<br/>work:embed:voyage-4-nano:query"]
        worker["embedding-query-batch-worker<br/>CPU-only batching worker"]
        claim["Atomic token-budget claim<br/>target ≤ 512 tokens"]
        redisResults["Redis result queue<br/>results:gateway_pod_id"]
    end

    subgraph gpuLayer["GPU serving layer"]
        direct["Direct vLLM path<br/>documents / long requests"]
        vllm["vllm-nano<br/>GPU pod running vllm serve"]
        split["Split batch results<br/>by request_id"]
    end

    queryRoute["query<br/>batch_size = 1<br/>tokens ≤ 512"]
    directRoute["document / multi-input<br/>/ long request"]

    client --> gateway
    gateway --> auth --> tokenize --> classify

    classify --> queryRoute --> redisWork
    classify --> directRoute --> direct

    redisWork --> worker --> claim --> vllm
    direct --> vllm

    vllm --> split --> redisResults --> gateway
    gateway --> client

    classDef gateway fill:#e8f1ff,stroke:#2563eb,stroke-width:1px,color:#111827;
    classDef redis fill:#fff7ed,stroke:#ea580c,stroke-width:1px,color:#111827;
    classDef worker fill:#ecfdf5,stroke:#16a34a,stroke-width:1px,color:#111827;
    classDef gpu fill:#f5f3ff,stroke:#7c3aed,stroke-width:1px,color:#111827;
    classDef decision fill:#fefce8,stroke:#ca8a04,stroke-width:1px,color:#111827;
    classDef route fill:#f3f4f6,stroke:#6b7280,stroke-width:1px,color:#111827;
    classDef client fill:#f0fdf4,stroke:#16a34a,stroke-width:1px,color:#111827;

    class gateway,auth,tokenize gateway;
    class redisWork,redisResults redis;
    class worker,claim,split worker;
    class direct,vllm gpu;
    class classify decision;
    class queryRoute,directRoute route;
    class client client;
```

### Phase 3A request flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant G as embedding-api gateway
    participant R as Redis
    participant W as batch-worker
    participant V as vLLM GPU pod

    C->>G: POST /v1/embeddings, model=voyage-4-nano, input_type=query
    G->>G: Validate auth and schema
    G->>G: Tokenize input
    G->>G: Classify as query: batch_size=1, tokens<=512
    G->>R: Enqueue work item: request_id, token_count, reply_to
    G->>G: Store pending Future: pending[request_id]

    W->>R: Atomically claim batch: sum(token_count)<=512
    R-->>W: Return N work items
    W->>V: POST /v1/embeddings as one batched request
    V-->>W: Batched embeddings
    W->>R: Push individual results to results:gateway_pod_id

    R-->>G: Result for request_id
    G->>G: Resolve pending Future
    G-->>C: Voyage-compatible response
```

### Kubernetes deployment shape

```mermaid
flowchart TB
    subgraph ns["Kubernetes namespace: inference"]
        subgraph cpu["CPU nodes"]
            api["Deployment: embedding-api<br/>FastAPI gateway<br/>replicas: 2"]
            redis["Deployment: redis<br/>Redis queue backend"]
            batch["Deployment: embedding-query-batch-worker<br/>CPU-only batching worker"]
        end

        subgraph gpu["GPU node pool"]
            vllm["Deployment: vllm-nano<br/>vLLM serve<br/>nvidia.com/gpu: 1"]
        end

        svcApi["Service: embedding-api"] --> api
        api --> redis
        batch --> redis
        batch --> svcVllm["Service: vllm-nano"]
        svcVllm --> vllm
    end

    user["Local test client<br/>curl / load_test_queries.py"] -->|"kubectl port-forward<br/>localhost:8000"| svcApi

    classDef cpu fill:#eff6ff,stroke:#2563eb,stroke-width:1px,color:#111827;
    classDef redis fill:#fff7ed,stroke:#ea580c,stroke-width:1px,color:#111827;
    classDef gpu fill:#f5f3ff,stroke:#7c3aed,stroke-width:1px,color:#111827;
    classDef service fill:#f3f4f6,stroke:#6b7280,stroke-width:1px,color:#111827;
    classDef client fill:#ecfdf5,stroke:#16a34a,stroke-width:1px,color:#111827;

    class api,batch cpu;
    class redis redis;
    class vllm gpu;
    class svcApi,svcVllm service;
    class user client;
```


### Batching policy

Phase 3A batches only short query embedding requests:

```text
Batchable query:
  model = voyage-4-nano
  input_type = query
  number of inputs = 1
  token_count <= 512

Not batched yet:
  document requests
  multi-input requests
  long requests over 512 tokens
  voyage-4-large shim requests
  reranking requests
```

The batch worker pulls queued query requests from Redis and forms a microbatch whose total token count is capped at `QUERY_BATCH_TARGET_TOKENS=512`.

This keeps the synchronous API behavior while reducing the number of vLLM calls under concurrent short-query traffic.

### Validated Phase 3A behavior

Example validation run:

```text
50 concurrent short query requests
→ 7 vLLM embedding calls
→ max observed batch_size = 19
→ max observed batch_tokens = 238
→ all batch_tokens <= 512
```

This confirms that the gateway no longer forwards every short query directly to vLLM. Instead, requests are routed through Redis, grouped by the batch worker, sent to vLLM as token-budgeted microbatches, and split back into individual client responses.

## Environment variables

```bash
LOCAL_API_KEYS='["local-dev-key"]'
VOYAGE_TOKENIZER_MODEL=voyageai/voyage-4-nano

VLLM_NANO_EMBEDDINGS_URL=http://vllm-nano:8000/v1/embeddings
VLLM_LARGE_SHIM_EMBEDDINGS_URL=http://vllm-large-shim:8000/v1/embeddings

# Optional, if your internal vLLM server requires auth.
VLLM_API_KEY=

REQUEST_TIMEOUT_SECONDS=60
MODEL_CONTEXT_TOKENS=32000
MAX_INPUTS=1000

# Phase 3 batching.
REDIS_URL=redis://redis:6379/0
REDIS_KEY_PREFIX=voyage-replica
ENABLE_QUERY_BATCHING=true
QUERY_MAX_TOKENS=512
QUERY_BATCH_TARGET_TOKENS=512
QUERY_MAX_WAIT_MS=10
QUERY_BATCH_MAX_ITEMS=128
GATEWAY_INSTANCE_ID=local-gateway
RESULT_QUEUE_TTL_SECONDS=120
BATCH_WORKER_MODEL=voyage-4-nano
BATCH_WORKER_WORKLOAD=query
```

## Run gateway locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export LOCAL_API_KEYS='["local-dev-key"]'
export VLLM_NANO_EMBEDDINGS_URL=http://localhost:8001/v1/embeddings
export VLLM_LARGE_SHIM_EMBEDDINGS_URL=http://localhost:8001/v1/embeddings
export REDIS_URL=redis://localhost:6379/0
export ENABLE_QUERY_BATCHING=false

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

To test Phase 3 batching locally, run Redis and a batch worker in separate terminals:

```bash
docker run --rm -p 6379:6379 redis:7-alpine
```

```bash
export LOCAL_API_KEYS='["local-dev-key"]'
export REDIS_URL=redis://localhost:6379/0
export ENABLE_QUERY_BATCHING=true
export VLLM_NANO_EMBEDDINGS_URL=http://localhost:8001/v1/embeddings
python -m app.batch_worker
```

Then run the gateway with `ENABLE_QUERY_BATCHING=true`.

## Example request

```bash
curl --request POST \
  --url http://localhost:8000/v1/embeddings \
  --header "Authorization: Bearer local-dev-key" \
  --header "content-type: application/json" \
  --data '{
    "input": "hello world",
    "model": "voyage-4-nano",
    "input_type": "query"
  }'
```

## Load-test query batching

After port-forwarding or running locally:

```bash
python scripts/load_test_queries.py --requests 20 --concurrency 20
```

Check batch-worker logs for lines like:

```text
claimed_query_batch batch_size=... batch_tokens=... target_tokens=512
completed_query_batch batch_size=... batch_tokens=...
```

For 20 concurrent short query requests, you should see fewer vLLM calls than client requests.

## Example internal vLLM worker

For the real GKE version, the vLLM runtime still runs as a separate GPU Deployment.

```bash
vllm serve voyageai/voyage-4-nano \
  --runner pooling \
  --convert embed \
  --hf-overrides '{"architectures":["VoyageQwen3BidirectionalEmbedModel"]}' \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name voyageai/voyage-4-nano
```

The gateway or batch-worker then calls:

```text
http://vllm-nano:8000/v1/embeddings
http://vllm-large-shim:8000/v1/embeddings
```

## Kubernetes notes

Phase 3 adds:

```text
redis Deployment/Service
embedding-query-batch-worker Deployment
GATEWAY_INSTANCE_ID from pod metadata.name
ENABLE_QUERY_BATCHING=true
```

Build and push the Phase 3 app image:

```bash
export IMAGE=us-central1-docker.pkg.dev/voyage-replica-dev/voyage-replica/embedding-api:phase3-amd64

docker buildx build \
  --platform linux/amd64 \
  -t "$IMAGE" \
  --push .
```

Apply:

```bash
kubectl apply -k k8s/base
kubectl -n inference rollout status deployment/embedding-api
kubectl -n inference rollout status deployment/embedding-query-batch-worker
```

Validate via port-forward:

```bash
kubectl -n inference port-forward svc/embedding-api 8000:80
```

## Notes

This gateway intentionally tokenizes both logical models with `voyageai/voyage-4-nano`.

That is because this project has only one real backend model. The logical `voyage-4-large` route is a platform shim for studying model routing and model-specific scaling boundaries, not a claim that the backend model is actually `voyage-4-large`.
