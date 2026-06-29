# Voyage-Compatible Embedding Gateway — Phase 1

This is the Phase 1 gateway for a self-hosted, Voyage-compatible embedding service.

It exposes:

```http
POST /v1/embeddings
Authorization: Bearer <local-api-key>
Content-Type: application/json
```

and forwards requests to internal vLLM embedding workers.

## Phase 1 scope

Implemented:

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
- Both logical models route to backend model `voyageai/voyage-4-nano`
- Internal vLLM call
- Voyage-compatible response shaping

Deferred:

- Redis token-aware batching
- KEDA autoscaling
- KServe
- Reranking
- Batch jobs with 12-hour completion window
- Contextualized embeddings
- Multimodal embeddings
- Observability

## Architecture

```text
Client
  ↓
GCP HTTPS Load Balancer / GKE Gateway
  ↓
embedding-api FastAPI gateway
  ↓
logical model router
  ├── voyage-4-nano lane
  │     ↓
  │   internal vLLM /v1/embeddings
  │
  └── voyage-4-large lane
        ↓
      internal vLLM /v1/embeddings

Both lanes initially load:
  voyageai/voyage-4-nano
```

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
```

## Run gateway locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export LOCAL_API_KEYS=local-dev-key
export VLLM_NANO_EMBEDDINGS_URL=http://localhost:8001/v1/embeddings
export VLLM_LARGE_SHIM_EMBEDDINGS_URL=http://localhost:8001/v1/embeddings

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Example request

```bash
curl --request POST \
  --url http://localhost:8000/v1/embeddings \
  --header "Authorization: Bearer local-dev-key" \
  --header "content-type: application/json" \
  --data '{
    "input": ["hello world", "mongodb atlas vector search"],
    "model": "voyage-4-nano",
    "input_type": "query"
  }'
```

The response model field remains `voyage-4-large`, but the gateway routes internally to the `large-shim` lane backed by `voyageai/voyage-4-nano`.

## Example internal vLLM worker

For the real GKE version, the worker should run as a separate GPU Deployment.

The exact vLLM command may change by vLLM version, but the intended shape is:

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

The gateway then calls:

```text
http://vllm-nano:8000/v1/embeddings
http://vllm-large-shim:8000/v1/embeddings
```

## Notes

This gateway intentionally tokenizes both logical models with `voyageai/voyage-4-nano`.

That is because Phase 1 has only one real backend model. The logical `voyage-4-large` route is a platform shim for studying model routing and model-specific scaling boundaries, not a claim that the backend model is actually `voyage-4-large`.
