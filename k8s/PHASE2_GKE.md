# Phase 2: GKE Deployment

Phase 2 deploys the FastAPI gateway to GKE and adds two internal vLLM worker lanes.

```text
Client
  ↓
GKE Gateway / Google Cloud Load Balancer
  ↓
embedding-api
  ↓
model router
  ├── vllm-nano
  └── vllm-large-shim

Both vLLM deployments load:
  voyageai/voyage-4-nano
```

## What Phase 2 proves

- The gateway is reachable through GKE networking.
- vLLM workers are internal-only Kubernetes Services.
- `voyage-4-nano` and `voyage-4-large` route to separate logical worker lanes.
- Both lanes load the same open-weight backend model.
- GPU scheduling works on GKE.
- The public response remains Voyage-compatible.

## What Phase 2 intentionally does not do

- Redis token-aware batching
- KEDA autoscaling
- KServe
- Prometheus/OpenTelemetry
- Batch API
- Reranking

## Deploy flow

### 1. Bootstrap GKE

```bash
PROJECT_ID=<your-gcp-project> \
REGION=us-central1 \
ZONE=us-central1-a \
./scripts/gke_bootstrap.sh
```

### 2. Build and push the gateway image

```bash
PROJECT_ID=<your-gcp-project> \
REGION=us-central1 \
AR_REPO=voyage-replica \
IMAGE_TAG=phase2 \
./scripts/build_push_gateway.sh
```

Update `k8s/base/embedding-api-deployment.yaml` with the pushed image:

```text
us-central1-docker.pkg.dev/<your-gcp-project>/voyage-replica/embedding-api:phase2
```

### 3. Create Kubernetes secrets

```bash
LOCAL_API_KEYS=local-dev-key \
HUGGING_FACE_HUB_TOKEN=<optional-hf-token> \
./scripts/create_k8s_secrets.sh
```

### 4. Deploy manifests

```bash
kubectl apply -k k8s/base
```

### 5. Watch pods

```bash
kubectl -n inference get pods -w
```

### 6. Check Gateway address

```bash
kubectl -n inference get gateway embedding-gateway
```

Once an address appears:

```bash
GATEWAY_HOST=<gateway-ip-or-host>

curl --request POST \
  --url "http://${GATEWAY_HOST}/v1/embeddings" \
  --header "Authorization: Bearer local-dev-key" \
  --header "content-type: application/json" \
  --data '{
    "input": ["Sample text 1", "Sample text 2"],
    "model": "voyage-4-large",
    "input_type": "query",
    "output_dimension": 256
  }'
```

## Debug commands

```bash
kubectl -n inference describe pod -l app=vllm-nano
kubectl -n inference logs -l app=vllm-nano -f

kubectl -n inference describe pod -l app=embedding-api
kubectl -n inference logs -l app=embedding-api -f
```

Port-forward the gateway:

```bash
kubectl -n inference port-forward svc/embedding-api 8000:80
curl http://localhost:8000/healthz
```

Port-forward vLLM directly:

```bash
kubectl -n inference port-forward svc/vllm-nano 8001:8000
curl http://localhost:8001/health
```

## Cost cleanup

```bash
kubectl delete namespace inference
gcloud container clusters delete voyage-replica --zone us-central1-a
```
