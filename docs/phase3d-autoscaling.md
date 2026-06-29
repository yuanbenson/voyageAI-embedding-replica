# Phase 3D: Prometheus, KEDA, HPA, and GPU Autoscaling Artifacts

Phase 3D wires Phase 3C's token-backlog metrics into Kubernetes autoscaling artifacts.

The goal is to demonstrate the production autoscaling chain:

```text
Redis queue token backlog
  -> metrics exporter exposes Prometheus metrics
  -> Prometheus scrapes /metrics
  -> KEDA Prometheus scaler queries Prometheus
  -> KEDA manages an HPA for the target Deployment
  -> HPA changes pod replicas
  -> pending GPU pods can trigger GKE cluster autoscaler
```

These artifacts are opt-in. Applying observability and CPU worker autoscaling is relatively safe. Applying the optional vLLM GPU scaler can trigger GPU node-pool scale-up and should only be used in a controlled demo.

## Artifacts

```text
k8s/observability/
  prometheus-configmap.yaml
  prometheus-deployment.yaml
  prometheus-service.yaml
  metrics-exporter-deployment.yaml
  metrics-exporter-service.yaml
  query-batch-worker-metrics-service.yaml
  document-batch-worker-metrics-service.yaml

k8s/autoscaling/
  query-worker-scaledobject.yaml
  document-worker-scaledobject.yaml
  kustomization.yaml

k8s/autoscaling/optional/
  vllm-nano-scaledobject.yaml
  warm-gpu-pool-notes.md

scripts/
  validate_phase3d_manifests.sh
  watch_autoscaling.sh
```

## Why a metrics exporter?

The gateway exposes `/metrics`, but KEDA needs a stable metric source for queue backlog. Phase 3D adds a small `embedding-metrics-exporter` Deployment that periodically inspects Redis and updates these Prometheus gauges:

```text
voyage_queue_items{model,workload}
voyage_queue_token_backlog{model,workload}
voyage_queue_oldest_item_age_seconds{model,workload}
voyage_autoscaling_recommended_replicas{model,workload}
voyage_autoscaling_estimated_drain_time_seconds{model,workload}
```

The important scaling signal is token backlog, not message count.

## Safe apply path

Install or verify KEDA separately. This repo does not vendor the KEDA controller manifests.

Then apply observability and CPU worker autoscaling:

```bash
kubectl apply -k k8s/observability
kubectl apply -k k8s/autoscaling
```

Watch status:

```bash
scripts/watch_autoscaling.sh
```

Port-forward Prometheus:

```bash
kubectl -n inference port-forward svc/prometheus 9090:9090
```

Useful Prometheus queries:

```promql
voyage_queue_token_backlog{model="voyage-4-nano",workload="query"}
voyage_queue_token_backlog{model="voyage-4-nano",workload="document"}
voyage_autoscaling_recommended_replicas{model="voyage-4-nano",workload="document"}
```

## KEDA worker scaling

Query worker scaling:

```promql
voyage_queue_token_backlog{model="voyage-4-nano",workload="query"}
```

Threshold:

```text
512 queued query tokens per query worker replica
```

Document worker scaling:

```promql
voyage_queue_token_backlog{model="voyage-4-nano",workload="document"}
```

Threshold:

```text
2048 queued document tokens per document worker replica
```

This keeps the scaling logic aligned with the serving policy:

```text
query lane target batch size: 512 tokens
document lane target batch size: 2048 tokens
```

## Optional GPU autoscaling demo

The optional vLLM scaler is intentionally not included in `k8s/autoscaling/kustomization.yaml`.

Apply only when you are ready for a GPU-costing demo:

```bash
kubectl apply -f k8s/autoscaling/optional/vllm-nano-scaledobject.yaml
```

The scaler uses:

```promql
sum(voyage_queue_token_backlog{model="voyage-4-nano"})
```

with threshold:

```text
8192 queued tokens per vLLM replica
```

If `vllm-nano` scales above the number of available GPU nodes, the extra pod may become Pending. If GKE cluster autoscaler is enabled and the GPU node pool max size allows it, GKE may add another GPU node.

Teardown:

```bash
kubectl delete -f k8s/autoscaling/optional/vllm-nano-scaledobject.yaml
kubectl -n inference scale deployment/vllm-nano --replicas=0
```

## GKE cluster autoscaler setup for a controlled demo

Cold pool:

```bash
gcloud container clusters update voyage-replica   --zone us-central1-a   --project voyage-replica-dev   --node-pool gpu-l4   --enable-autoscaling   --min-nodes 0   --max-nodes 2
```

Return to no-GPU-cost state:

```bash
kubectl -n inference scale deployment/vllm-nano --replicas=0

gcloud container clusters resize voyage-replica   --node-pool=gpu-l4   --num-nodes=0   --zone=us-central1-a   --project=voyage-replica-dev
```

## Validation without running GPUs

You can still validate manifests locally:

```bash
scripts/validate_phase3d_manifests.sh
```

This does not create any cloud resources when used with `kubectl --dry-run=client`.

## Non-goals

Phase 3D does not implement:

```text
image pre-pull
model weight caching
TEIEngine
engine abstraction
actual Triton/CUDA kernels
async /v1/batches
durable queue leases
```

Those belong to later phases.
