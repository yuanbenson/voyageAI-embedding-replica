# Warm GPU Pool Simulation Notes

This repo does not enable a warm GPU pool by default. These notes are for a controlled demo only.

## Cold pool

```bash
gcloud container clusters update voyage-replica   --zone us-central1-a   --project voyage-replica-dev   --node-pool gpu-l4   --enable-autoscaling   --min-nodes 0   --max-nodes 2
```

Pros: cheapest idle state.  
Cons: slowest scale-up because the GPU node must be created before the model pod can start.

## Warm node pool

```bash
gcloud container clusters update voyage-replica   --zone us-central1-a   --project voyage-replica-dev   --node-pool gpu-l4   --enable-autoscaling   --min-nodes 1   --max-nodes 2
```

Pros: avoids node scale-from-zero latency.  
Cons: pays for an idle GPU node.

## Warm model replica

```bash
kubectl -n inference scale deployment/vllm-nano --replicas=1
```

Pros: fastest query response because the model is already loaded.  
Cons: most expensive idle mode.

Return to no-GPU-cost state:

```bash
kubectl -n inference scale deployment/vllm-nano --replicas=0

gcloud container clusters resize voyage-replica   --node-pool=gpu-l4   --num-nodes=0   --zone=us-central1-a   --project=voyage-replica-dev
```
