#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Validating Phase 3D observability manifests..."
kubectl apply --dry-run=client -k "$ROOT/k8s/observability"

echo "Validating Phase 3D autoscaling manifests..."
kubectl apply --dry-run=client -k "$ROOT/k8s/autoscaling"

echo "Validating optional vLLM GPU autoscaling manifest..."
kubectl apply --dry-run=client -f "$ROOT/k8s/autoscaling/optional/vllm-nano-scaledobject.yaml"

echo "Phase 3D manifests are syntactically valid for kubectl client dry-run."
