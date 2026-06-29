#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-inference}"
LOCAL_API_KEYS="${LOCAL_API_KEYS:-local-dev-key}"
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "${NAMESPACE}" create secret generic embedding-gateway-secrets \
  --from-literal=LOCAL_API_KEYS="${LOCAL_API_KEYS}" \
  --from-literal=HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "${NAMESPACE}" create secret generic vllm-secrets \
  --from-literal=HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Created/updated secrets in namespace ${NAMESPACE}."
