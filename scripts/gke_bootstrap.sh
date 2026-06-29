#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
CLUSTER="${CLUSTER:-voyage-replica}"
AR_REPO="${AR_REPO:-voyage-replica}"

gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  artifactregistry.googleapis.com

if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Voyage replica images"
fi

if ! gcloud container clusters describe "${CLUSTER}" --zone="${ZONE}" >/dev/null 2>&1; then
  gcloud container clusters create "${CLUSTER}" \
    --zone="${ZONE}" \
    --release-channel=regular \
    --machine-type=e2-standard-4 \
    --num-nodes=2 \
    --workload-pool="${PROJECT_ID}.svc.id.goog" \
    --gateway-api=standard
fi

if ! gcloud container node-pools describe gpu-l4 \
  --cluster="${CLUSTER}" \
  --zone="${ZONE}" >/dev/null 2>&1; then

  gcloud container node-pools create gpu-l4 \
    --cluster="${CLUSTER}" \
    --zone="${ZONE}" \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1,gpu-driver-version=default \
    --num-nodes=1 \
    --min-nodes=0 \
    --max-nodes=1 \
    --enable-autoscaling \
    --node-labels=accelerator=nvidia-l4 \
    --node-taints=nvidia.com/gpu=present:NoSchedule
fi

gcloud container clusters get-credentials "${CLUSTER}" --zone="${ZONE}"

echo "GKE bootstrap complete."
echo "Cluster: ${CLUSTER}"
echo "Artifact Registry repo: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"
