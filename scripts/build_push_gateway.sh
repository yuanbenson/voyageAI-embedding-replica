#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-voyage-replica}"
IMAGE_TAG="${IMAGE_TAG:-phase2}"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/embedding-api:${IMAGE_TAG}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -t "${IMAGE}" .
docker push "${IMAGE}"

echo "Pushed ${IMAGE}"
echo "Update k8s/base/embedding-api-deployment.yaml image to:"
echo "  ${IMAGE}"
