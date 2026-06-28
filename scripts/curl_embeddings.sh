#!/usr/bin/env bash
set -euo pipefail

curl --request POST \
  --url "${GATEWAY_URL:-http://localhost:8000}/v1/embeddings" \
  --header "Authorization: Bearer ${LOCAL_API_KEY:-local-dev-key}" \
  --header "content-type: application/json" \
  --data '{
    "input": [
      "Sample text 1",
      "Sample text 2"
    ],
    "model": "voyage-4-large",
    "input_type": "query"
  }'
