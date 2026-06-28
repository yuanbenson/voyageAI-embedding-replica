# K8s manifests placeholder

Phase 1 code is the Python gateway. The next step is adding GKE manifests:

- `Namespace`
- `Secret` for local API keys and optional Hugging Face token
- `Deployment` for `embedding-api`
- `Service` for `embedding-api`
- `Deployment` for `vllm-nano`
- `Service` for `vllm-nano`
- `Deployment` for `vllm-large-shim`
- `Service` for `vllm-large-shim`
- `Gateway` / `HTTPRoute` or `Ingress`

Keep the gateway CPU-only. Put vLLM workers on GPU node pools.
