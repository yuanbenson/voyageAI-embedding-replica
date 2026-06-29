#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-inference}"

echo "Watching Phase 3D autoscaling state in namespace: $NS"
echo

echo "ScaledObjects:"
kubectl -n "$NS" get scaledobject 2>/dev/null || true

echo
echo "HPAs:"
kubectl -n "$NS" get hpa 2>/dev/null || true

echo
echo "Pods:"
kubectl -n "$NS" get pods -o wide

echo
echo "Nodes:"
kubectl get nodes

echo
echo "Prometheus query examples:"
cat <<'EOF'
# Port-forward Prometheus:
kubectl -n inference port-forward svc/prometheus 9090:9090

# Then query:
open 'http://localhost:9090/graph?g0.expr=voyage_queue_token_backlog'
open 'http://localhost:9090/graph?g0.expr=voyage_autoscaling_recommended_replicas'
EOF
