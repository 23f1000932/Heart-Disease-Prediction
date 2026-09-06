# Deliverable 6: Performance Monitoring & Stress Testing (wrk)

## Method
`wrk` was used to stress test the live `/predict` endpoint with a
POST + JSON body (via a Lua script), at 2000 concurrent connections
across 8 threads for 30 seconds, with a 10s request timeout.

Command:
    wrk -t8 -c2000 -d30s --timeout 10s --latency -s post.lua http://35.225.151.184/predict

## Results
- Requests/sec: 40.87
- Total completed: 1230 requests in 30.1s
- Socket timeouts: 835
- Latency: avg 6.95s, p50 6.85s, p75 8.41s, p90 9.59s, p99 9.89s

## HPA behavior
The HorizontalPodAutoscaler correctly detected the load spike (CPU
usage hit 130% against a 60% target) and scaled the deployment from
1 pod to the configured maximum of 3 pods within the test window.

## Analysis
Even after scaling to the maximum of 3 pods, the API could not keep
up with 2000 concurrent connections: latency ballooned into multiple
seconds and roughly 40% of requests timed out. The bottleneck is not
GKE or FastAPI itself, but the per-pod resource allocation: each pod
is configured with only 250m CPU request / 500m CPU limit, so even
at 3 pods (the maximum allowed by this deliverable's constraint), the
deployment has at most ~1.5 CPU cores total to handle the load —
insufficient for 2000 concurrent synchronous inference requests.

This demonstrates the autoscaler is functioning correctly, but also
shows that pod *count* alone (capped at 3, per the assignment
requirement) is not sufficient to absorb this level of concurrent
load without either increasing per-pod CPU limits or reducing
per-request latency (e.g., through async request handling, batching,
or a lighter-weight serving layer).
