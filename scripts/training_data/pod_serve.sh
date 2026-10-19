#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server for the teacher or the checker on :8000 and wait until healthy.
# One model at a time: both do not fit on one 80 GB GPU together (56 GB + 65 GB of weights).
#   bash scripts/training_data/pod_serve.sh teacher|checker|stop
set -u
export HF_HOME=/root/hf HF_HUB_OFFLINE=1   # weights already downloaded at pinned revisions; no network at serve time
export PATH=/root/vllm/bin:$PATH        # vLLM JIT-compiles some kernels at warm-up and needs its venv's ninja
COMMON="--port 8000 --seed 2026 --max-model-len 8192 --gpu-memory-utilization 0.92"
case "$1" in
  teacher) ARGS="Qwen/Qwen3.5-27B --revision fc05daec18b0a78c049392ed2e771dde82bdf654 --served-model-name teacher --language-model-only --max-num-seqs 128";;
  checker) ARGS="openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a --served-model-name checker --max-num-seqs 64";;
  stop) ;;
  *) echo "usage: $0 teacher|checker|stop"; exit 2;;
esac
pkill -f "vllm serve" && while pgrep -f "vllm serve" >/dev/null; do sleep 2; done
[ "$1" = stop ] && exit 0
setsid nohup /root/vllm/bin/vllm serve $ARGS $COMMON >/root/logs/serve_$1.log 2>&1 </dev/null &
start=$(date +%s)
until curl -sf localhost:8000/health >/dev/null; do
  sleep 5
  if ! pgrep -f "vllm serve" >/dev/null; then echo "server died"; tail -n 40 /root/logs/serve_$1.log; exit 1; fi
done
echo "$1 ready after $(( $(date +%s) - start ))s"
