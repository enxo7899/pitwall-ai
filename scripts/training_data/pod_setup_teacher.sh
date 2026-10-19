#!/usr/bin/env bash
# Gate 3 pod setup, run once on a fresh RunPod 80 GB pod (code already copied to /root/pitwall).
# Everything runs in parallel: our training env, a separate vLLM env, and both model downloads.
# Logs go to /root/logs; /root/logs/setup.done appears when all four have finished.
set -u
cd /root/pitwall
mkdir -p /root/logs
export HF_HOME=/root/hf
export UV_LOCK_TIMEOUT=1800   # parallel uv installs share a cache lock; wait instead of failing
pip install -q uv >/root/logs/pip_uv.log 2>&1

TEACHER=Qwen/Qwen3.5-27B;   TEACHER_REV=fc05daec18b0a78c049392ed2e771dde82bdf654
CHECKER=openai/gpt-oss-120b; CHECKER_REV=b5c939de8f754692c1647ca79fbf85e8c1e70f8a

(uv sync --python 3.12 >/root/logs/uv_sync.log 2>&1; echo "exit $?" >>/root/logs/uv_sync.log) &
(uv venv /root/vllm --python 3.12 >/root/logs/vllm_install.log 2>&1 && \
 VIRTUAL_ENV=/root/vllm uv pip install vllm --torch-backend=auto >>/root/logs/vllm_install.log 2>&1; \
 echo "exit $?" >>/root/logs/vllm_install.log) &
(uvx --from huggingface_hub hf download $TEACHER --revision $TEACHER_REV >/root/logs/dl_teacher.log 2>&1; \
 echo "exit $?" >>/root/logs/dl_teacher.log) &
(uvx --from huggingface_hub hf download $CHECKER --revision $CHECKER_REV --exclude "original/*" --exclude "metal/*" \
 >/root/logs/dl_checker.log 2>&1; echo "exit $?" >>/root/logs/dl_checker.log) &
wait
touch /root/logs/setup.done
