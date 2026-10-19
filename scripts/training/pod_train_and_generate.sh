#!/usr/bin/env bash
# Gates 4-5 on the GPU pod, unattended: train the adapter(s), then answer the dev and test sets with all three arms.
# Run after `teacher_pipeline.py select` has written data/train/train.jsonl.
set -eu
cd /root/pitwall
bash scripts/training_data/pod_serve.sh stop
T=data/train/train.jsonl
uv run python scripts/training/train_adapter.py --data $T --out outputs/c500 --epochs 3
uv run python scripts/evaluation/generate_arms.py --items data/eval/dev.jsonl --adapter outputs/c500/adapter --out outputs/c500/dev.jsonl --batch 48
echo MAIN_TRAINED
for f in test safety probe; do
  uv run python scripts/evaluation/generate_arms.py --items data/eval/$f.jsonl --adapter outputs/c500/adapter --out outputs/c500/$f.jsonl --k 3 --batch 48
done
echo TEST_GENERATED
for n in 125 250; do  # exploratory learning curve (H4): arm C only, dev only
  uv run python scripts/training/train_adapter.py --data $T --limit $n --out outputs/c$n --epochs 3
  uv run python scripts/evaluation/generate_arms.py --items data/eval/dev.jsonl --arms C --adapter outputs/c$n/adapter --out outputs/c$n/dev.jsonl --batch 48
done
echo ALL_DONE
