#!/usr/bin/env bash
# Deviation D-1: train 6 epochs with validation loss on the held-out examples, saving the adapter every epoch,
# then answer the dev questions with every saved adapter (arm C only; A and B are unaffected).
set -eu
cd /root/pitwall
uv run python scripts/training/train_adapter.py --data data/train/train.jsonl --eval-data data/train/heldout.jsonl \
  --epochs 6 --out outputs/sweep
for ck in outputs/sweep/trainer/checkpoint-* outputs/sweep/adapter; do
  name=$(basename "$ck")
  uv run python scripts/evaluation/generate_arms.py --items data/eval/dev.jsonl --arms C --adapter "$ck" \
    --out "outputs/sweep/dev_${name}.jsonl" --batch 48
done
echo SWEEP_DONE
