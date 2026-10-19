"""Gate 8: independent audit of the published claims. Every check either passes or names what failed.

Usage:
    uv run python scripts/tools/audit.py              # data, leakage and metric checks (seconds)
    uv run python scripts/tools/audit.py --adapter    # also reload the adapter on CPU in fp32 (a few minutes)
Writes experiments/audit/audit_report.json.

The metric recomputation deliberately uses plain json and arithmetic rather than src/pitwall/metrics.py,
so a bug in the library could not hide itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from math import comb
from pathlib import Path

OUT = Path("experiments/audit/audit_report.json")
RESULTS = Path("experiments/results")
JUDGED = {
    "A": RESULTS / "judgements/exam-c500_judgements.jsonl",
    "B": RESULTS / "judgements/exam-c500_judgements.jsonl",
    "C": RESULTS / "judgements/exam-c315_judgements.jsonl",
}
ADAPTER = Path(os.environ.get("PITWALL_ADAPTER", "outputs/sweep/checkpoint-315"))  # local adapter folder


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_freeze() -> dict:
    hashes = json.loads(Path("data/eval/hashes.json").read_text())
    prereg = Path("docs/PREREGISTRATION.md").read_text()
    files = {name: hashlib.sha256(Path("data/eval", name).read_bytes()).hexdigest() for name in hashes}
    commits = subprocess.run(
        ["git", "log", "--format=%h", "--", "data/eval/"], capture_output=True, text=True
    ).stdout.split()
    return {
        "files_match_hashes": all(files[n] == h for n, h in hashes.items()),
        "hashes_in_preregistration": all(h in prereg for h in hashes.values()),
        "commits_touching_eval_set": len(commits),
    }


def check_leakage() -> dict:
    from pitwall.teacher import find_leaks

    train = jsonl(Path("data/train/train.jsonl")) + jsonl(Path("data/train/heldout.jsonl"))
    evals = [r for split in ("dev", "test", "safety", "probe") for r in jsonl(Path(f"data/eval/{split}.jsonl"))]
    leaks = find_leaks([t["question"] for t in train], evals)
    text = " ".join(t["question"] + " " + t["answer"] for t in train).lower()
    verbatim = [e["id"] for e in evals if len(e["question"].split()) >= 4 and e["question"].lower().strip() in text]
    return {
        "training_questions": len(train),
        "exam_questions": len(evals),
        "near_duplicates": len(leaks),
        "exam_questions_copied_verbatim": verbatim,
    }


def check_metrics() -> dict:
    summary = json.loads((RESULTS / "summary.json").read_text())
    test_ids = {r["id"] for r in jsonl(Path("data/eval/test.jsonl"))}

    def passed(r: dict) -> bool:
        return (r["style"] or 0) >= 2 and r["correct"] is True and (r["helpful"] or 0) >= 2

    rows, report = {}, {}
    for arm, path in JUDGED.items():
        rows[arm] = {
            r["item_id"]: r
            for r in jsonl(path)
            if r["arm"] == arm and r["sample"] == 0 and r["items"].endswith("test.jsonl")
        }
        n_pass = sum(passed(r) for r in rows[arm].values())
        report[arm] = {
            "covers_frozen_test_set": set(rows[arm]) == test_ids,
            "joint_success": n_pass,
            "matches_summary": n_pass == summary["arms"][arm]["jsr"]["successes"],
        }
    c_only = sum(passed(rows["C"][i]) and not passed(rows["B"][i]) for i in test_ids)
    b_only = sum(passed(rows["B"][i]) and not passed(rows["C"][i]) for i in test_ids)
    n, k = c_only + b_only, min(c_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    reported = summary["comparisons"]["C_vs_B"]["joint_success"]
    report["C_vs_B"] = {
        "difference_pp": round(100 * (c_only - b_only) / len(test_ids), 1),
        "mcnemar_p": round(p, 4),
        "matches_summary": abs(p - reported["mcnemar"]["p_value"]) < 1e-9
        and round(100 * (c_only - b_only) / len(test_ids), 1) == reported["difference_pp"],
    }
    return report


def check_adapter() -> dict:
    import torch
    from peft import PeftModel

    from pitwall.templates import render_prompt
    from pitwall.train import load_base

    model, tokenizer = load_base("cpu", torch.float32)
    model.eval()
    ids = tokenizer(render_prompt(tokenizer, "How do I keep basil alive indoors?"), return_tensors="pt")
    with torch.no_grad():
        base = model(**ids).logits[0, -1].clone()
        peft = PeftModel.from_pretrained(model, str(ADAPTER)).eval()
        with peft.disable_adapter():
            off = peft(**ids).logits[0, -1]
        on = peft(**ids).logits[0, -1]
    lora_b = [p for n, p in peft.named_parameters() if "lora_B" in n]
    return {
        "adapter_sha256": hashlib.sha256((ADAPTER / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "off_equals_base_max_gap": float((off - base).abs().max()),
        "on_changes_output_max_gap": round(float((on - base).abs().max()), 2),
        "lora_parameters": sum(p.numel() for n, p in peft.named_parameters() if "lora_" in n),
        "lora_B_trained": f"{sum(bool(p.abs().sum() > 0) for p in lora_b)}/{len(lora_b)}",
    }


def check_repository() -> dict:
    def git_files(*flags: str) -> list[str]:
        return subprocess.run(["git", "ls-files", *flags], capture_output=True, text=True).stdout.splitlines()

    tracked = git_files()
    # Tracked files that .gitignore says must stay out (secrets, private notes), plus any model weights.
    should_be_ignored = git_files("--cached", "--ignored", "--exclude-standard")
    weights = [f for f in tracked if f.endswith((".safetensors", ".bin", ".pt"))]
    secrets = subprocess.run(["python3", "scripts/tools/check_secrets.py"], capture_output=True, text=True)
    return {
        "tracked_files": len(tracked),
        "ignored_or_weight_files_tracked": sorted(set(should_be_ignored + weights)),
        "secret_scan": secrets.stdout.strip().splitlines()[-1] if secrets.stdout.strip() else secrets.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="store_true")
    args = parser.parse_args()
    report = {
        "freeze": check_freeze(),
        "leakage": check_leakage(),
        "metrics": check_metrics(),
        "repository": check_repository(),
    }
    if args.adapter:
        report["adapter"] = check_adapter()
    elif OUT.exists() and "adapter" in json.loads(OUT.read_text()):
        report["adapter"] = {
            **json.loads(OUT.read_text())["adapter"],
            "note": "carried over from the last --adapter run",
        }
    failures = []
    if not (report["freeze"]["files_match_hashes"] and report["freeze"]["hashes_in_preregistration"]):
        failures.append("evaluation set does not match its fingerprints")
    if report["freeze"]["commits_touching_eval_set"] != 1:
        failures.append("evaluation set changed after the freeze commit")
    if report["leakage"]["near_duplicates"] or report["leakage"]["exam_questions_copied_verbatim"]:
        failures.append("training data overlaps the evaluation set")
    if not all(v["matches_summary"] for v in report["metrics"].values()):
        failures.append("recomputed metrics differ from summary.json")
    if not all(v.get("covers_frozen_test_set", True) for v in report["metrics"].values()):
        failures.append("judgements do not cover the frozen test set")
    if report["repository"]["ignored_or_weight_files_tracked"] or not report["repository"]["secret_scan"].startswith(
        "No secrets"
    ):
        failures.append("repository contains private files or secrets")
    if args.adapter and (
        report["adapter"]["off_equals_base_max_gap"] != 0.0 or report["adapter"]["lora_parameters"] != 17_432_576
    ):
        failures.append("adapter does not reload cleanly")
    report["failures"] = failures
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
