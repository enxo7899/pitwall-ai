"""Publish the adapter and the live-demo engine to the Hugging Face Hub.

Usage:
    uv run --env-file .env python scripts/tools/publish_hf.py model [--push]
    uv run --env-file .env python scripts/tools/publish_hf.py space [--push]

Without --push the folder is only assembled under build/, so it can be inspected first.
`model` uploads the epoch-5 adapter (refusing any file whose SHA-256 differs from the one the audit
checked) with the model card; `space` uploads space/app.py plus the project's own library modules,
pinned to the adapter revision that is live on the Hub, on free ZeroGPU hardware.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

from pitwall.config import MODEL_REVISION

USER = "enxo7899"
MODEL_REPO = f"{USER}/pitwall-qwen3-1.7b-lora"
SPACE_REPO = f"{USER}/pitwall-ai"
GITHUB = f"https://github.com/{USER}/pitwall-ai"
WEBSITE = "https://pitwall-ai-two.vercel.app"
ADAPTER_DIR = Path("outputs/sweep/checkpoint-315")
AUDIT = Path("experiments/audit/audit_report.json")
BUILD = Path("build")
LIBRARY = ["__init__.py", "config.py", "templates.py", "style.py", "generate.py"]  # what the engine imports


def build_model() -> Path:
    out = BUILD / "hf_model"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    weights = ADAPTER_DIR / "adapter_model.safetensors"
    expected = json.loads(AUDIT.read_text())["adapter"]["adapter_sha256"]
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != expected:
        sys.exit(f"adapter weights {actual[:12]}… differ from the audited {expected[:12]}…; not publishing")
    shutil.copy(weights, out / weights.name)
    config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
    config["revision"] = MODEL_REVISION  # pin the base model, as in every experiment
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    card = Path("docs/MODEL_CARD.md").read_text().replace('"<this repository>"', f'"{MODEL_REPO}"')
    links = (
        f"**Code, data and full evaluation:** [{GITHUB.removeprefix('https://')}]({GITHUB}) · "
        f"**Live demo:** [{WEBSITE.removeprefix('https://')}]({WEBSITE})\n\n"
    )
    title_end = card.index("\n", card.index("# PITWALL AI")) + 1
    (out / "README.md").write_text(card[:title_end] + "\n" + links + card[title_end:].lstrip("\n"))
    return out


def build_space(adapter_revision: str | None) -> Path:
    out = BUILD / "hf_space"
    shutil.rmtree(out, ignore_errors=True)
    (out / "pitwall").mkdir(parents=True)
    for name in ("app.py", "README.md", "requirements.txt"):
        shutil.copy(Path("space") / name, out / name)
    for name in LIBRARY:
        shutil.copy(Path("src/pitwall") / name, out / "pitwall" / name)
    if adapter_revision:
        (out / "adapter_revision.txt").write_text(adapter_revision + "\n")
    return out


def main() -> None:
    target, push = sys.argv[1], "--push" in sys.argv
    api = HfApi()
    if target == "model":
        folder = build_model()
        if push:
            api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
            commit = api.upload_folder(repo_id=MODEL_REPO, folder_path=folder, commit_message="PITWALL AI adapter")
            print(f"pushed {MODEL_REPO} at {commit.oid}")
    elif target == "space":
        revision = api.model_info(MODEL_REPO).sha if push else None
        folder = build_space(revision)
        if push:
            api.create_repo(
                SPACE_REPO, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True
            )
            commit = api.upload_folder(
                repo_id=SPACE_REPO, repo_type="space", folder_path=folder, commit_message="PITWALL AI engine"
            )
            print(f"pushed {SPACE_REPO} at {commit.oid}, adapter pinned to {revision}")
    else:
        sys.exit("usage: publish_hf.py model|space [--push]")
    print(f"assembled {folder}: {sorted(p.name for p in folder.rglob('*') if p.is_file())}")


if __name__ == "__main__":
    main()
