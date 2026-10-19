"""Scan files that git would commit for anything that looks like a secret.

Usage:
    python3 scripts/tools/check_secrets.py            # every tracked or not-ignored file
    python3 scripts/tools/check_secrets.py --staged   # only files staged for the next commit
                                                # (what the pre-commit hook runs)

Prints file:line and the kind of match, never the matched text. Exit code 1 if
anything is found, so the pre-commit hook blocks the commit.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/tools/<file> -> repository root

PATTERNS: dict[str, re.Pattern[str]] = {
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_\-]{32,}"),
    "Private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # KEY=..., token: "...", secret='...' with a long opaque value
    "Key assignment": re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[A-Za-z0-9_\-./+]{20,}"),
}


def candidate_files(staged_only: bool) -> list[Path]:
    """Files git would commit: staged ones, or all tracked + untracked-but-not-ignored."""
    if staged_only:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    else:
        cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / name for name in out.splitlines() if name]


def scan(path: Path) -> list[tuple[int, str]]:
    """Return (line number, pattern name) for each suspicious line; binary files are skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pattern in PATTERNS.items():
            if pattern.search(line):
                hits.append((lineno, name))
    return hits


def main() -> int:
    staged_only = "--staged" in sys.argv[1:]
    files = candidate_files(staged_only)
    found = 0
    for path in files:
        for lineno, name in scan(path):
            print(f"POSSIBLE SECRET  {path.relative_to(ROOT)}:{lineno}  ({name})")
            found += 1
    scope = "staged files" if staged_only else "files git would commit"
    if found:
        print(f"\n{found} possible secret(s) in {scope}. Commit blocked; remove them first.")
        return 1
    print(f"No secrets found in {len(files)} {scope}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
