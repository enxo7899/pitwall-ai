"""Report which API keys are configured in `.env`, without ever printing their values.

Usage:
    python3 scripts/tools/check_env.py            # every key (all gates)
    python3 scripts/tools/check_env.py --gate 1   # only keys needed up to and including Gate 1

Exit code 0 when every required key is set, 1 otherwise. Uses only the standard
library, so it runs before any packages are installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/tools/<file> -> repository root

# key name -> (first gate that needs it, what it is used for, expected prefix or None if unknown)
REQUIRED: dict[str, tuple[int, str, str | None]] = {
    "RUNPOD_API_KEY": (1, "GPU pods: training + self-hosted teacher", None),
    "ANTHROPIC_API_KEY": (2, "judge + test drafting", "sk-ant-"),
    "HF_TOKEN": (4, "publish the adapter (write access)", "hf_"),
}


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines, skipping blanks and comments. Values are never printed."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def status(value: str | None, prefix: str | None) -> str:
    """Classify a key without revealing it: missing, set, or set but oddly formatted."""
    if not value:
        return "MISSING"
    if prefix and not value.startswith(prefix):
        return f"SET (but does not start with '{prefix}', check it)"
    return f"set ({len(value)} characters)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", type=int, default=4, help="check keys needed up to this gate")
    args = parser.parse_args()

    env_path = ROOT / ".env"
    if not env_path.exists():
        print(f"No .env file at {env_path}. Copy .env.example to .env first.")
        return 1
    values = load_dotenv(env_path)

    all_ok = True
    for key, (gate, purpose, prefix) in REQUIRED.items():
        if gate > args.gate:
            result = f"not needed until Gate {gate}"
        else:
            result = status(values.get(key), prefix)
            all_ok &= result.startswith("set")
        print(f"{key:<18} {result:<48} Gate {gate}: {purpose}")
    print(f"\nKeys for Gate {args.gate} are ready." if all_ok else "\nSome required keys are missing or look wrong.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
