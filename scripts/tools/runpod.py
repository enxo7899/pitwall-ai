"""Create, inspect and delete the GPU pod through RunPod's REST API, using RUNPOD_API_KEY from .env.

Usage:
    python3 scripts/tools/runpod.py create [gpu_type] [disk_gb] [name]   # default: one RTX 4090, 50 GB, Secure Cloud
    python3 scripts/tools/runpod.py status <pod_id>   # price, state, and the SSH command once ready
    python3 scripts/tools/runpod.py delete <pod_id>   # removes the pod and its disk; billing stops
    python3 scripts/tools/runpod.py balance           # account balance and current spend per hour
    python3 scripts/tools/runpod.py gpus [min_gb]     # Secure Cloud GPU types with price and stock

The API key is read from .env and never printed. Standard library only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from check_env import load_dotenv  # same folder

ROOT = Path(__file__).resolve().parents[2]  # scripts/tools/<file> -> repository root
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:1.4.0-cu1290-torch2130-ubuntu2204"  # stable tag; our own locked torch is installed on top
SSH_KEY = Path.home() / ".ssh" / "pitwall_runpod"


def api(method: str, path: str, body: dict | None = None) -> tuple[int, dict | None]:
    """Call the REST API with curl (the documented client; RunPod's firewall rejects Python's default
    web client with Cloudflare error 1010). The key goes in through stdin, never the command line."""
    key = load_dotenv(ROOT / ".env").get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY missing in .env")
    config = f'header = "Authorization: Bearer {key}"\nheader = "Content-Type: application/json"\n'
    command = ["curl", "-s", "-X", method, "-w", "\n%{http_code}", "-K", "-", API + path]
    if body is not None:
        command += ["--data-binary", json.dumps(body)]
    result = subprocess.run(command, input=config, capture_output=True, text=True, timeout=90)
    raw, _, code = result.stdout.rpartition("\n")
    try:
        data = json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError:
        data = {"error": raw}
    return int(code or 0), data


def graphql(query: str) -> dict:
    """Balance and GPU stock live only in RunPod's GraphQL API; same curl + stdin key handling."""
    key = load_dotenv(ROOT / ".env").get("RUNPOD_API_KEY")
    config = f'header = "Authorization: Bearer {key}"\nheader = "Content-Type: application/json"\n'
    command = ["curl", "-s", "-K", "-", "https://api.runpod.io/graphql", "--data-binary", json.dumps({"query": query})]
    result = subprocess.run(command, input=config, capture_output=True, text=True, timeout=90)
    return json.loads(result.stdout)


def create(gpu: str = "NVIDIA GeForce RTX 4090", disk_gb: int = 50, name: str = "pitwall-gate1") -> None:
    body = {
        "name": name,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "imageName": IMAGE,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": (SSH_KEY.with_suffix(".pub")).read_text().strip()},
        "allowedCudaVersions": ["13.0"],  # our locked torch ships CUDA 13.0 libraries
    }
    status, pod = api("POST", "/pods", body)
    if status != 201:
        sys.exit(f"create failed ({status}): {pod}")
    print(f"created pod {pod['id']} | ${pod.get('costPerHr')}/h | status {pod.get('desiredStatus')}")


def show(pod_id: str) -> None:
    status, pod = api("GET", f"/pods/{pod_id}?includeMachine=true")
    if status != 200:
        sys.exit(f"status failed ({status}): {pod}")
    ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
    print(
        f"{pod.get('desiredStatus')} | ${pod.get('costPerHr')}/h | {(pod.get('gpu') or {}).get('displayName')} | ip={ip} ports={ports}"
    )
    if ip and "22" in ports:
        print(f"ssh -i {SSH_KEY} -p {ports['22']} root@{ip}")


def delete(pod_id: str) -> None:
    status, body = api("DELETE", f"/pods/{pod_id}")
    print("deleted" if status in (200, 204) else f"delete failed ({status}): {body}")


def balance() -> None:
    me = graphql("{ myself { clientBalance currentSpendPerHr } }")["data"]["myself"]
    print(f"balance ${me['clientBalance']:.2f} | spending ${me['currentSpendPerHr']:.2f}/h now")


def gpus(min_gb: int = 0) -> None:
    query = (
        "{ gpuTypes { id memoryInGb securePrice lowestPrice(input: {gpuCount: 1, secureCloud: true}) "
        "{ stockStatus uninterruptablePrice } } }"
    )
    for g in sorted(graphql(query)["data"]["gpuTypes"], key=lambda g: g["securePrice"] or 99):
        stock = (g.get("lowestPrice") or {}).get("stockStatus")
        if g["memoryInGb"] >= min_gb and g["securePrice"] and stock:
            print(f"{g['id']:<40} {g['memoryInGb']:>4} GB  ${g['securePrice']:.2f}/h  stock={stock}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "create":
        args = sys.argv[2:]
        create(*(args[:1]), *([int(args[1])] if len(args) > 1 else []), *(args[2:3]))
    elif command == "balance":
        balance()
    elif command == "gpus":
        gpus(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif command in ("status", "delete") and len(sys.argv) == 3:
        (show if command == "status" else delete)(sys.argv[2])
    else:
        sys.exit(__doc__)
