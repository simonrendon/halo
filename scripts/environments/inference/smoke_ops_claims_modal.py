#!/usr/bin/env python
"""Smoke-test the Ops Claims recipe on Modal: serve gpt-oss-20b on one H100 with Halo's vLLM image and the recipe's
parsers, then run Halo's eval episode loop (eval_env.py under the recipe) against it, all inside Modal.

    halo run prepare-ops-claims-dataset --domain refunds --out data/ops_claims/refunds
    modal run scripts/environments/inference/smoke_ops_claims_modal.py --num-examples 20 [--customer llm]

``--customer llm`` passes ``$OPENROUTER_API_KEY`` to the eval container as a Modal secret."""

import argparse
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "ops_claims"
MODEL = "unsloth/gpt-oss-20b-BF16"
RECIPE = "examples/grpo/environmental/gptoss/vllm/gptoss-20b-ops-claims-lora-ep1.yaml"
VLLM_IMAGE = "public.ecr.aws/whitecircle/halo:vllm-0.32.0"
SERVER_START_TIMEOUT_S = 1800  # a first start downloads the ~40 GB checkpoint into the cache volume

app = modal.App("halo-ops-claims-smoke")
hf_cache = modal.Volume.from_name("halo-hf-cache", create_if_missing=True)

server_image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    .add_local_file(REPO / "jinja-templates/gpt-oss/gpt-oss-harmony.jinja", "/templates/gpt-oss-harmony.jinja")
)
client_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.11,<2.12",
        "transformers>=5.20.2,<5.21",
        "trl>=1.14.2,<1.15",
        "accelerate",
        "peft",
        "datasets",
        "openai",
        "mcp",
        "ruamel-yaml",
        "pydantic",
        "aiohttp",
        "backoff",
        "pandas",
        "pyarrow",
        "tabulate",
        "wandb",
        "multiprocess",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_dir(REPO, "/halo", ignore=[".git", "**/__pycache__", "checkpoints", "results", "data"])
    .add_local_dir(DATA, "/data")
)


@app.function(
    image=server_image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=3600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=8000, startup_timeout=SERVER_START_TIMEOUT_S)
def serve():
    # The flags the recipe's header names, so the eval exercises the parsers training rolls out through.
    cmd = [
        "vllm", "serve", MODEL, "--port", "8000", "--gpu-memory-utilization", "0.9", "--dtype", "auto",
        "--moe-backend", "triton", "--return-tokens-as-token-ids", "--logprobs-mode", "processed_logprobs",
        "--enable-prefix-caching", "--enable-return-routed-experts", "--enable-auto-tool-choice",
        "--tool-parser-plugin", "/opt/gpt_oss_text_tool_parser.py", "--tool-call-parser", "gpt_oss_text",
        "--reasoning-parser-plugin", "/opt/gpt_oss_reasoning_parser.py", "--reasoning-parser", "openai_gptoss",
        "--chat-template", "/templates/gpt-oss-harmony.jinja",
    ]  # fmt: skip
    subprocess.Popen(cmd, env={**os.environ, "VLLM_ENFORCE_STRICT_TOOL_CALLING": "0", "VLLM_SERVER_DEV_MODE": "1"})


@app.function(
    image=client_image,
    timeout=3600,
    secrets=[modal.Secret.from_dict({"OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")})],
)
def smoke(base_url: str, domain: str, num_examples: int, customer: str) -> str:
    """eval_env.py's report and saved trajectories, after waiting for the server's /health."""
    deadline = time.time() + SERVER_START_TIMEOUT_S
    while True:
        try:
            if urllib.request.urlopen(f"{base_url}/health", timeout=30).status == 200:
                break
        except OSError:
            pass
        if time.time() > deadline:
            return f"the server at {base_url} never answered /health within {SERVER_START_TIMEOUT_S} s"
        time.sleep(15)
    env_kwargs = {"domain": domain} | ({"customer_backend": "scripted"} if customer == "scripted" else {})
    cmd = [
        "python", "scripts/environments/inference/eval_env.py", "--training_config", RECIPE,
        "--dataset", f"/data/{domain}/hf", "--split", "test", "--context_fields", "scenario",
        "--num_examples", str(num_examples), "--base_url", f"{base_url}/v1", "--model", MODEL,
        "--env_kwargs", repr(env_kwargs).replace("'", '"'), "--save_trajectories", "/tmp/trajectories.jsonl",
        "--max_workers", "16",
    ]  # fmt: skip
    out = subprocess.run(cmd, cwd="/halo", env={**os.environ, "PYTHONPATH": "/halo"}, capture_output=True, text=True)
    traj = Path("/tmp/trajectories.jsonl")
    return f"exit {out.returncode}\n{(out.stdout + out.stderr)[-12000:]}\n=====TRAJECTORIES=====\n" + (
        traj.read_text() if traj.exists() else ""
    )


@app.local_entrypoint()
def main(num_examples: int = 20, domain: str = "refunds", customer: str = "scripted"):
    if customer not in ("scripted", "llm"):
        raise SystemExit(f"--customer must be 'scripted' or 'llm', got {customer!r}")
    if customer == "llm" and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("--customer llm reads $OPENROUTER_API_KEY, which is unset")
    if not (DATA / domain / "hf").exists():
        raise SystemExit(f"no {DATA / domain / 'hf'}; run halo run prepare-ops-claims-dataset --domain {domain} first")
    result = smoke.remote(serve.get_web_url(), domain, num_examples, customer)
    out = REPO / "results" / f"smoke-ops-claims-{domain}-{customer}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(result)
    print(result.split("=====TRAJECTORIES=====")[0])
    print(f"full output with trajectories: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-examples", type=int, default=20)
    ap.add_argument("--domain", default="refunds")
    ap.add_argument("--customer", default="scripted", choices=["scripted", "llm"])
    args = ap.parse_args()
    with modal.enable_output(), app.run():
        main.local(args.num_examples, args.domain, args.customer)
