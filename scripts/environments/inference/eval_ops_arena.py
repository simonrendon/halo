#!/usr/bin/env python
"""Evaluate an OpenAI-compatible chat model on Ops Arena or Ops Claims (``--environment``).

    halo run eval-ops-arena --data data/ops_arena/eval.jsonl --base-url http://localhost:8000/v1 --model <id> \\
        --out results/ops_arena

Writes ``episodes.jsonl`` and ``summary.json``; ``--summarize-only <episodes.jsonl>`` re-reads saved episodes."""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from openai import AsyncOpenAI

from halo.environments.tasks.ops_arena.claims import DOMAINS
from halo.environments.tasks.ops_arena.claims_environment import OpsClaimsEnvironment
from halo.environments.tasks.ops_arena.environment import OpsArenaEnvironment
from halo.environments.tasks.ops_arena.evaluation import (
    format_claims_summary,
    format_summary,
    run_episode,
    summarize,
    summarize_claims,
)

SUMMARIES = {"ops_arena": (summarize, format_summary), "ops_claims": (summarize_claims, format_claims_summary)}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh]


async def evaluate(args) -> None:
    rows = load_jsonl(args.data)
    if args.families:
        rows = [r for r in rows if r["family"] in args.families.split(",")]
    if args.limit:
        random.Random(0).shuffle(rows)
        rows = rows[: args.limit]

    customer = {
        "customer_backend": args.customer_backend,
        "customer_model": args.customer_model,
        "customer_base_url": args.customer_base_url,
        "customer_api_key_env": args.customer_api_key_env,
        "max_turns": args.max_turns,
    }
    if args.environment == "ops_claims":
        env = OpsClaimsEnvironment(domain=args.domain, guided=args.guided, **customer)
    else:
        env = OpsArenaEnvironment(**customer)
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"), timeout=300)
    extra = json.loads(args.extra_body) if args.extra_body else None
    sem = asyncio.Semaphore(args.concurrency)
    episodes: list[dict] = []

    async def one(row):
        async with sem:
            episodes.append(await run_episode(env, client, args.model, row, args.temperature, args.max_tokens, extra))
            if len(episodes) % 10 == 0:
                print(f"{len(episodes)}/{len(rows)} done", flush=True)

    await asyncio.gather(*(one(r) for r in rows))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")
    summarize_fn, format_fn = SUMMARIES[args.environment]
    summary = summarize_fn(episodes)
    summary["config"] = {k: v for k, v in vars(args).items() if "key" not in k}
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(format_fn(summary))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--environment", default="ops_arena", choices=sorted(SUMMARIES))
    ap.add_argument("--data", help="eval.jsonl from prepare-ops-arena-dataset or prepare-ops-claims-dataset")
    ap.add_argument("--domain", default="refunds", choices=sorted(DOMAINS), help="ops_claims: the cases' domain")
    ap.add_argument("--guided", action="store_true", help="ops_claims: append the domain's signal briefing")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="variable holding the policy endpoint's key")
    ap.add_argument("--customer-backend", default="scripted", choices=["scripted", "llm"])
    ap.add_argument("--customer-model")
    ap.add_argument("--customer-base-url")
    ap.add_argument("--customer-api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="evaluate a seeded random subset of this many rows")
    ap.add_argument("--families", default="", help="comma-separated families to keep")
    ap.add_argument("--extra-body", default="", help='JSON for the API, e.g. \'{"reasoning_effort": "low"}\'')
    ap.add_argument("--out", default="results/ops_arena")
    ap.add_argument("--summarize-only", default="")
    args = ap.parse_args()
    if args.summarize_only:
        summarize_fn, format_fn = SUMMARIES[args.environment]
        print(format_fn(summarize_fn(load_jsonl(args.summarize_only))))
        return
    if not (args.data and args.model):
        ap.error("--data and --model are required unless --summarize-only is given")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
