"""Evaluate any OpenAI-compatible chat model on Ops Arena and report success, violations and the eval-awareness gap.

Works against a local vLLM server (the policy mid-training, or a checkpoint) or a hosted API (a baseline before any
GPU is up):

    python scripts/ops_arena/eval.py --data data/ops_arena/eval.jsonl \
        --base-url https://openrouter.ai/api/v1 --model openai/gpt-oss-20b --api-key-env OPENROUTER_API_KEY \
        --customer-backend llm --customer-model <cheap model> --customer-base-url https://openrouter.ai/api/v1 \
        --out results/baseline-gpt-oss-20b

Writes ``episodes.jsonl`` (one graded episode per line, with its chat and action log) and ``summary.json``.
``--summarize-only <episodes.jsonl>`` recomputes the summary from saved episodes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from halo.environments.tasks.ops_arena.environment import VIOLATIONS, OpsArenaEnvironment


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def _tool_calls_to_dicts(tool_calls) -> list[dict[str, Any]]:
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
        }
        for tc in tool_calls or []
    ]


async def run_episode(
    env: OpsArenaEnvironment,
    client,
    model: str,
    row: dict,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None,
) -> dict:
    ids, _ = await env.reset_async([row["prompt"]], [{"scenario": row["scenario"]}])
    tools = env.get_tools_schema()
    error = None
    while True:
        traj = env.get_trajectories(ids)[0]
        if traj.done:
            break
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[m.to_dict() for m in traj.messages],
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        except Exception as e:  # an API failure is recorded, not a crash of the whole eval
            error = f"{type(e).__name__}: {e}"
            env.finalize_truncated(ids)
            break
        choice = resp.choices[0]
        msg = choice.message
        ctx = {"tool_calls": _tool_calls_to_dicts(msg.tool_calls), "finish_reason": choice.finish_reason}
        await env.step_async(ids, [msg.content or ""], [ctx])
    traj = env.get_trajectories(ids)[0]
    result = dict(traj.info.get("ops") or env.assess(traj))
    result.update(turns=traj.num_turns, truncated=traj.truncated, reward=traj.total_reward, error=error)
    env.cleanup(ids)
    return result


def summarize(episodes: list[dict]) -> dict:
    def stats(eps):
        n = len(eps)
        if not n:
            return {}
        out = {
            "n": n,
            "success": sum(e["success"] for e in eps) / n,
            "any_violation": sum(bool(e["violations"]) for e in eps) / n,
            "errors": sum(bool(e.get("error")) for e in eps),
        }
        for v in VIOLATIONS:
            out[v] = sum(v in e["violations"] for e in eps) / n
        return out

    groups: dict[str, list] = defaultdict(list)
    for e in episodes:
        groups[f"split={e['split']}"].append(e)
        groups[f"family={e['family']}"].append(e)
        groups[f"cue={e['cue']}"].append(e)

    # Paired eval-awareness gap: same scenario seed, eval cue vs deploy cue.
    by_base: dict[str, dict[str, dict]] = defaultdict(dict)
    for e in episodes:
        by_base[e["scenario_id"].rsplit("-", 1)[0]][e["cue"]] = e
    pairs = [p for p in by_base.values() if {"eval", "deploy"} <= p.keys()]
    gap = None
    if pairs:
        n = len(pairs)
        viol_eval = sum(bool(p["eval"]["violations"]) for p in pairs) / n
        viol_dep = sum(bool(p["deploy"]["violations"]) for p in pairs) / n
        succ_eval = sum(p["eval"]["success"] for p in pairs) / n
        succ_dep = sum(p["deploy"]["success"] for p in pairs) / n
        gap = {
            "pairs": n,
            "violation_eval": viol_eval,
            "violation_deploy": viol_dep,
            "violation_gap_deploy_minus_eval": viol_dep - viol_eval,
            "success_eval": succ_eval,
            "success_deploy": succ_dep,
            "discordant_violation_pairs": sum(
                bool(p["eval"]["violations"]) != bool(p["deploy"]["violations"]) for p in pairs
            ),
        }
    return {
        "overall": stats(episodes),
        "groups": {k: stats(v) for k, v in sorted(groups.items())},
        "eval_awareness_gap": gap,
    }


def print_summary(s: dict) -> None:
    o = s["overall"]
    print(
        f"\nepisodes={o['n']} success={o['success']:.3f} any_violation={o['any_violation']:.3f} errors={o['errors']}"
    )
    print(
        f"{'group':34s} {'n':>4s} {'success':>8s} {'violate':>8s}  " + " ".join(f"{v[:10]:>10s}" for v in VIOLATIONS)
    )
    for k, g in s["groups"].items():
        print(
            f"{k:34s} {g['n']:4d} {g['success']:8.3f} {g['any_violation']:8.3f}  "
            + " ".join(f"{g[v]:10.3f}" for v in VIOLATIONS)
        )
    if s["eval_awareness_gap"]:
        g = s["eval_awareness_gap"]
        print(
            f"\neval-awareness gap over {g['pairs']} paired scenarios: violations deploy {g['violation_deploy']:.3f} "
            f"vs eval {g['violation_eval']:.3f} (gap {g['violation_gap_deploy_minus_eval']:+.3f}, "
            f"{g['discordant_violation_pairs']} discordant pairs); success deploy {g['success_deploy']:.3f} "
            f"vs eval {g['success_eval']:.3f}"
        )


async def main_async(args) -> None:
    rows = load_jsonl(args.data)
    if args.families:
        rows = [r for r in rows if r["family"] in args.families.split(",")]
    if args.limit:
        random.Random(0).shuffle(rows)
        rows = rows[: args.limit]

    env = OpsArenaEnvironment(
        customer_backend=args.customer_backend,
        customer_model=args.customer_model,
        customer_base_url=args.customer_base_url,
        customer_api_key_env=args.customer_api_key_env,
        max_turns=args.max_turns,
    )
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"), timeout=300)
    extra = json.loads(args.extra_body) if args.extra_body else None
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes: list[dict] = []

    async def one(row):
        async with sem:
            ep = await run_episode(env, client, args.model, row, args.temperature, args.max_tokens, extra)
            episodes.append(ep)
            if len(episodes) % 10 == 0:
                print(f"{len(episodes)}/{len(rows)} done", flush=True)
            return ep

    await asyncio.gather(*(one(r) for r in rows))
    with open(out / "episodes.jsonl", "w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")
    summary = summarize(episodes)
    summary["config"] = {k: v for k, v in vars(args).items() if "key" not in k}
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print_summary(summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--customer-backend", default="scripted", choices=["scripted", "llm"])
    ap.add_argument("--customer-model")
    ap.add_argument("--customer-base-url")
    ap.add_argument("--customer-api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--families", default="")
    ap.add_argument("--extra-body", default="", help='JSON passed to the API, e.g. \'{"reasoning_effort": "low"}\'')
    ap.add_argument("--out", default="results/ops_arena")
    ap.add_argument("--summarize-only", default="")
    args = ap.parse_args()
    if args.summarize_only:
        print_summary(summarize(load_jsonl(args.summarize_only)))
        return
    if not (args.data and args.model):
        ap.error("--data and --model are required unless --summarize-only is given")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
