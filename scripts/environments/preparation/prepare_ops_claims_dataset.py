#!/usr/bin/env python
"""Write Ops Claims datasets as JSON Lines: ``prompt`` plus a ``scenario`` column (the case, JSON-encoded).

    halo run prepare-ops-claims-dataset --domain refunds --out data/ops_claims/refunds --train 4000 --eval 100

* ``train.jsonl``: training-style cases, cue drawn per case.
* ``eval.jsonl``: unseen training-style seeds plus held-out-style cases, each under both cues.
* ``persuasion.jsonl``: ``--replays`` deploy-cue cases, each replayed under every style in REPLAY_STYLES with the
  evidence and truth unchanged, plus a second ``calm`` run of each case as the rerun noise floor.
* ``stress.jsonl``: ``--stress`` cases under each robustness condition in STRESS_CONDITIONS alone (development).
* ``final.jsonl`` with ``final.manifest.json`` (``--final N``): the frozen test, fresh seeds whose conditions are drawn
  in combination (alternate wording, missing records, redundant evidence, a shifted prior), half with held-out styles.
  Freeze it before tuning and score it once.
* ``hf/``: a ``save_to_disk`` DatasetDict for the trainer, ``test`` a small deploy-cue monitor.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

from halo.environments.tasks.ops_arena.claims import DOMAINS, build_claim

EVAL_SEED_OFFSET = 1_000_000
MONITOR_SEED_OFFSET = 2_000_000
REPLAY_SEED_OFFSET = 3_000_000
REPLAY_STYLES = ("calm", "anger", "sympathy", "authority", "chargeback", "insider")
STRESS_SEED_OFFSET = 4_000_000
FINAL_SEED_OFFSET = 9_000_000
STRESS_CONDITIONS = {
    "alt_wording": {"wording": "alt"},
    "missing_records": {"missing_rate": 0.3},
    "redundant_evidence": {"echo": True},
    "low_prior": {"prior": 0.2},
    "high_prior": {"prior": 0.6},
}


def final_conditions(rng: random.Random) -> dict:
    return {
        "wording": rng.choice(["standard", "alt"]),
        "missing_rate": rng.choice([0.0, 0.2]),
        "echo": rng.random() < 0.5,
        "prior": rng.choice([0.2, 0.4, 0.6]),
    }


def row(case: dict) -> dict:
    return {
        "prompt": case["prompt"],
        "scenario": json.dumps(case),
        "family": case["family"],
        "cue": case["cue"],
        "split": case["split"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="refunds", choices=sorted(DOMAINS))
    ap.add_argument("--out", default="data/ops_claims/refunds")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--eval", type=int, default=100, help="cases per split (training styles, held-out styles)")
    ap.add_argument("--monitor", type=int, default=100)
    ap.add_argument("--replays", type=int, default=50, help="cases replayed under every style in REPLAY_STYLES")
    ap.add_argument("--stress", type=int, default=50, help="cases per condition in STRESS_CONDITIONS")
    ap.add_argument("--final", type=int, default=0, help="frozen test cases to write (0: none)")
    ap.add_argument("--seed", type=int, default=0, help="draws the training cues")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cues = random.Random(args.seed)
    dom = args.domain
    train = [row(build_claim(s, "train", cues.choice(["eval", "deploy"]), dom)) for s in range(args.train)]
    evals = [
        row(build_claim(EVAL_SEED_OFFSET + s, split, cue, dom))
        for split in ("train", "heldout")
        for s in range(args.eval)
        for cue in ("deploy", "eval")
    ]
    replays = [
        row(build_claim(REPLAY_SEED_OFFSET + s, "train", "deploy", dom, style, replica))
        for s in range(args.replays)
        for style, replica in [*((st, 0) for st in REPLAY_STYLES), ("calm", 1)]
    ]
    stress = [
        row(build_claim(STRESS_SEED_OFFSET + s, "train", "deploy", dom, **cond))
        for cond in STRESS_CONDITIONS.values()
        for s in range(args.stress)
    ]
    outputs = [("train", train), ("eval", evals), ("persuasion", replays), ("stress", stress)]
    if args.final:
        draw = random.Random(f"final:{dom}")
        final = [
            row(
                build_claim(
                    FINAL_SEED_OFFSET + s, "heldout" if s % 2 else "train", "deploy", dom, **final_conditions(draw)
                )
            )
            for s in range(args.final)
        ]
        outputs.append(("final", final))
    for name, rows in outputs:
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows -> {out / f'{name}.jsonl'}")
    if args.final:
        digest = hashlib.sha256((out / "final.jsonl").read_bytes()).hexdigest()
        manifest = {"domain": dom, "rows": args.final, "sha256": digest, "seed_offset": FINAL_SEED_OFFSET}
        (out / "final.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"final sha256 {digest}")
    monitor = [row(build_claim(MONITOR_SEED_OFFSET + s, "train", "deploy", dom)) for s in range(args.monitor)]
    DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(monitor)}).save_to_disk(str(out / "hf"))
    print(f"hf: train {len(train)} / test {len(monitor)} rows -> {out / 'hf'}")


if __name__ == "__main__":
    main()
