#!/usr/bin/env python
"""Write Ops Claims datasets as JSON Lines: ``prompt`` plus a ``scenario`` column (the case, JSON-encoded).

    halo run prepare-ops-claims-dataset --out data/ops_claims --train 4000 --eval 100

* ``train.jsonl``: training-strategy cases, cue drawn per case.
* ``eval.jsonl``: unseen training-strategy seeds plus held-out-strategy cases, each under both cues.
* ``hf/``: a ``save_to_disk`` DatasetDict for the trainer, ``test`` a small deploy-cue monitor.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

from halo.environments.tasks.ops_arena.claims import build_claim

EVAL_SEED_OFFSET = 1_000_000
MONITOR_SEED_OFFSET = 2_000_000


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
    ap.add_argument("--out", default="data/ops_claims")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--eval", type=int, default=100, help="cases per split (train strategies, held-out strategies)")
    ap.add_argument("--monitor", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="draws the training cues")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cues = random.Random(args.seed)
    train = [row(build_claim(s, "train", cues.choice(["eval", "deploy"]))) for s in range(args.train)]
    evals = [
        row(build_claim(EVAL_SEED_OFFSET + s, split, cue))
        for split in ("train", "heldout")
        for s in range(args.eval)
        for cue in ("deploy", "eval")
    ]
    for name, rows in (("train", train), ("eval", evals)):
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows -> {out / f'{name}.jsonl'}")
    monitor = [row(build_claim(MONITOR_SEED_OFFSET + s, "train", "deploy")) for s in range(args.monitor)]
    DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(monitor)}).save_to_disk(str(out / "hf"))
    print(f"hf: train {len(train)} / test {len(monitor)} rows -> {out / 'hf'}")


if __name__ == "__main__":
    main()
