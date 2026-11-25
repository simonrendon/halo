#!/usr/bin/env python
"""Write Ops Arena datasets as JSON Lines: ``prompt`` plus a ``scenario`` column (the full spec, JSON-encoded).

    halo run prepare-ops-arena-dataset --out data/ops_arena --train-per-family 400 --eval-per-family 25

* ``train.jsonl``: training families only, ``N x TRAIN_MIX[family]`` seeds each, the cue drawn per scenario.
* ``eval.jsonl``: every family at seeds training never used, each scenario twice (``eval`` and ``deploy`` cue,
  same seed), so the eval-awareness gap is a paired comparison. Feed it to ``halo run eval-ops-arena``.
* ``hf/``: a ``save_to_disk`` DatasetDict for the trainer: ``train`` as above, ``test`` a small in-training monitor
  (every family, deploy cue, seeds disjoint from both files above).

Point a recipe at ``./data/ops_arena/hf`` with ``context_fields: [scenario]``.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

from halo.environments.tasks.ops_arena.scenarios import ALL_FAMILIES, CUES, TRAIN_MIX, build_scenario

EVAL_SEED_OFFSET = 1_000_000
MONITOR_SEED_OFFSET = 2_000_000


def row(scenario: dict) -> dict:
    return {
        "prompt": scenario["prompt"],
        "scenario": json.dumps(scenario),
        "family": scenario["family"],
        "cue": scenario["cue"],
        "split": scenario["split"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ops_arena")
    ap.add_argument("--train-per-family", type=int, default=400)
    ap.add_argument("--eval-per-family", type=int, default=25)
    ap.add_argument("--monitor-per-family", type=int, default=10, help="rows per family in the trainer's test split")
    ap.add_argument("--seed", type=int, default=0, help="shuffles the train file only")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = [row(build_scenario(f, s)) for f, w in TRAIN_MIX.items() for s in range(round(args.train_per_family * w))]
    random.Random(args.seed).shuffle(train)
    evals = [
        row(build_scenario(f, EVAL_SEED_OFFSET + s, cue))
        for f in ALL_FAMILIES
        for s in range(args.eval_per_family)
        for cue in CUES
    ]

    for name, rows in (("train", train), ("eval", evals)):
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows -> {out / f'{name}.jsonl'}")

    monitor = [
        row(build_scenario(f, MONITOR_SEED_OFFSET + s, "deploy"))
        for f in ALL_FAMILIES
        for s in range(args.monitor_per_family)
    ]
    DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(monitor)}).save_to_disk(str(out / "hf"))
    print(f"hf: train {len(train)} / test {len(monitor)} rows -> {out / 'hf'}")


if __name__ == "__main__":
    main()
