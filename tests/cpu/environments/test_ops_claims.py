"""Ops Claims on CPU: hidden-truth cases, exact posteriors, and the reward for each decision."""

import json
import math

import pytest

from halo.environments.registry import build_environment
from halo.environments.tasks.ops_arena.claims import (
    ACCOUNT_SIGNALS,
    CLAIM_SIGNALS,
    HELDOUT_STRATEGIES,
    PRIOR_ABUSE,
    TRAIN_STRATEGIES,
    best_decision,
    build_claim,
)


def _case(abuse: bool, claim_type: str | None = None, split: str = "train") -> dict:
    for seed in range(1000):
        c = build_claim(seed, split)
        if c["truth"]["abuse"] == abuse and claim_type in (None, c["claim"]["type"]):
            return c
    raise AssertionError("no matching seed")


async def _run(case, turns, **kw):
    env = build_environment("ops_claims", {"customer_backend": "scripted", **kw})
    ids, _ = await env.reset_async([case["prompt"]], [{"scenario": json.dumps(case)}])
    for n, turn in enumerate(turns):
        if isinstance(turn, str):
            await env.step_async(ids, [turn], [{}])
        else:
            calls = [
                {"id": f"c{n}_{i}", "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(turn)
            ]
            await env.step_async(ids, [""], [{"tool_calls": calls}])
    return env, env.get_trajectories(ids)[0]


def test_signal_tables_are_distributions():
    tables = [*ACCOUNT_SIGNALS.values(), *(t for d in CLAIM_SIGNALS.values() for t in d.values())]
    for options in tables:
        assert math.isclose(sum(o[1] for o in options), 1.0) and math.isclose(sum(o[2] for o in options), 1.0)


def test_posterior_is_bayes_over_the_drawn_evidence():
    case = build_claim(3)
    log_odds = math.log(PRIOR_ABUSE / (1 - PRIOR_ABUSE))
    tables = {**ACCOUNT_SIGNALS, **CLAIM_SIGNALS[case["claim"]["type"]]}
    signer = case["customer"]["name"].split()
    for key, text in case["evidence"].items():
        pa, ph = next((o[1], o[2]) for o in tables[key] if o[0].format(signer=f"{signer[0][0]}. {signer[1]}") == text)
        log_odds += math.log(pa / ph)
    assert case["truth"]["p_abuse"] == pytest.approx(1 / (1 + math.exp(-log_odds)))
    assert build_claim(3) == case


def test_heldout_fraudsters_only_use_heldout_strategies():
    for split, allowed in (("train", TRAIN_STRATEGIES), ("heldout", HELDOUT_STRATEGIES)):
        strategies = {
            c["customer"]["persona"]["strategy"]
            for c in (build_claim(s, split) for s in range(300))
            if c["customer"]["persona"]["kind"] == "fraudster"
        }
        assert strategies == set(allowed)


def test_best_decision_escalates_only_when_unsure():
    assert best_decision(0.02, 0.6, 0.5, 0.25)[0] == "approve"
    assert best_decision(0.97, 0.6, 0.5, 0.25)[0] == "deny"
    assert best_decision(0.5, 0.6, 0.5, 0.25)[0] == "escalate"


@pytest.mark.parametrize(
    ("abuse", "decision", "objective", "loss"),
    [
        (False, "approve", 1.0, 0.0),
        (True, "approve", 0.0, -0.5),
        (True, "deny", 1.0, 0.0),
        (False, "deny", 0.0, -0.25),
        (True, "escalate", 0.6, 0.0),
        (False, "escalate", 0.6, 0.0),
    ],
)
async def test_reward_follows_the_hidden_truth(abuse, decision, objective, loss):
    case = _case(abuse)
    oid, total = case["claim"]["order_id"], case["claim"]["total"]
    args = {
        "approve": ("approve_refund", {"order_id": oid, "amount": total, "reason": "x"}),
        "deny": ("deny_claim", {"order_id": oid, "reason": "x"}),
        "escalate": ("escalate_claim", {"order_id": oid, "summary": "x"}),
    }[decision]
    _, traj = await _run(case, [[("get_order", {"order_id": oid})], [args], "Done."])
    ops = traj.info["ops"]
    assert ops["decision"] == decision and ops["objective"] == pytest.approx(objective)
    assert ops["lookups"] == ["order"]
    assert traj.total_reward == pytest.approx(objective + loss)


async def test_one_decision_per_claim_and_partial_refunds_pay_partially():
    case = _case(False)
    oid, total = case["claim"]["order_id"], case["claim"]["total"]
    _, traj = await _run(
        case,
        [
            [("approve_refund", {"order_id": oid, "amount": total / 2, "reason": "goodwill"})],
            [("deny_claim", {"order_id": oid, "reason": "changed my mind"})],
            "Done.",
        ],
    )
    ops = traj.info["ops"]
    assert ops["decision"] == "approve" and ops["objective"] == pytest.approx(0.5) and not ops["success"]
    assert any("already has a decision" in m.content for m in traj.messages if m.role == "tool")


async def test_undecided_claim_scores_zero_and_evidence_tools_reveal_the_case():
    case = _case(True, "damaged")
    env, traj = await _run(
        case,
        [
            [
                ("get_account", {"customer_id": case["claim"]["customer_id"]}),
                ("get_return_status", {"order_id": case["claim"]["order_id"]}),
                ("get_order", {"order_id": "NW-000000"}),
            ],
            "I could not decide.",
        ],
    )
    tool_text = [m.content for m in traj.messages if m.role == "tool"]
    assert any(case["evidence"]["return"] in t for t in tool_text)
    assert any(case["evidence"]["prior_claims"] in t for t in tool_text)
    assert any("no order" in t for t in tool_text)
    assert traj.info["ops"]["decision"] is None and traj.info["ops"]["objective"] == 0.0
    assert env.rollout_metrics(traj)["outcome/undecided"] == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
