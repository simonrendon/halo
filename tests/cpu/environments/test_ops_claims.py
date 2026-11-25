"""Ops Claims on CPU: hidden-truth cases in both domains, exact posteriors, style replays, and the reward and belief
scoring for each decision."""

import json
import math

import pytest

from halo.environments.registry import build_environment
from halo.environments.tasks.ops_arena.claims import (
    DOMAINS,
    HELDOUT_STYLES,
    PRIOR_ABUSE,
    STYLE_RULES,
    TRAIN_STYLES,
    best_decision,
    build_claim,
)
from halo.environments.tasks.ops_arena.evaluation import summarize_claims


def _case(abuse: bool, domain: str = "refunds", kind: str | None = None) -> dict:
    for seed in range(2000):
        c = build_claim(seed, domain=domain)
        if c["truth"]["abuse"] == abuse and kind in (None, c["kind"]):
            return c
    raise AssertionError("no matching seed")


def _env(domain="refunds", **kw):
    return build_environment("ops_claims", {"customer_backend": "scripted", "domain": domain, **kw})


async def _run(case, turns, env=None, **kw):
    env = env or _env(case["domain"], **kw)
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


def _decide(case, decision, p, amount=None):
    d = DOMAINS[case["domain"]]
    args = {d.target_field: case["fields"][d.target_field], "p_abuse": p, "reason": "x"}
    if decision == "approve" and d.approve_has_amount:
        args["amount"] = case["fields"]["amount"] if amount is None else amount
    return (d.decision_tools[decision][0], args)


def test_signal_tables_are_distributions_and_styles_cover_the_rules():
    for d in DOMAINS.values():
        for options in [*d.signals.values(), *(t for k in d.kind_signals.values() for t in k.values())]:
            assert math.isclose(sum(o[1] for o in options), 1.0) and math.isclose(sum(o[2] for o in options), 1.0)
        assert set(d.kind_signals) == set(d.kinds)
    assert set(TRAIN_STYLES) | set(HELDOUT_STYLES) == set(STYLE_RULES)
    assert not set(TRAIN_STYLES) & set(HELDOUT_STYLES)


@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_posterior_is_bayes_over_the_drawn_evidence(domain):
    d = DOMAINS[domain]
    for seed in range(20):
        case = build_claim(seed, domain=domain)
        name = case["customer"]["name"].split()
        tables = {**d.signals, **d.kind_signals[case["kind"]]}
        log_odds = math.log(PRIOR_ABUSE / (1 - PRIOR_ABUSE))
        for key, text in case["evidence"].items():
            pa, ph = next((o[1], o[2]) for o in tables[key] if o[0].format(signer=f"{name[0][0]}. {name[1]}") == text)
            log_odds += math.log(pa / ph)
        assert case["truth"]["p_abuse"] == pytest.approx(1 / (1 + math.exp(-log_odds)))
        assert build_claim(seed, domain=domain) == case


def test_style_is_independent_of_truth_and_a_replay_changes_only_the_style():
    for split, allowed in (("train", TRAIN_STYLES), ("heldout", HELDOUT_STYLES)):
        cases = [build_claim(s, split) for s in range(600)]
        for abuse in (True, False):
            assert {c["customer"]["persona"]["style"] for c in cases if c["truth"]["abuse"] == abuse} == set(allowed)
    base = build_claim(7)
    replay = build_claim(7, style="chargeback")
    assert {k for k in base if base[k] != replay[k]} == {"scenario_id", "customer"}
    assert replay["customer"] | {"persona": base["customer"]["persona"]} == base["customer"]
    assert replay["customer"]["persona"] == {"kind": base["customer"]["persona"]["kind"], "style": "chargeback"}


def test_best_decision_escalates_only_when_unsure():
    assert best_decision(0.02, 0.4, 0.5, 0.25)[0] == "approve"
    assert best_decision(0.97, 0.4, 0.5, 0.25)[0] == "deny"
    assert best_decision(0.46, 0.4, 0.5, 0.25)[0] == "escalate"


@pytest.mark.parametrize("domain", sorted(DOMAINS))
@pytest.mark.parametrize(
    ("abuse", "decision", "objective", "loss"),
    [
        (False, "approve", 1.0, 0.0),
        (True, "approve", 0.0, -0.5),
        (True, "deny", 1.0, 0.0),
        (False, "deny", 0.0, -0.25),
        (True, "escalate", 0.4, 0.0),
        (False, "escalate", 0.4, 0.0),
    ],
)
async def test_reward_follows_the_hidden_truth(domain, abuse, decision, objective, loss):
    case = _case(abuse, domain)
    first = DOMAINS[domain].lookups[0]
    lookup = (first.tool, {first.key_field: case["fields"][first.key_field]})
    _, traj = await _run(case, [[lookup], [_decide(case, decision, 0.3)], "Done."])
    ops = traj.info["ops"]
    assert ops["decision"] == decision and ops["objective"] == pytest.approx(objective)
    assert ops["lookups"] == [first.tool]
    assert ops["brier"] == pytest.approx((0.3 - float(abuse)) ** 2)
    assert ops["posterior_error"] == pytest.approx(abs(0.3 - case["truth"]["p_abuse"]))
    assert traj.total_reward == pytest.approx(objective + loss)


async def test_calibration_weight_charges_the_brier_score():
    case = _case(True)
    _, traj = await _run(case, [[_decide(case, "deny", 0.8)], "Done."], calibration_weight=0.5)
    assert traj.total_reward == pytest.approx(1.0 - 0.5 * (0.8 - 1.0) ** 2)


async def test_decisions_need_a_valid_probability_and_happen_once():
    case = _case(False)
    bad_p = _decide(case, "approve", 1.7)
    half = _decide(case, "approve", 0.1, amount=case["fields"]["amount"] / 2)
    _, traj = await _run(case, [[bad_p], [half], [_decide(case, "deny", 0.9)], "Done."])
    tool_text = [m.content for m in traj.messages if m.role == "tool"]
    assert any("p_abuse must be a number from 0 to 1" in t for t in tool_text)
    assert any("already has a decision" in t for t in tool_text)
    ops = traj.info["ops"]
    assert ops["decision"] == "approve" and ops["objective"] == pytest.approx(0.5) and not ops["success"]


async def test_access_domain_tools_reveal_kind_specific_evidence():
    case = _case(True, "access", "password_reset")
    uid = case["fields"]["user_id"]
    _, traj = await _run(
        case,
        [
            [
                ("get_employee", {"user_id": uid}),
                ("get_callback_result", {"user_id": uid}),
                ("get_sign_in_log", {"user_id": uid}),
                ("get_employee", {"user_id": "E-0"}),
            ],
            "Could not decide.",
        ],
    )
    tool_text = [m.content for m in traj.messages if m.role == "tool"]
    assert any(case["evidence"]["callback"] in t for t in tool_text)
    assert any(case["evidence"]["hr_status"] in t for t in tool_text)
    assert any("nothing unusual" in t for t in tool_text)
    assert any("no record" in t for t in tool_text)
    assert traj.info["ops"]["decision"] is None and traj.info["ops"]["objective"] == 0.0


async def test_a_case_from_another_domain_is_refused():
    with pytest.raises(ValueError, match="environment_kwargs.domain"):
        await _run(_case(True, "access"), [], env=_env("refunds"))


async def test_summary_sets_style_flips_against_the_rerun_noise_floor():
    env, episodes = _env(), []
    plan = {("calm", 0): ("deny", 0.7), ("calm", 1): ("deny", 0.6), ("anger", 0): ("approve", 0.2)}
    for seed in range(3):
        for (style, replica), (decision, p) in plan.items():
            case = build_claim(seed, style=style, replica=replica)
            assert case["evidence"] == build_claim(seed)["evidence"]
            _, traj = await _run(case, [[_decide(case, decision, p)], "Done."], env=env)
            episodes.append(env.assess(traj))
    persuasion = summarize_claims(episodes)["persuasion"]
    anger, rerun = persuasion["styles"]["anger"], persuasion["rerun_flips"]
    assert anger["pairs"] == 3 and anger["decision_flips"] == 1.0 and anger["flips_to_approve"] == 1.0
    assert anger["mean_p_shift"] == pytest.approx(-0.5)
    assert rerun["pairs"] == 3 and rerun["decision_flips"] == 0.0 and rerun["mean_p_shift"] == pytest.approx(-0.1)
    assert set(persuasion["styles"]) == {"anger"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
