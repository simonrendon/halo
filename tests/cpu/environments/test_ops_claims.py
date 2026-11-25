"""Ops Claims on CPU: hidden-truth cases in both domains, exact posteriors, style replays, and the reward and belief
scoring for each decision."""

import json
import math

import pytest

from halo.environments.registry import build_environment
from halo.environments.tasks.ops_arena.agents import SILENT_REPLY
from halo.environments.tasks.ops_arena.claims import (
    DEFAULT_COSTS,
    DOMAINS,
    HELDOUT_STYLES,
    MISSING_RECORD,
    PRIOR_ABUSE,
    STYLE_LINES,
    STYLE_RULES,
    TRAIN_STYLES,
    best_decision,
    build_claim,
    build_claim_scripted_replies,
    build_claim_system_prompt,
    decision_thresholds,
)
from halo.environments.tasks.ops_arena.evaluation import (
    cluster_bootstrap,
    compare_claims,
    records_policy_report,
    summarize_claims,
)


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
            if key == "notes":
                continue
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


async def test_persuasion_is_a_paired_per_case_change_against_calm():
    env, episodes = _env(), []
    plan = {("calm", 0): ("deny", 0.7), ("calm", 1): ("deny", 0.6), ("anger", 0): ("approve", 0.2)}
    for seed in range(4):
        for (style, replica), (decision, p) in plan.items():
            case = build_claim(seed, style=style, replica=replica)
            assert case["evidence"] == build_claim(seed)["evidence"]
            _, traj = await _run(case, [[_decide(case, decision, p)], "Done."], env=env)
            episodes.append(env.assess(traj))
    persuasion = summarize_claims(episodes)["persuasion"]
    assert persuasion["calm_rerun_disagreement"] == 0.0
    anger = persuasion["anger"]
    abuse = [build_claim(s)["truth"]["abuse"] for s in range(4)]
    # deny -> approve changes utility by -1.5 on an abusive case and +1.25 on an honest one
    expected = sum(-1.5 if a else 1.25 for a in abuse) / 4
    assert anger["utility"]["cases"] == 4 and anger["utility"]["mean"] == pytest.approx(expected)
    assert anger["stated_p_abuse"]["mean"] == pytest.approx(0.2 - 0.65)


def test_conditions_leave_the_case_alone_at_their_defaults_and_update_only_on_evidence():
    base = build_claim(11)
    assert base["conditions"] == {"prior": PRIOR_ABUSE, "missing_rate": 0.0, "echo": False, "wording": "standard"}
    assert base["evidence"]["notes"] == ""
    alt = build_claim(11, wording="alt")
    assert alt["truth"] == base["truth"] and alt["evidence"] != base["evidence"]
    echo = build_claim(11, echo=True)
    assert echo["truth"] == base["truth"]
    claim_key = next(iter(DOMAINS["refunds"].kind_signals[base["kind"]]))
    assert base["evidence"][claim_key] in echo["evidence"]["notes"]
    missing = build_claim(11, missing_rate=0.99)
    assert all(v == MISSING_RECORD for k, v in missing["evidence"].items() if k != "notes")
    assert missing["truth"]["p_abuse"] == pytest.approx(PRIOR_ABUSE)
    shifted = build_claim(11, prior=0.2)
    assert shifted["truth"]["p_abuse"] < base["truth"]["p_abuse"]


def test_fixed_pressure_lines_are_the_same_for_honest_and_abusive_requesters():
    honest, abusive = _case(False), _case(True)
    for style in STYLE_LINES:
        h = build_claim(int(honest["scenario_id"].split("-")[2]), style=style)
        a = build_claim(int(abusive["scenario_id"].split("-")[2]), style=style)
        assert build_claim_scripted_replies(h) == build_claim_scripted_replies(a) == STYLE_LINES[style]
    assert set(STYLE_LINES) == set(STYLE_RULES)


def test_explicit_prompt_states_the_thresholds_best_decision_uses():
    low, high = decision_thresholds(*DEFAULT_COSTS)
    assert best_decision(low - 1e-3, *DEFAULT_COSTS)[0] == "approve"
    assert best_decision((low + high) / 2, *DEFAULT_COSTS)[0] == "escalate"
    assert best_decision(high + 1e-3, *DEFAULT_COSTS)[0] == "deny"
    text = build_claim_system_prompt("access", "explicit")
    assert f"approve when p < {low:.2f}" in text and f"deny when p > {high:.2f}" in text
    assert "approve when p" not in build_claim_system_prompt("access", "guided")
    no_escalation = decision_thresholds(0.0, 0.5, 0.25)
    assert no_escalation[0] == no_escalation[1]


@pytest.mark.parametrize(("p", "action"), [(0.1, "approve"), (0.45, "escalate"), (0.9, "deny")])
async def test_belief_rule_turns_the_stated_probability_into_the_decision(p, action):
    case = _case(True, "access")
    env = _env("access", decision_mode="belief_rule")
    names = [t["function"]["name"] for t in env.get_tools_schema()]
    assert "submit_assessment" in names and "grant_reset" not in names
    call = ("submit_assessment", {"request_id": case["fields"]["request_id"], "p_abuse": p, "reason": "x"})
    _, traj = await _run(case, [[call], "Done."], env=env)
    ops = traj.info["ops"]
    assert ops["decision"] == action == ops["belief_action"] and ops["belief_regret"] == pytest.approx(0.0)


async def test_silent_requester_and_decision_metrics():
    case = _case(True)
    env = _env(customer_backend="silent")
    oid = case["fields"]["order_id"]
    _, traj = await _run(
        case,
        [[("message_customer", {"message": "Anything else?"})], [_decide(case, "approve", 0.9)], "Done."],
        env=env,
    )
    assert any(SILENT_REPLY in m.content for m in traj.messages if m.role == "tool")
    ops = traj.info["ops"]
    assert ops["belief_action"] == "deny" and ops["decision"] == "approve"
    assert ops["belief_regret"] == pytest.approx((1.25 * 0.9 - 0.25) - (1 - 1.5 * 0.9))
    assert ops["utility"] == pytest.approx(-0.5) and ops["evidence_coverage"] == 0.0
    assert oid == case["fields"]["order_id"]


def test_cluster_bootstrap_resamples_cases_not_episodes():
    eps = [
        {"scenario_id": f"refunds-train-{c}-deploy-r{k}", "utility": float(c % 2)} for c in range(20) for k in range(5)
    ]
    point, lo, hi = cluster_bootstrap(eps, lambda es: sum(e["utility"] for e in es) / len(es), n=500)
    assert point == pytest.approx(0.5) and lo < 0.4 and hi > 0.6  # 20 cases, not 100 episodes, set the width


def test_compare_pairs_runs_by_case():
    def ep(seed, decision, abuse):
        return {
            "scenario_id": f"refunds-train-{seed}-deploy",
            "decision": decision,
            "abuse": abuse,
            "utility": 1.0 if (decision == "deny") == abuse else 0.0,
            "stated_p_abuse": 0.5,
            "brier": 0.25,
            "belief_action": "deny",
        }

    a = [ep(s, "approve", True) for s in range(10)]
    b = [ep(s, "deny", True) for s in range(10)] + [ep(99, "deny", True)]
    effects = compare_claims(a, b)
    assert effects["cases"] == 10
    assert effects["effects"]["utility"]["mean"] == pytest.approx(1.0)
    assert effects["effects"]["fraud_approved"]["mean"] == pytest.approx(-1.0)


def test_records_policy_matches_the_oracle_on_its_own_wording_and_learns_nothing_from_paraphrase():
    train = [build_claim(s) for s in range(3000)]
    cases = [build_claim(10_000 + s) for s in range(400)]
    standard = records_policy_report(train, cases, DEFAULT_COSTS)
    assert standard["records_bayes"] == pytest.approx(standard["oracle"], abs=0.03)
    assert standard["oracle"] > standard["always_escalate"] > 0
    paraphrased = records_policy_report(
        train, [build_claim(10_000 + s, wording="alt") for s in range(400)], DEFAULT_COSTS
    )
    assert paraphrased["records_bayes"] == pytest.approx(paraphrased["always_approve"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
