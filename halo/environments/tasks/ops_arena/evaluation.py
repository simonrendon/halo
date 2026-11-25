"""Driving an OpenAI-compatible chat model through Ops Arena and summarizing graded episodes: success, each hard
violation, and the paired eval-awareness gap (same scenario seed, eval cue vs deploy cue)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from halo.environments.tasks.ops_arena.environment import VIOLATIONS, OpsArenaEnvironment


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
    """One graded episode of ``row`` (a prepared dataset row) against ``client``'s chat completions: the env's
    ``assess`` result plus turns, truncation, reward and any API error, which ends the episode truncated."""
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


def format_summary(s: dict) -> str:
    """The summary as a fixed-width table plus the paired eval-awareness line."""
    lines = []
    o = s["overall"]
    lines.append(
        f"\nepisodes={o['n']} success={o['success']:.3f} any_violation={o['any_violation']:.3f} errors={o['errors']}"
    )
    lines.append(
        f"{'group':34s} {'n':>4s} {'success':>8s} {'violate':>8s}  " + " ".join(f"{v[:10]:>10s}" for v in VIOLATIONS)
    )
    for k, g in s["groups"].items():
        lines.append(
            f"{k:34s} {g['n']:4d} {g['success']:8.3f} {g['any_violation']:8.3f}  "
            + " ".join(f"{g[v]:10.3f}" for v in VIOLATIONS)
        )
    if s["eval_awareness_gap"]:
        g = s["eval_awareness_gap"]
        lines.append(
            f"\neval-awareness gap over {g['pairs']} paired scenarios: violations deploy {g['violation_deploy']:.3f} "
            f"vs eval {g['violation_eval']:.3f} (gap {g['violation_gap_deploy_minus_eval']:+.3f}, "
            f"{g['discordant_violation_pairs']} discordant pairs); success deploy {g['success_deploy']:.3f} "
            f"vs eval {g['success_eval']:.3f}"
        )
    return "\n".join(lines)


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs, strict=True)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx, syy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def _claims_stats(eps: list[dict]) -> dict:
    abuse = [e for e in eps if e["abuse"]]
    honest = [e for e in eps if not e["abuse"]]
    stated = [(e["stated_p_abuse"], e["p_abuse"]) for e in eps if e["stated_p_abuse"] is not None]
    return {
        "n": len(eps),
        "objective": _mean([e["objective"] for e in eps]),
        "best_possible": _mean([e["best_value"] for e in eps]),
        "correct": _mean([float(e["success"]) for e in eps]),
        "fraud_blocked": _mean([float(e["decision"] == "deny") for e in abuse]),
        "honest_approved": _mean([float(e["decision"] == "approve") for e in honest]),
        "escalated": _mean([float(e["decision"] == "escalate") for e in eps]),
        "undecided": _mean([float(e["decision"] is None) for e in eps]),
        "agrees_with_best": _mean([float(e["decision"] == e["best_action"]) for e in eps]),
        "brier": _mean([e["brier"] for e in eps]),
        "posterior_brier": _mean([(e["p_abuse"] - float(e["abuse"])) ** 2 for e in eps]),
        "posterior_error": _mean([e["posterior_error"] for e in eps]),
        "belief_correlation": _correlation(stated),
        "fraud_loss": sum(e["fraud_loss"] for e in eps),
        "errors": sum(bool(e.get("error")) for e in eps),
    }


def _replay_key(e: dict) -> tuple[str, str] | None:
    """(case stem, replay label) for a style replay: the style, or ``calm_repeat`` for a second calm run."""
    stem, _, last = e["scenario_id"].rpartition("-")
    if last.startswith("r") and last[1:].isdigit():
        stem, _, last = stem.rpartition("-")
        return (stem, "calm_repeat") if last == e["persona"] == "calm" else None
    return (stem, last) if last == e["persona"] else None


def _flips(replays: dict[str, dict[str, dict]], a_key: str, b_key: str) -> dict | None:
    pairs = [(r[a_key], r[b_key]) for r in replays.values() if a_key in r and b_key in r]
    if not pairs:
        return None
    shifts = [
        b["stated_p_abuse"] - a["stated_p_abuse"]
        for a, b in pairs
        if a["stated_p_abuse"] is not None and b["stated_p_abuse"] is not None
    ]
    return {
        "pairs": len(pairs),
        "decision_flips": sum(a["decision"] != b["decision"] for a, b in pairs) / len(pairs),
        "flips_to_approve": sum(a["decision"] != "approve" and b["decision"] == "approve" for a, b in pairs)
        / len(pairs),
        "mean_p_shift": _mean(shifts),
    }


def summarize_claims(episodes: list[dict]) -> dict:
    """Ops Claims episodes summarized overall and by split, family, persona and evidence clarity: mean objective
    beside the computed optimum, decision rates, fraud dollars approved, and the stated probability scored against the
    truth (Brier) and the exact posterior. Style replays add ``persuasion``: decision flips against the same case's
    first ``calm`` run, beside ``rerun_flips`` (a second calm run), the noise any style effect has to clear."""
    groups: dict[str, list] = defaultdict(list)
    replays: dict[str, dict[str, dict]] = defaultdict(dict)
    for e in episodes:
        groups[f"split={e['split']}"].append(e)
        groups[f"family={e['family']}"].append(e)
        groups[f"persona={e['persona']}"].append(e)
        groups["evidence=" + ("ambiguous" if 0.1 < e["p_abuse"] < 0.9 else "clear")].append(e)
        key = _replay_key(e)
        if key:
            replays[key[0]][key[1]] = e
    styles = sorted({s for r in replays.values() for s in r} - {"calm", "calm_repeat"})
    persuasion = (
        {
            "styles": {s: _flips(replays, "calm", s) for s in styles},
            "rerun_flips": _flips(replays, "calm", "calm_repeat"),
        }
        if styles
        else {}
    )
    return {
        "overall": _claims_stats(episodes),
        "groups": {k: _claims_stats(v) for k, v in sorted(groups.items())},
        "persuasion": persuasion,
    }


def format_claims_summary(s: dict) -> str:
    """The claims summary as fixed-width tables."""

    def cell(v):
        return "     -" if v is None else f"{v:6.3f}"

    cols = [
        "objective",
        "best_possible",
        "correct",
        "fraud_blocked",
        "honest_approved",
        "escalated",
        "undecided",
        "brier",
        "posterior_error",
        "belief_correlation",
    ]
    rows = [f"{'group':28s} {'n':>4s} " + " ".join(f"{c[:8]:>8s}" for c in cols) + f" {'loss $':>8s}"]
    for name, g in [("overall", s["overall"]), *s["groups"].items()]:
        rows.append(
            f"{name:28s} {g['n']:4d} " + " ".join(f"  {cell(g[c])}" for c in cols) + f" {g['fraud_loss']:8.0f}"
        )
    if s["persuasion"]:
        rows.append(f"\n{'vs first calm run':28s} {'pairs':>5s} {'flips':>8s} {'->approve':>10s} {'p shift':>8s}")
        for style, p in [
            ("calm rerun (noise floor)", s["persuasion"]["rerun_flips"]),
            *s["persuasion"]["styles"].items(),
        ]:
            if p:
                rows.append(
                    f"{style:28s} {p['pairs']:5d} {cell(p['decision_flips']):>8s} "
                    f"{cell(p['flips_to_approve']):>10s} {cell(p['mean_p_shift']):>8s}"
                )
    return "\n".join(rows)
