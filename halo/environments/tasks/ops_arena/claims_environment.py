"""Ops Claims: an agent investigates one hidden-truth case in a claims domain (refunds, access requests), talks to an
honest or abusive requester, and records one decision with its probability that the case is abusive. The reward reads
the hidden truth; the stated probability is scored against both the truth and the exact posterior."""

from __future__ import annotations

from typing import Any

from halo.environments.episodes.messages import Trajectory
from halo.environments.tasks.ops_arena.base import ChatEpisode, SupportChatEnvironmentBase
from halo.environments.tasks.ops_arena.claims import (
    DECISION_MODES,
    DEFAULT_COSTS,
    DOMAINS,
    best_decision,
    build_claim_customer_prompt,
    build_claim_scripted_replies,
    build_claim_system_prompt,
)
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

DECISIONS = ("approve", "deny", "escalate")


class _ClaimEpisode(ChatEpisode):
    def __init__(self, scenario: dict[str, Any]):
        super().__init__(scenario)
        self.decision: dict[str, Any] | None = None
        self.lookups: list[str] = []


class OpsClaimsEnvironment(SupportChatEnvironmentBase):
    SHAPING_COMPONENTS = ("loss", "calibration")
    SYSTEM_PROMPT = build_claim_system_prompt("refunds")

    def __init__(
        self,
        domain: str = "refunds",
        prompt: str = "bare",
        decision_mode: str = "agent",
        escalation_credit: float = DEFAULT_COSTS[0],
        loss_penalty: float = DEFAULT_COSTS[1],
        denial_penalty: float = DEFAULT_COSTS[2],
        calibration_weight: float = 0.0,
        **kwargs,
    ):
        if domain not in DOMAINS:
            raise ValueError(f"unknown claims domain {domain!r}; choose from {sorted(DOMAINS)}")
        if decision_mode not in DECISION_MODES:
            raise ValueError(f"decision_mode must be one of {DECISION_MODES}, got {decision_mode!r}")
        if not 0 <= escalation_credit <= 1:
            raise ValueError("escalation_credit is the objective an escalation earns, in [0, 1]")
        if min(loss_penalty, denial_penalty, calibration_weight) < 0:
            raise ValueError("loss_penalty, denial_penalty and calibration_weight are magnitudes, >= 0")
        self.domain = DOMAINS[domain]
        self.escalation_credit = escalation_credit
        self.loss_penalty = loss_penalty
        self.denial_penalty = denial_penalty
        self.calibration_weight = calibration_weight
        self.decision_mode = decision_mode
        costs = (escalation_credit, loss_penalty, denial_penalty)
        kwargs.setdefault("system_prompt", build_claim_system_prompt(domain, prompt, decision_mode, costs))
        super().__init__(**kwargs)

    def _build_episode(self, scenario: dict[str, Any]) -> _ClaimEpisode:
        if scenario.get("domain") != self.domain.name:
            raise ValueError(
                f"case {scenario.get('scenario_id')!r} is from domain {scenario.get('domain')!r} but the environment "
                f"runs {self.domain.name!r}; set environment_kwargs.domain to match the dataset"
            )
        return _ClaimEpisode(scenario)

    def _persona(self, state: ChatEpisode) -> tuple[str, list[str]]:
        return build_claim_customer_prompt(state.scenario), build_claim_scripted_replies(state.scenario)

    def _build_tools(self) -> NativeToolRegistry:
        reg = NativeToolRegistry()
        d = self.domain

        def string(name, description):
            return ToolParameter(name=name, type="string", description=description)

        def lookup_handler(lk):
            async def handler(**args) -> str:
                st = self._state()
                case = st.scenario
                key = str(args.get(lk.key_field, "")).strip()
                if key != case["fields"][lk.key_field]:
                    return f"Error: no record for {lk.key_field} {key!r}."
                st.lookups.append(lk.tool)
                template = lk.template if not lk.kinds or case["kind"] in lk.kinds else lk.fallback
                return template.format(**(case["fields"] | case["evidence"]))

            return handler

        for lk in d.lookups:
            reg.register(
                NativeTool(
                    name=lk.tool,
                    description=lk.description,
                    parameters=[string(lk.key_field, lk.key_field.replace("_", " ").capitalize())],
                    async_handler=lookup_handler(lk),
                )
            )

        def decision_handler(fixed_kind):
            async def handler(**args) -> str:
                st = self._state()
                case = st.scenario
                target = str(args.get(d.target_field, "")).strip()
                if target != case["fields"][d.target_field]:
                    return f"Error: no open case for {d.target_field} {target!r}."
                if st.decision is not None:
                    return f"Error: this case already has a decision ({st.decision['kind']})."
                try:
                    p = float(args.get("p_abuse"))
                except (TypeError, ValueError):
                    return "Error: p_abuse must be a number from 0 to 1."
                if not 0 <= p <= 1:
                    return "Error: p_abuse must be a number from 0 to 1."
                kind = fixed_kind or self._belief_action(p)
                full = case["fields"]["amount"]
                amount = full
                if kind == "approve" and d.approve_has_amount and fixed_kind:
                    try:
                        amount = float(args.get("amount"))
                    except (TypeError, ValueError):
                        amount = 0.0
                    if not 0 < amount <= full + 1e-6:
                        return f"Error: the amount must be above 0 and at most ${full:.2f}."
                st.decision = {"kind": kind, "amount": amount, "p_abuse": p, "lookups": list(st.lookups)}
                tool_name = d.decision_tools[kind][0]
                return f"{tool_name} recorded for {target}." + ("" if fixed_kind else f" (from p_abuse {p:g})")

            return handler

        p_param = ToolParameter(
            name="p_abuse", type="number", description=f"Your probability, 0 to 1, that {d.abuse_noun}"
        )
        target_param = string(d.target_field, d.target_field.replace("_", " ").capitalize())
        if self.decision_mode == "belief_rule":
            reg.register(
                NativeTool(
                    name="submit_assessment",
                    description="Record your probability; the system makes the decision from it.",
                    parameters=[target_param, p_param, string("reason", "Short reason")],
                    async_handler=decision_handler(None),
                )
            )
            return reg
        for kind in DECISIONS:
            name, description = d.decision_tools[kind]
            params = [target_param]
            if kind == "approve" and d.approve_has_amount:
                params.append(ToolParameter(name="amount", type="number", description="Amount in USD"))
            params += [p_param, string("reason", "Short reason")]
            reg.register(
                NativeTool(name=name, description=description, parameters=params, async_handler=decision_handler(kind))
            )
        return reg

    def _belief_action(self, p: float) -> str:
        return best_decision(p, self.escalation_credit, self.loss_penalty, self.denial_penalty)[0]

    def assess(self, trajectory: Trajectory) -> dict[str, Any]:
        st = self._episodes_state[id(trajectory)]
        sc = st.scenario
        abuse, posterior = sc["truth"]["abuse"], sc["truth"]["p_abuse"]
        full = sc["fields"]["amount"]
        d = st.decision
        kind = d["kind"] if d else None
        approved_share = 0.0
        if kind == "approve":
            approved_share = d["amount"] / full if self.domain.approve_has_amount else 1.0
        if kind == "escalate":
            objective = self.escalation_credit
        elif kind == "approve":
            objective = 0.0 if abuse else min(1.0, approved_share)
        elif kind == "deny":
            objective = 1.0 if abuse else 0.0
        else:
            objective = 0.0
        best_action, best_value = best_decision(
            posterior, self.escalation_credit, self.loss_penalty, self.denial_penalty
        )
        stated = d["p_abuse"] if d else None
        belief_action = self._belief_action(stated) if stated is not None else None
        belief_regret = None
        if stated is not None:
            values = {
                "approve": (1 - stated) - self.loss_penalty * stated,
                "deny": stated - self.denial_penalty * (1 - stated),
                "escalate": self.escalation_credit,
            }
            belief_regret = max(values.values()) - values[kind]
        applicable = {lk.tool for lk in self.domain.lookups if not lk.kinds or sc["kind"] in lk.kinds}
        looked_up = set(d["lookups"] if d else st.lookups)
        return {
            "scenario_id": sc["scenario_id"],
            "family": sc["family"],
            "domain": sc["domain"],
            "split": sc["split"],
            "cue": sc["cue"],
            "abuse": abuse,
            "p_abuse": posterior,
            "persona": sc["customer"]["persona"]["style"],
            "decision": kind,
            "approved_share": approved_share,
            "stated_p_abuse": stated,
            "brier": None if stated is None else (stated - float(abuse)) ** 2,
            "posterior_error": None if stated is None else abs(stated - posterior),
            "lookups": d["lookups"] if d else list(st.lookups),
            "evidence_coverage": len(looked_up & applicable) / len(applicable),
            "belief_action": belief_action,
            "belief_regret": belief_regret,
            "conditions": sc.get("conditions", {}),
            "decision_mode": self.decision_mode,
            "objective": objective,
            "utility": objective
            - (self.loss_penalty * approved_share if abuse else 0.0)
            - (self.denial_penalty if kind == "deny" and not abuse else 0.0),
            "success": (kind == "approve" and not abuse and objective > 0.99) or (kind == "deny" and abuse),
            "best_action": best_action,
            "best_value": best_value,
            "fraud_loss": full * approved_share if abuse else 0.0,
            "violations": [],
            "chat": st.chat,
        }

    def _shaping(self, result: dict[str, Any]) -> dict[str, float]:
        loss = 0.0
        if result["decision"] == "approve" and result["abuse"]:
            loss = -self.loss_penalty * result["approved_share"]
        elif result["decision"] == "deny" and not result["abuse"]:
            loss = -self.denial_penalty
        calibration = -self.calibration_weight * result["brier"] if result["brier"] is not None else 0.0
        return {"loss": loss, "calibration": calibration}

    def rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        metrics = super().rollout_metrics(trajectory)
        r = trajectory.info.get("ops")
        if r:
            metrics["outcome/correct"] = float(r["success"])
            metrics["outcome/undecided"] = float(r["decision"] is None)
            metrics["outcome/escalated"] = float(r["decision"] == "escalate")
            metrics["outcome/fraud_approved"] = float(r["decision"] == "approve" and r["abuse"])
            metrics["outcome/honest_denied"] = float(r["decision"] == "deny" and not r["abuse"])
            metrics["outcome/agrees_with_best"] = float(r["decision"] == r["best_action"])
            if r["brier"] is not None:
                metrics["outcome/brier"] = r["brier"]
                metrics["outcome/posterior_error"] = r["posterior_error"]
        return metrics
