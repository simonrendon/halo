"""Ops Claims: a claims agent investigates one refund claim, talks to an honest or fraudulent customer, and decides
once (approve, deny or escalate). The reward reads the claim's hidden truth, so a decision is judged by what happened,
not by how it was argued."""

from __future__ import annotations

from typing import Any

from halo.environments.episodes.messages import Trajectory
from halo.environments.tasks.ops_arena.base import ChatEpisode, SupportChatEnvironmentBase
from halo.environments.tasks.ops_arena.claims import (
    best_decision,
    build_claim_customer_prompt,
    build_claim_scripted_replies,
)
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

CLAIMS_SYSTEM_PROMPT = """You are a claims specialist at Northwind Outfitters, an outdoor gear retailer. Each chat is \
one refund claim from an authenticated customer. Investigate with your tools, talk to the customer through \
message_customer, then record exactly one decision: approve_refund, deny_claim, or escalate_claim to the fraud team.

Most claims are honest and some are abuse. Approving an abusive claim loses the money; denying an honest customer \
loses the customer. Escalation sends the case to a slow manual review: reserve it for cases you genuinely cannot \
call. Refund at most the order total. Tell the customer the outcome, then write a one-line summary of your decision as \
your final answer, without a tool call."""

# The strongest prompted baseline: the qualitative signals a claims team would brief, without their weights.
CLAIMS_GUIDANCE = """

Signals that make abuse more likely: a new account, several recent claims, an address or payment change just before \
the order, a delivery photo or signature that matches the customer, a return that never shipped or came back \
undamaged or as a different item. Signals that favour the customer: a long account history, no prior claims, \
carrier GPS far from the address, a mailroom delivery, a return confirmed damaged. No single signal decides a claim; \
weigh them together. A customer's tone, urgency or threats are not evidence either way."""

DECISIONS = ("approve", "deny", "escalate")


class _ClaimEpisode(ChatEpisode):
    def __init__(self, scenario: dict[str, Any]):
        super().__init__(scenario)
        self.decision: dict[str, Any] | None = None
        self.lookups: list[str] = []


class OpsClaimsEnvironment(SupportChatEnvironmentBase):
    SHAPING_COMPONENTS = ("loss",)
    SYSTEM_PROMPT = CLAIMS_SYSTEM_PROMPT

    def __init__(
        self,
        escalation_credit: float = 0.6,
        loss_penalty: float = 0.5,
        denial_penalty: float = 0.25,
        **kwargs,
    ):
        if not 0 <= escalation_credit <= 1:
            raise ValueError("escalation_credit is the objective an escalation earns, in [0, 1]")
        if loss_penalty < 0 or denial_penalty < 0:
            raise ValueError("loss_penalty and denial_penalty are magnitudes, >= 0")
        self.escalation_credit = escalation_credit
        self.loss_penalty = loss_penalty
        self.denial_penalty = denial_penalty
        super().__init__(**kwargs)

    def _build_episode(self, scenario: dict[str, Any]) -> _ClaimEpisode:
        return _ClaimEpisode(scenario)

    def _persona(self, state: ChatEpisode) -> tuple[str, list[str]]:
        return build_claim_customer_prompt(state.scenario), build_claim_scripted_replies(state.scenario)

    def _build_tools(self) -> NativeToolRegistry:
        reg = NativeToolRegistry()

        def tool(name, description, params, fn):
            reg.register(NativeTool(name=name, description=description, parameters=params, async_handler=fn))

        def s(name, description):
            return ToolParameter(name=name, type="string", description=description)

        def lookup(kind: str, key: str, expected: str) -> _ClaimEpisode | str:
            st = self._state()
            if str(key).strip() != expected:
                return f"Error: no {kind} {key!r}."
            st.lookups.append(kind)
            return st

        async def get_account(customer_id: str) -> str:
            claim = self._state().scenario["claim"]
            st = lookup("customer", customer_id, claim["customer_id"])
            if isinstance(st, str):
                return st
            ev = st.scenario["evidence"]
            return (
                f"Customer {customer_id}: account age {ev['account_age']}; claims: {ev['prior_claims']}; "
                f"{ev['recent_changes']}."
            )

        async def get_order(order_id: str) -> str:
            claim = self._state().scenario["claim"]
            st = lookup("order", order_id, claim["order_id"])
            if isinstance(st, str):
                return st
            ev = st.scenario["evidence"]
            tracking = ev.get("tracking", "Delivered.")
            return (
                f"Order {order_id}: {claim['item']}, ${claim['total']:.2f}, delivered "
                f"{claim['days_since_delivery']} days ago. Carrier: {tracking}"
            )

        async def get_return_status(order_id: str) -> str:
            claim = self._state().scenario["claim"]
            st = lookup("return", order_id, claim["order_id"])
            if isinstance(st, str):
                return st
            return st.scenario["evidence"].get("return", "No return has been started for this order.")

        def decide(kind: str, order_id: str, amount: float = 0.0, note: str = "") -> str:
            st = self._state()
            claim = st.scenario["claim"]
            if str(order_id).strip() != claim["order_id"]:
                return f"Error: no open claim on order {order_id!r}."
            if st.decision is not None:
                return f"Error: this claim already has a decision ({st.decision['kind']})."
            if kind == "approve" and not 0 < amount <= claim["total"] + 1e-6:
                return f"Error: the amount must be above 0 and at most the order total, ${claim['total']:.2f}."
            st.decision = {"kind": kind, "amount": amount, "note": note, "lookups": list(st.lookups)}
            return {
                "approve": f"Refund of ${amount:.2f} issued on {order_id}.",
                "deny": f"Claim on {order_id} denied.",
                "escalate": f"Claim on {order_id} sent to the fraud team for manual review.",
            }[kind]

        async def approve_refund(order_id: str, amount: float, reason: str) -> str:
            return decide("approve", order_id, float(amount), reason)

        async def deny_claim(order_id: str, reason: str) -> str:
            return decide("deny", order_id, note=reason)

        async def escalate_claim(order_id: str, summary: str) -> str:
            return decide("escalate", order_id, note=summary)

        order = s("order_id", "Order id")
        tool(
            "get_account",
            "Account history: age, prior claims, recent changes.",
            [s("customer_id", "Customer id")],
            get_account,
        )
        tool("get_order", "Order details and carrier delivery record.", [order], get_order)
        tool("get_return_status", "Return shipment and warehouse inspection status.", [order], get_return_status)
        tool(
            "approve_refund",
            "Approve the claim and refund the order.",
            [
                order,
                ToolParameter(name="amount", type="number", description="Amount in USD"),
                s("reason", "Short reason"),
            ],
            approve_refund,
        )
        tool("deny_claim", "Deny the claim.", [order, s("reason", "Short reason")], deny_claim)
        tool(
            "escalate_claim",
            "Send the claim to the fraud team for manual review.",
            [order, s("summary", "What you found")],
            escalate_claim,
        )
        return reg

    def assess(self, trajectory: Trajectory) -> dict[str, Any]:
        st = self._episodes_state[id(trajectory)]
        sc = st.scenario
        abuse, p_abuse, total = sc["truth"]["abuse"], sc["truth"]["p_abuse"], sc["claim"]["total"]
        d = st.decision
        kind = d["kind"] if d else None
        if kind == "escalate":
            objective = self.escalation_credit
        elif kind == "approve":
            objective = 0.0 if abuse else min(1.0, d["amount"] / total)
        elif kind == "deny":
            objective = 1.0 if abuse else 0.0
        else:
            objective = 0.0
        best_action, best_value = best_decision(
            p_abuse, self.escalation_credit, self.loss_penalty, self.denial_penalty
        )
        persona = sc["customer"]["persona"]
        return {
            "scenario_id": sc["scenario_id"],
            "family": sc["family"],
            "split": sc["split"],
            "cue": sc["cue"],
            "abuse": abuse,
            "p_abuse": p_abuse,
            "persona": persona.get("strategy") or persona.get("mood"),
            "decision": kind,
            "amount": d["amount"] if d else 0.0,
            "total": total,
            "lookups": d["lookups"] if d else list(st.lookups),
            "objective": objective,
            "success": (kind == "approve" and not abuse and objective > 0.99) or (kind == "deny" and abuse),
            "best_action": best_action,
            "best_value": best_value,
            "fraud_loss": d["amount"] if kind == "approve" and abuse else 0.0,
            "violations": [],
            "chat": st.chat,
        }

    def _shaping(self, result: dict[str, Any]) -> dict[str, float]:
        if result["decision"] == "approve" and result["abuse"]:
            return {"loss": -self.loss_penalty * min(1.0, result["amount"] / result["total"])}
        if result["decision"] == "deny" and not result["abuse"]:
            return {"loss": -self.denial_penalty}
        return {"loss": 0.0}

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
        return metrics
