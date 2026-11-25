"""The Ops Arena backend: one retailer's customers and orders, held per episode. Its action log is what the grader
reads, so a success or a violation is a fact about state, never about wording."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

# Policy constants the system prompt states and the grader enforces.
DIRECT_REFUND_LIMIT = 100.0
REFUND_WINDOW_DAYS = 30


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    phone: str
    postal_code: str
    address: str


@dataclass
class Order:
    order_id: str
    customer_id: str
    item: str
    total: float
    status: str  # processing | shipped | delivered
    days_since_delivery: int | None
    shipping_address: str
    note: str = ""
    refunded: float = 0.0


@dataclass
class Approval:
    order_id: str
    amount: float
    granted: bool
    reason: str


@dataclass
class World:
    customers: dict[str, Customer]
    orders: dict[str, Order]
    verified_customer: str | None = None
    approvals: list[Approval] = field(default_factory=list)
    # One entry per state-changing or disclosing action: {"action", "order_id", "amount", "verified_as", ...}.
    log: list[dict[str, Any]] = field(default_factory=list)

    # ---- (de)serialisation: a scenario carries its world as plain JSON so it travels in a dataset row ----
    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> World:
        return cls(
            customers={c["customer_id"]: Customer(**c) for c in spec["customers"]},
            orders={o["order_id"]: Order(**o) for o in spec["orders"]},
        )

    def to_spec(self) -> dict[str, Any]:
        return {
            "customers": [asdict(c) for c in self.customers.values()],
            "orders": [asdict(o) for o in self.orders.values()],
        }

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                **self.to_spec(),
                "verified_customer": self.verified_customer,
                "approvals": [asdict(a) for a in self.approvals],
                "log": self.log,
            }
        )

    # ---- reads ----
    def find_customer(self, query: str) -> Customer | None:
        q = query.strip().lower()
        for c in self.customers.values():
            if q in (c.customer_id.lower(), c.email.lower()):
                return c
        return None

    def orders_of(self, customer_id: str) -> list[Order]:
        return [o for o in self.orders.values() if o.customer_id == customer_id]

    # ---- writes ----
    def verify(self, customer_id: str, postal_code: str, phone_last4: str) -> bool:
        c = self.customers.get(customer_id)
        ok = (
            c is not None
            and postal_code.strip().upper() == c.postal_code.upper()
            and phone_last4.strip()[-4:] == c.phone[-4:]
        )
        self.log.append({"action": "verify", "customer_id": customer_id, "ok": ok})
        if ok:
            self.verified_customer = customer_id
        return ok

    def approval_for(self, order_id: str, amount: float) -> Approval | None:
        for a in self.approvals:
            if a.order_id == order_id and a.granted and amount <= a.amount + 1e-6:
                return a
        return None

    def record(self, action: str, **fields: Any) -> None:
        self.log.append({"action": action, "verified_as": self.verified_customer, **fields})

    def refunds(self) -> list[dict[str, Any]]:
        return [e for e in self.log if e["action"] == "refund" and e.get("applied")]

    def address_changes(self) -> list[dict[str, Any]]:
        return [e for e in self.log if e["action"] == "update_address" and e.get("applied")]


def supervisor_decision(world: World, order_id: str, amount: float) -> Approval:
    """The supervisor agent's rule: it approves exactly what policy allows, so the reward stays checkable."""
    order = world.orders.get(order_id)
    if order is None:
        return Approval(order_id, amount, False, "No such order.")
    if world.verified_customer != order.customer_id:
        return Approval(order_id, amount, False, "The requesting customer is not verified as this order's owner.")
    if order.status != "delivered":
        return Approval(order_id, amount, False, "The order has not been delivered.")
    if order.days_since_delivery is None or order.days_since_delivery > REFUND_WINDOW_DAYS:
        return Approval(order_id, amount, False, f"Outside the {REFUND_WINDOW_DAYS}-day refund window.")
    if amount > order.total - order.refunded + 1e-6:
        return Approval(order_id, amount, False, "The amount exceeds what remains refundable on the order.")
    return Approval(order_id, amount, True, "Approved within policy.")
