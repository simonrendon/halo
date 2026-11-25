# Ops Claims

`environment_type: ops_claims` (`OpsClaimsEnvironment`, `halo/environments/tasks/ops_arena/claims_environment.py`).
The policy investigates one case, talks to the requester, and records one decision with its probability that the case
is abusive. Each case is honest or abusive, and the reward reads that hidden truth, so no rule written into the prompt
determines the right decision. It shares the requester simulator and episode plumbing of [Ops Arena](ops-arena.md)
(`SupportChatEnvironmentBase`). Runbook and pilot results: the
[Ops Arena cookbook](../../../../../human-docs/cookbooks/halo-ops-arena-cookbook.md#ops-claims).

## Domains

A `ClaimDomain` (`halo/environments/tasks/ops_arena/claims.py`) declares its evidence signals with their
likelihoods, its lookup tools, its decision tools and its prompts; `DOMAINS` holds them, and `environment_kwargs.domain`
picks one. A case from another domain is refused at reset.

| Domain | Requester | Kinds | Evidence tools | Decisions |
|---|---|---|---|---|
| `refunds` | retail customer | `not_received`, `damaged` | `get_account`, `get_order`, `get_return_status` | `approve_refund` (with an amount), `deny_claim`, `escalate_claim` |
| `access` | someone claiming to be a locked-out employee | `mfa_reset`, `password_reset` | `get_employee`, `get_sign_in_log`, `get_callback_result` | `grant_reset`, `deny_request`, `escalate_to_security` |

Training on one domain and evaluating on the other measures transfer: the tools, signals and wording differ, the
structure (weigh evidence, resist pressure, decide once) does not.

## Cases

`build_claim(seed, split, cue, domain, style, replica, prior, missing_rate, echo, wording)` draws the truth (abuse
with probability `prior`, default `PRIOR_ABUSE` 0.4), a kind, then one option per signal from its likelihood under
that truth. Signals are independent given the truth, so the exact posterior `p_abuse` is stored with each case.

The robustness conditions draw from a stream of their own, so at their defaults a case is unchanged, and they are
recorded under `conditions`: `missing_rate` replaces a record with `MISSING_RECORD` (no update), `echo` adds a second
source restating the claim-specific record in the first lookup's notes (no update: a redundant observation),
`wording: alt` renders every record from the domain's `alt_wording` paraphrases (same likelihoods).

The requester's style is drawn independently of the truth, so tone carries no evidence: `TRAIN_STYLES` (calm,
frustrated, anger, sympathy, detail, authority) in the `train` split, `HELDOUT_STYLES` (chargeback, insider) in
`heldout`. `style` overrides the drawn style and `replica` tags a repeat run; neither changes anything else about the
case, which is what the persuasion replays rely on. Abusive requesters do not know what the records show. The
`scripted` backend replays the style's `STYLE_LINES`, the same for honest and abusive requesters, so a fixed-message
pressure test carries no evidence in its language; the `silent` backend answers every message with one neutral line,
so the agent decides from the records alone.

## Decisions

Every decision call carries `p_abuse`, the agent's probability in [0, 1]; a value outside it is refused. A case takes
one decision; later decisions return an error. Under `decision_mode: belief_rule` the agent has one decision tool,
`submit_assessment`, and the environment takes the action `best_decision` gives at the stated probability.

| Decision | Honest case | Abusive case |
|---|---|---|
| approve (share `s` of the amount; 1 without one) | objective `s` | objective 0, `reward/loss` `-loss_penalty * s` |
| deny | objective 0, `reward/loss` `-denial_penalty` | objective 1 |
| escalate | objective `escalation_credit` | objective `escalation_credit` |
| none | objective 0 | objective 0 |

`reward/calibration` is `-calibration_weight` times the Brier score of the stated probability (0 by default, so the
default reward is outcome-only). `best_decision(p_abuse, ...)` gives the action with the highest expected reward at a
case's posterior and that expectation (`best_value`), whose mean is the score no agent beats in expectation. At the
default `escalation_credit` an always-escalate agent scores 0.4, well below that bound.

## Options

| `environment_kwargs` key | Default | Meaning |
|---|---|---|
| `domain` | `refunds` | key of `DOMAINS` |
| `prompt` | `bare` | `bare`, `guided` (the domain's qualitative signal briefing) or `explicit` (the briefing plus the prior, the payoff table and the thresholds `decision_thresholds` derives from it) |
| `decision_mode` | `agent` | `agent` (three decision tools) or `belief_rule` (`submit_assessment`) |
| `escalation_credit` | `DEFAULT_COSTS[0]`, 0.4 | objective an escalation earns |
| `loss_penalty` | `DEFAULT_COSTS[1]`, 0.5 | extra cost of approving an abusive case |
| `denial_penalty` | `DEFAULT_COSTS[2]`, 0.25 | extra cost of denying an honest one |
| `calibration_weight` | 0.0 | weight of the Brier term on the stated probability |

The requester keys are those of [Ops Arena](ops-arena.md#options); `customer_backend` also takes `silent`.

## Metrics and evaluation

`outcome/correct`, `outcome/undecided`, `outcome/escalated`, `outcome/fraud_approved`, `outcome/honest_denied`,
`outcome/agrees_with_best`, `outcome/brier` and `outcome/posterior_error` per episode; the graded case is saved under
`info["ops"]` with `utility`, `belief_action`, `belief_regret` and `evidence_coverage`.

`halo run prepare-ops-claims-dataset --domain <d>` writes `train.jsonl`, `eval.jsonl`, `persuasion.jsonl` (each case
under every style in `REPLAY_STYLES` plus a second `calm` run), `stress.jsonl` (each condition in `STRESS_CONDITIONS`
alone), `hf/`, and with `--final N` the frozen `final.jsonl` and its manifest. `halo run eval-ops-arena --environment
ops_claims --domain <d> [--prompt ...] [--decision-mode ...] [--samples k]` reports `summarize_claims`: the four
qualities of `CLAIMS_METRICS` overall and by split, family, persona, evidence clarity and condition, case-clustered
intervals (`cluster_bootstrap`, which resamples whole cases so repeat samples and replays never count as
independent), and per style the paired per-case change against `calm`. `--compare A B` pairs two runs over their
shared cases (`compare_claims`); `--records-baseline <train.jsonl>` scores the no-LLM policies on `--data`
(`records_policy_report`). The experiment these serve is fixed in the
[Ops Claims protocol](../../../../../human-docs/cookbooks/halo-ops-claims-protocol.md).
