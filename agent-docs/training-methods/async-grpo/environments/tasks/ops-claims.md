# Ops Claims

`environment_type: ops_claims` (`OpsClaimsEnvironment`, `halo/environments/tasks/ops_arena/claims_environment.py`).
The policy is a claims agent: it investigates one refund claim, talks to the customer, and records one decision. Each
claim is honest or abusive, and the reward reads that hidden truth, so no rule written into the prompt determines the
right decision. It shares the customer simulator and episode plumbing of [Ops Arena](ops-arena.md)
(`SupportChatEnvironmentBase`). Runbook and pilot results: the
[Ops Arena cookbook](../../../../../human-docs/cookbooks/halo-ops-arena-cookbook.md#ops-claims).

## Cases

`build_claim(seed, split, cue)` draws the truth (abuse with probability `PRIOR_ABUSE`, 0.4), a claim type
(`not_received` or `damaged`), then one option per evidence signal from known likelihoods under that truth
(`ACCOUNT_SIGNALS`, `CLAIM_SIGNALS`): account age, prior claims, recent account changes, and the carrier record
(not received) or the return inspection (damaged). Because the signals are independent given the truth, the exact
posterior `p_abuse` is stored with each case. About 60 % of cases have a posterior below 0.1 or above 0.9.

The customer is an honest customer in one of three moods, or a fraudster with one persuasion strategy. Training
uses `TRAIN_STRATEGIES` (sympathy, anger, confident detail, a claimed prior promise); the `heldout` split uses only
`HELDOUT_STRATEGIES` (chargeback threats, a self-declared retail insider). Fraudsters do not know what the records show.

## Tools

`get_account`, `get_order` (with the carrier record), `get_return_status`, `message_customer`, and the decisions
`approve_refund` (at most the order total), `deny_claim` and `escalate_claim`. A claim takes one decision; later
decisions return an error.

## Reward

| Decision | Honest claim | Abusive claim |
|---|---|---|
| approve (amount `a` of total `t`) | objective `a / t` | objective 0, `reward/loss` `-loss_penalty * a / t` |
| deny | objective 0, `reward/loss` `-denial_penalty` | objective 1 |
| escalate | objective `escalation_credit` | objective `escalation_credit` |
| none | objective 0 | objective 0 |

`best_decision(p_abuse, ...)` gives the action with the highest expected reward at a case's posterior and that
expectation (`best_value`); its mean over cases is the score no agent can beat in expectation.

## Options

| `environment_kwargs` key | Default | Meaning |
|---|---|---|
| `escalation_credit` | 0.6 | objective an escalation earns |
| `loss_penalty` | 0.5 | extra cost of approving an abusive claim |
| `denial_penalty` | 0.25 | extra cost of denying an honest customer |

The customer keys are those of [Ops Arena](ops-arena.md#options); fraudsters need `customer_backend: llm`.

## Metrics and evaluation

`outcome/correct`, `outcome/undecided`, `outcome/escalated`, `outcome/fraud_approved`, `outcome/honest_denied` and
`outcome/agrees_with_best` per episode; the graded case is saved under `info["ops"]`. `halo run
prepare-ops-claims-dataset` writes the data; `halo run eval-ops-arena --environment ops_claims [--guided]` reports the
objective beside the best possible score, by split, claim type, customer persona and whether the evidence is clear.
`--guided` appends `CLAIMS_GUIDANCE`, the qualitative signal briefing that serves as the prompted baseline.
