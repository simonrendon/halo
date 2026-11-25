# Ops Claims Evaluation Protocol

The hypothesis, controls, metrics, test sets and decision rules for the Ops Claims training experiment, fixed before
any training run or final-set score. Environment reference:
[Ops Claims](../../agent-docs/training-methods/async-grpo/environments/tasks/ops-claims.md) ↗; runbook and baselines:
the [Ops Arena cookbook](halo-ops-arena-cookbook.md#ops-claims).

## Primary hypothesis

Outcome-only GRPO on the `refunds` domain, standard wording, raises decision utility on cases the training never
shows (paraphrased records, combined robustness conditions, held-out requester styles, and the `access` domain) above
the strongest control below, without a regression past the limits under [Acceptable regressions](#acceptable-regressions).

In-distribution gains do not test it: on standard-wording cases a naive Bayes policy fitted to the training records,
with no language model, matches the exact posterior's policy (0.853 against 0.849 on `refunds`, 0.750 against 0.744
on `access`). It drops to the approve-everything score on paraphrased records, so the hypothesis is about what a
language model adds there.

## Measurements

Every episode is graded on four qualities (`CLAIMS_METRICS`), reported before and after training:

| Quality | Measures |
|---|---|
| Outcome | utility (objective plus the loss terms), harmful decisions (abuse approved or honest denied), fraud denied, honest approved, escalation, beside the exact-posterior bound |
| Evidence | Brier score and ECE of the stated `p_abuse` against the truth; its error and correlation against the posterior |
| Decision given belief | share of decisions equal to the cost-optimal action at the agent's own `p_abuse`; regret at that probability |
| Execution | undecided episodes, coverage of the applicable evidence tools, turns, tokens |
| Pressure | per style, the per-case change against `calm` in utility, harmful decisions and stated probability, interactive (LLM requester) and with fixed messages (`customer_backend: scripted`, the same lines for honest and abusive requesters) |

`p_abuse` is an elicited number, not the model's internal belief; a calibration claim rests on the
`calibration_weight` ablation below, not on the outcome-only run. Consistency is reported only beside utility, since
an agent that always denies is perfectly consistent.

## Controls

All at matched serving (the same checkpoint family behind the same vLLM flags, temperature, `max_turns` and token
limits), on the same cases:

| Control | Question it answers |
|---|---|
| `prompt: guided` | the baseline briefing |
| `prompt: explicit` (prior, payoff table, thresholds) | can clearer instructions close the gap? |
| `decision_mode: belief_rule` (probability in, action by `best_decision`) | is action selection the bottleneck? |
| `customer_backend: silent` | how much comes from the dialogue rather than the records? |
| SFT on the base model's best trajectories on `refunds` train, same token budget as the RL run | does RL add value beyond demonstrations? |
| `records_bayes` and `oracle` (`records_policy_report`) | how much of each test needs a language model at all? |
| `calibration_weight: 0.5` run | does probability quality need its own reward? |

## Test sets

| Set | Status |
|---|---|
| `eval.jsonl`, `persuasion.jsonl`, `stress.jsonl` (both domains), Ops Arena eval | development: inspected, used for choices |
| `examples/grpo/environmental/ops-arena/frozen/{refunds,access}-final.jsonl` | frozen: 200 cases each, sha256 in the manifests; scored once, after training and every control are fixed |
| an independently authored part (new wording and at least one mechanism the generator does not have) | to be written by someone who has not seen the generator's tables, before training starts |

The frozen sets draw their conditions in combination per case (wording, missing records, redundant evidence, prior
0.2 / 0.4 / 0.6), half with held-out styles. Their paraphrases reuse the table `stress.jsonl` uses, so only the
combinations, seeds and styles are new to them; the independent part carries the unseen wording. Final scores use
3 samples per case, case-clustered 95 % intervals, the development requester (`anthropic/claude-haiku-5.5`) and one
other requester model named before scoring.

## Decision rules

- **Stop before training** if `prompt: explicit` or `decision_mode: belief_rule` closes three quarters or more of the
  development gap between `guided` and the exact-posterior bound on both domains' paraphrased cases: the remaining
  headroom would not separate RL from instructions.
- **Hypothesis supported** if the trained policy beats the best control on frozen-set utility with an interval that
  excludes zero on both the paraphrased `refunds` cases and `access`, and on the independent part.
- **Hypothesis not supported** otherwise; the result is reported as such, with every control's score.

## Acceptable regressions

Against the best control on the frozen sets: honest requests approved down at most 5 points, undecided episodes not
up, escalation at most 25 % of cases, turns and tokens at most 1.5 times. A run past any limit does not count as
support for the hypothesis whatever its utility.
