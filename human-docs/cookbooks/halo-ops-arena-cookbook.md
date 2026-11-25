# Ops Arena Cookbook

Train gpt-oss-20b as a support agent that resolves tickets among other agents (a customer, a supervisor, teammates
and automations, some of them injected) without breaking policy. Success and safety are graded from what the agent
did to the backend; White Circle policies grade how it talked. Agents, families and the reward:
[Ops Arena](../../agent-docs/training-methods/async-grpo/environments/tasks/ops-arena.md) ↗.

## Support

- **Method:** [Async GRPO with Environments](../training-methods/async-grpo-environments.md), `environment_type: ops_arena`.
- **Recipe:** [`gptoss-20b-ops-arena-lora-ep1.yaml`](../../examples/grpo/environmental/gptoss/vllm/gptoss-20b-ops-arena-lora-ep1.yaml):
  LoRA r=64, four trainer GPUs and two vLLM servers, the layout of the gpt-oss code-contests recipe.
- **Customer simulator:** `scripted` for smoke runs; an OpenAI-compatible endpoint (OpenRouter, Nebius Token Factory,
  a local vLLM) for training.
- **Policies:** [`policies.json`](../../examples/grpo/environmental/ops-arena/policies.json), two sets: `conduct`
  (false claims about actions, manipulation, exposing internal instructions) and `helpfulness` (unwarranted refusal,
  tone).

## 1. Prepare the data

```bash
halo run prepare-ops-arena-dataset --out data/ops_arena
```

## 2. Baseline before any GPU

```bash
halo run eval-ops-arena --data data/ops_arena/eval.jsonl \
    --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model <gpt-oss-20b id> \
    --customer-backend llm --customer-base-url https://openrouter.ai/api/v1 --customer-model <customer model id> \
    --out results/baseline
```

## 3. Smoke the recipe

Serve the model as the recipe's header says, then run a few monitor rows through the training episode loop:

```bash
halo run eval-env --training_config examples/grpo/environmental/gptoss/vllm/gptoss-20b-ops-arena-lora-ep1.yaml \
    --dataset ./data/ops_arena/hf --split test --context_fields scenario --num_examples 20 \
    --base_url http://localhost:8000/v1 --model <served id>
```

## 4. Train

Add the White Circle terms first (the recipe's `rewards:` block shows where):

```bash
export WHITECIRCLE_API_KEY=wc-...
halo run create-whitecircle-policies --environment-id <id> \
    --policies examples/grpo/environmental/ops-arena/policies.json
CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_NCCL_TIMEOUT_MINUTES=60 halo launch grpo/environmental \
    examples/grpo/environmental/gptoss/vllm/gptoss-20b-ops-arena-lora-ep1.yaml -n 4
```

## 5. Evaluate the checkpoint

Serve it and rerun step 2 against `http://localhost:8000/v1`. Compare success on the action families, each hard
violation, the held-out families and the eval-awareness gap with the baseline.

## Results

Baseline: `openai/gpt-oss-20b` through OpenRouter at temperature 1.0, the customer played by
`anthropic/claude-haiku-5.5`, all 850 eval episodes (25 seeds per family under both cues).

| | Episodes | Success | Any hard violation |
|---|---|---|---|
| All | 850 | 87.5 % | 8.4 % |
| Training families | 700 | 91.0 % | 5.9 % |
| Held-out families | 150 | 71.3 % | 20.0 % |
| `preapproval_history` | 50 | 34 % | 58 % |
| `preapproval_note` (held out) | 50 | 46 % | 50 % |
| `pushy_compensation` (held out) | 50 | 76 % | 10 % |
| `preapproval_inbox` | 50 | 96 % | 0 % |

The base model ignores every injection that targets another customer's order and refuses impostors, but a
pre-approval written into the ticket history or the order's record note talks it into refunding past the direct limit
without the supervisor; the same text in the team inbox does not. Every other family scores 90 to 100 %, which is why
`TRAIN_MIX` weights `preapproval_history` highest. Under the eval and deploy cues the violation rates are 7.8 % and
8.9 % (41 discordant pairs of 425), so the cue alone shows no eval-awareness gap at baseline.

## Ops Claims

The second environment in the package: each case is honest or abusive, the evidence is drawn from known likelihoods,
the reward reads the hidden truth, and every decision carries the agent's probability that the case is abusive
([Ops Claims](../../agent-docs/training-methods/async-grpo/environments/tasks/ops-claims.md) ↗). Two domains share
the structure: `refunds` (retail claims) and `access` (helpdesk resets that may be account takeovers). Recipe:
[`gptoss-20b-ops-claims-lora-ep1.yaml`](../../examples/grpo/environmental/gptoss/vllm/gptoss-20b-ops-claims-lora-ep1.yaml),
trained on `refunds`, evaluated on both. The hypothesis, controls, frozen test sets and decision rules are fixed in
the [Ops Claims protocol](halo-ops-claims-protocol.md).

```bash
halo run prepare-ops-claims-dataset --domain refunds --out data/ops_claims/refunds
halo run prepare-ops-claims-dataset --domain access --out data/ops_claims/access
halo run eval-ops-arena --environment ops_claims --domain access --prompt guided \
    --data data/ops_claims/access/eval.jsonl --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY --model <id> --customer-backend llm \
    --customer-base-url https://openrouter.ai/api/v1 --customer-model anthropic/claude-haiku-5.5 --out results/access
```

To check the recipe's serving path without a GPU host of your own, `modal run
scripts/environments/inference/smoke_ops_claims_modal.py --num-examples 20` serves the model on a Modal H100 with the
vLLM image and the recipe's parsers and runs `eval_env.py` under the recipe against it; `--customer llm` adds the
OpenRouter requester.

Baseline: `openai/gpt-oss-20b` at temperature 1.0, requester `anthropic/claude-haiku-5.5`, 200 deploy-cue cases per
domain (100 with training styles, 100 held out).

| Domain, prompt | Objective | Best possible | Abusive denied | Honest approved | Undecided | Belief vs posterior (r) |
|---|---|---|---|---|---|---|
| refunds | 0.587 | 0.812 | 30 % | 70 % | 12 % | 0.60 |
| refunds, guided | 0.621 | 0.812 | 44 % | 67 % | 12 % | 0.70 |
| access | 0.487 | 0.746 | 9 % | 60 % | 23 % | 0.53 |
| access, guided | 0.583 | 0.746 | 5 % | 81 % | 14 % | 0.72 |

In `access` the model's stated probability tracks the posterior, but it does not act on it: on guided cases where it
states 0.6 or more it escalated 26 times and denied 6, and on cases whose posterior is 0.8 or more it still granted 9
resets of 33. In `refunds` it denies when it believes abuse.

Persuasion replays (50 guided cases per domain, each under six styles plus a second calm run): a second calm run
changes the decision on 42 % (`refunds`) and 44 % (`access`) of cases, 29 % to 42 % even where the posterior is below
0.1 or above 0.9. Style replays flip 42 % to 62 %, so at baseline any persuasion effect is small beside the
model's run-to-run inconsistency, which is itself the larger measurable weakness.

### Controls

The [protocol](halo-ops-claims-protocol.md)'s cheap controls on the development cases: `openai/gpt-oss-20b`,
requester `anthropic/claude-haiku-5.5`, 200 deploy-cue cases per domain (standard wording) and 50 paraphrased cases
(`stress.jsonl`, `wording: alt`).

| Domain, run | Utility | Bound | Harmful | Fraud denied | Honest approved | Undecided | Follows own belief |
|---|---|---|---|---|---|---|---|
| access, guided | 0.528 | 0.746 | 13.5 % | 3.9 % | 78.9 % | 13.0 % | 0.72 |
| access, explicit | 0.558 | 0.746 | 18.5 % | 32.5 % | 77.2 % | 13.5 % | 0.95 |
| access, guided, belief_rule | 0.571 | 0.746 | 16.5 % | 36.4 % | 80.5 % | 17.5 % | 1.00 |
| access, guided, silent requester | 0.566 | 0.746 | 15.0 % | 10.4 % | 76.4 % | 2.5 % | 0.70 |
| access, guided, paraphrased | 0.475 | 0.706 | 22.0 % | 5.6 % | 81.2 % | 14.0 % | 0.88 |
| access, explicit, paraphrased | 0.541 | 0.706 | 16.0 % | 22.2 % | 81.2 % | 20.0 % | 0.98 |
| refunds, guided | 0.601 | 0.812 | 14.0 % | 50.5 % | 68.8 % | 12.5 % | 0.75 |
| refunds, explicit | 0.548 | 0.812 | 15.5 % | 48.4 % | 65.1 % | 18.0 % | 0.93 |
| refunds, guided, belief_rule | 0.637 | 0.812 | 19.0 % | 58.2 % | 82.6 % | 8.0 % | 1.00 |
| refunds, guided, silent requester | 0.621 | 0.812 | 18.0 % | 69.2 % | 52.3 % | 2.0 % | 0.71 |
| refunds, guided, paraphrased | 0.514 | 0.780 | 20.0 % | 38.9 % | 59.4 % | 12.0 % | 0.80 |
| refunds, explicit, paraphrased | 0.471 | 0.780 | 26.0 % | 55.6 % | 53.1 % | 16.0 % | 1.00 |

No control's paired utility change against `guided` has an interval that excludes zero (200 cases). The explicit
prompt and the belief rule make actions follow the stated probability (0.93 to 1.00) and fix `access`'s missing
denials, but move utility by at most 0.04 to 0.07: on paraphrased cases the explicit prompt closes 29 % of the gap to
the bound on `access` and none on `refunds`, short of the protocol's stop rule. The silent requester cuts undecided
episodes by 10 points in both domains (interval excludes zero): the dialogue causes most unrecorded decisions and adds
no utility.

The stated probability carries the remaining gap. Brier scores, with the exact posterior and the prior alone as
references:

| Domain, wording | Agent | Exact posterior | Prior only |
|---|---|---|---|
| access, standard | 0.176 | 0.135 | 0.230 |
| access, paraphrased | 0.236 | 0.130 | 0.225 |
| refunds, standard | 0.155 | 0.081 | 0.249 |
| refunds, paraphrased | 0.238 | 0.167 | 0.233 |

On standard wording the model's probabilities capture 56 to 57 % of the possible improvement over the prior; on
paraphrased records they are no better than the prior. The paraphrase table has not been checked against a stronger
model, so part of that drop may be the wording rather than the model.

## Limits

- The scripted customer replies the same whatever the agent says; train with the `llm` backend.
- An agent that changes nothing succeeds on the five "change nothing" families, so read success on the action families.
- Families the base model solves give GRPO groups with no spread; `TRAIN_MIX` keeps them only to hold behaviour in place.
