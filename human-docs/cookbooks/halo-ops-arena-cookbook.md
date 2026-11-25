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

The second environment in the package: each refund claim is honest or abusive, the evidence is drawn from known
likelihoods, and the reward reads the hidden truth
([Ops Claims](../../agent-docs/training-methods/async-grpo/environments/tasks/ops-claims.md) ↗). Recipe:
[`gptoss-20b-ops-claims-lora-ep1.yaml`](../../examples/grpo/environmental/gptoss/vllm/gptoss-20b-ops-claims-lora-ep1.yaml).

```bash
halo run prepare-ops-claims-dataset --out data/ops_claims
halo run eval-ops-arena --environment ops_claims --guided --data data/ops_claims/eval.jsonl \
    --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model <id> \
    --customer-backend llm --customer-base-url https://openrouter.ai/api/v1 \
    --customer-model anthropic/claude-haiku-5.5 --out results/claims
```

Pilot: 80 deploy-cue cases (60 with training strategies, 20 held out), customer `anthropic/claude-haiku-5.5`.

| Agent | Mean objective | Best possible | No decision recorded | Abusive claims denied | Honest claims approved |
|---|---|---|---|---|---|
| gpt-oss-20b | 0.525 | 0.875 | 25 % | 39 % | 49 % |
| gpt-oss-20b, `--guided` | 0.642 | 0.875 | 20 % | 42 % | 67 % |

On the decisions it does record, guided gpt-oss-20b scores 0.80 against 0.88. Its remaining losses are a decision
told to the customer but never recorded, and abusive claims approved: the signal briefing moved fraud denial from
39 % to 42 % while it raised honest approvals by 18 points. `anthropic/claude-sonnet-5.5` with the briefing
completed 26 of the cases and scored 0.89 on them against 0.93 possible, recording a decision every time.

## Limits

- The scripted customer replies the same whatever the agent says; train with the `llm` backend.
- An agent that changes nothing succeeds on the five "change nothing" families, so read success on the action families.
- Families the base model solves give GRPO groups with no spread; `TRAIN_MIX` keeps them only to hold behaviour in place.
