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

## Limits

- The scripted customer replies the same whatever the agent says; train with the `llm` backend.
- An agent that changes nothing succeeds on four of the ten eval families, so read success on the action families.
