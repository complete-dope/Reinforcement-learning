# Reinforcement Learning for LLMs

Minimal, single-file implementations of the RL algorithms used to fine-tune language models. Each script trains a small model on a toy task so the full loss can be read in one screen.

## Setup

```bash
pip install torch transformers
```

Most scripts load `HuggingFaceTB/SmolLM2-135M-Instruct` through `llm/base.py` on Apple Silicon (`device_map="mps"`). Change the device there for CUDA or CPU.

Run any script from inside `llm/`:

```bash
cd llm
python grpo.py
```

## Files

| File | Algorithm | Model | Task |
|---|---|---|---|
| `base.py` | Shared model and tokenizer loader | SmolLM2-135M-Instruct | Run directly for a chat sanity check |
| `policy_grad.py` | REINFORCE | SmolLM2-135M-Instruct | Reward 1 if the output contains "India" |
| `ppo.py` | PPO-style clipped objective | SmolLM2-135M-Instruct | Reward = count of "Paris" in the output |
| `grpo.py` | GRPO Done Right (Dr. GRPO) | SmolLM2-135M-Instruct | Same as PPO |
| `dpo.py` | DPO | SmolLM2-135M-Instruct | Prefer `" Paris."` over `" Paris"` |
| `toy_example.py` | GRPO, self-contained | GPT-2 | Reward 1 if the output contains "Paris" |
| `ppo.ipynb`, `dpo.ipynb` | Experiment logs | | Training runs and notes |
| `kimi.py` | Placeholder | | Empty |

## How the algorithms build on each other

**1. REINFORCE** (`policy_grad.py`). Sample an answer, score it, and push up the log-prob of rewarded tokens.

```
loss = -R * sum(log π(token | prompt))
```

**2. Baseline and advantage.** Subtract the group mean reward so only above-average answers get reinforced.

```
A = R - mean(R)
```

**3. Importance sampling** (`ppo.py`). Generation is expensive, so one batch of samples is reused for `K` gradient steps. The ratio corrects for the policy drifting from the one that generated the samples.

```
ratio = exp(log π_new - log π_old)
```

The gradient of `-A * ratio` equals `-A * ratio * ∇log π_new`, which reduces to the REINFORCE gradient when `ratio = 1`. The derivation is written out in the header of `ppo.py`.

**4. Clipping.** Stop the update once the ratio leaves `[1-ε, 1+ε]`.

```
policy_loss = -min(A * ratio, A * clip(ratio, 1-ε, 1+ε))
```

**5. KL penalty.** Keep the policy close to a frozen reference model using the k3 estimator.

```
kl   = exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1
loss = policy_loss + β * kl
```

**6. Length normalisation.** This is the only code difference between `ppo.py` and `grpo.py`.

| Script | Normaliser |
|---|---|
| `ppo.py` | Number of real tokens, `mask.sum()` |
| `grpo.py` | Fixed constant, `G * MAX_NEW` |

Dividing by a constant removes the bias that per-sequence length normalisation introduces, which is the Dr. GRPO fix.

**7. DPO** (`dpo.py`). No sampling and no reward function. Given a preferred and a rejected answer, widen the gap between their log-prob gains over the reference model.

```
margin = β * [(new_good - ref_good) - (new_bad - ref_bad)]
loss   = -log σ(margin)
```

## Notes

- `ppo.py` has no value network. It uses the group mean as the baseline, so it is PPO's clipped objective rather than full actor-critic PPO.
- `ppo.ipynb` records a bug where the KL was written as `new_lp - ref_lp`. Its gradient only ever lowers `new_lp`, so it never pulls the policy back to the reference. The k3 estimator above fixes this.
- Default hyperparameters for PPO and GRPO are `G=8`, `MAX_NEW=8`, `ε=0.2`, `β=0.05`, `lr=1e-6`, `K=2`.
- The `gym/` folder is not covered here.
