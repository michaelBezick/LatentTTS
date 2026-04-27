# RL Experiment Summary: Generator-Side Latent Communication

## What We're Trying to Do

The coconut model generates latent "thoughts" before producing a final answer. At test time we sample **8 independent trajectories** and pick the best one using a process reward model (PRM). This is called best-of-N.

The hypothesis: if the 8 trajectories could *talk to each other* during generation — sharing information about which reasoning paths are working — we could get better answers than 8 fully independent chains.

To test this, we train a small **cross-path attention module** that lets each trajectory attend to the latent states of the other 7 trajectories at each latent step. The module is trained with RL: trajectories that lead to correct answers get reinforced.

**The baseline we're trying to beat:** ~43% accuracy on GSM validation (best-of-8, no communication).

---

## The Training Objective

The total loss has three components:

| Term | What it does | Weight |
|------|-------------|--------|
| **Policy loss** | RL signal — rewards correct trajectories, penalizes wrong ones | implicit |
| **Anchor loss** | Keeps the communication module from drifting too far from "do nothing" | `anchor_weight` |
| **Diversity loss** | Penalizes trajectories for being too similar to each other | `diversity_penalty_weight` |

The anchor is crucial: without it, the module freely modifies latent states and sends the model out-of-distribution, collapsing generation quality.

---

## Experiment Results at a Glance

**Baseline (no communication module):** ~43% `eval_selected_accuracy`

| Run | anchor | comm_every | diversity | steps | Best Eval Acc | Train Trend | Verdict |
|-----|--------|-----------|-----------|-------|---------------|-------------|---------|
| [1] Low anchor, fast comm (attempt 1) | 0.01 | every 1 | 0.1 | 1,000 | **0.8%** | collapsed | ❌ |
| [2] Low anchor, fast comm (attempt 2) | 0.01 | every 1 | 0.1 | 1,000 | **4.7%** | collapsed | ❌ |
| [3] Low anchor, slower comm | 0.01 | every 2 | 0.2 | 1,000 | **2.1%** | collapsed | ❌ |
| [4] Low anchor, slower comm (repro) | 0.01 | every 2 | 0.2 | 1,000 | **6.8%** | collapsed | ❌ |
| [5] **High anchor** | **0.5** | every 2 | 0.2 | 1,000 | **18.0%** | improving | ⚠️ best so far |
| [6] Medium anchor, longer run | 0.05 | every 2 | 0.05 | 10,000 | **13.3%** | roughly flat | ❌ |
| [7] Very high diversity penalty | 0.05 | every 2 | 1.0 | 10,000 | **2.7%** | collapsed | ❌ |
| [8] Very high diversity (repro) | 0.05 | every 2 | 1.0 | 10,000 | **2.5%** | collapsed | ❌ |
| [9] High anchor, frozen PRM | 0.5 | every 2 | 0.2 | 1,000 | **16.8%** | improving | ⚠️ close to [5] |
| [10] Dense reward + high anchor | 0.5 | every 2 | 0.2 | 10,000 | **13.3%** | slow improve | ❌ no gain |

> All runs use: coconut generator, attention communication, noise-based diversity (`noise_std=0.1`), 8 trajectories, leave-one-out reward baseline, cosine LR schedule.

---

## What We Learned From Each Phase

### Phase 1: Low Anchor (runs 1–4)

With `anchor_weight=0.01`, the module is essentially unconstrained. Early in training it makes large, random modifications to the latent states — before it has learned anything useful. This immediately corrupts the generator's reasoning, causing **coverage collapse**: the fraction of problems where any trajectory finds the correct answer drops from ~43% to ~5–10%.

With collapsed coverage, the RL signal is nearly all zeros (nothing to learn from), and the model can't recover. **Runs faster or slower comm both collapse** — the root cause is the weak anchor, not the frequency.

> **Key insight:** the coconut latent space is highly sensitive to modification. A randomly initialized communication module acts like noise injection, not coordination.

### Phase 2: High Anchor (runs 5, 9)

Raising `anchor_weight` to 0.5 — a 50x increase — forces the module to stay close to "do nothing." The anchor loss drops to ~0.04 (near zero), confirming the constraint is active. With generation quality mostly preserved, coverage holds at ~21% on eval and the model does learn: training accuracy rises from ~29% → ~34% over 1,000 steps.

**Best eval accuracy: 18%**, which is the highest we've seen — but still less than half the baseline.

Freezing the PRM during training (run 9) gives nearly identical results (16.8%), suggesting the PRM fine-tuning isn't driving performance.

> **Key insight:** the anchor is necessary but not sufficient. Even with anchor_weight=0.5, the trained module still cuts eval coverage from 43% to ~21%, meaning the communication is still hurting generation.

### Phase 3: Longer Training (runs 6, 7, 8)

Extending to 10,000 steps with medium anchor (0.05) gives 13.3% — *worse* than the 1,000-step high-anchor run. Longer training doesn't compensate for the coverage collapse from a weak anchor.

Very high diversity penalty (1.0 weight) catastrophically collapses both runs (runs 7, 8) to ~2.5%. Forcing trajectory diversity fights directly against the reward signal — trajectories can't all be different *and* all be good simultaneously on hard problems.

> **Key insight:** diversity and accuracy are in direct conflict when coverage is already low. Maximizing diversity makes the model explore more, but there's nothing to explore toward.

### Phase 4: Dense Credit + High Anchor (run 10 — most recent)

Combining the best anchor setting (0.5) with a denser reward signal and 10,000 training steps. The run completed ~5,000 steps and shows a gentle upward training curve (25% → 30%), but eval accuracy peaked at **13.3%** — lower than the simpler 1,000-step high-anchor run.

The denser reward did not help. The bottleneck isn't reward sparsity; it's that the communication module damages generation quality (low coverage) before it can learn anything useful. More signal about *bad* behavior doesn't fix this.

---

## The Core Problem

Every experiment shows the same pattern: **the trained communication module hurts the generator more than it helps it coordinate.**

```
Baseline (no comm):  43% accuracy,  ~43% coverage
Best trained run:    18% accuracy,  ~21% coverage
```

Coverage is the key diagnostic. It measures "what fraction of problems does at least one of the 8 trajectories solve?" The communication module cuts this nearly in half, even in the best case. You can't select a good answer if none of the 8 trajectories are correct.

The likely cause: the coconut model's latent reasoning process is a tightly tuned sequential computation. Injecting cross-path attention mid-generation — even with a strong anchor — creates small distribution shifts that compound across latent steps. The model was never trained with this interference, so it has no robustness to it.

---

## Summary of Hyperparameter Effects

| Hyperparameter | Effect | Recommendation |
|----------------|--------|----------------|
| `anchor_weight` | **Most important.** Low → collapse. High (0.5) → best results. | Keep at ≥ 0.5 |
| `communication_every` | Slower (every 2) is better than every step | Keep at 2 |
| `diversity_penalty_weight` | Any value above ~0.2 collapses training | Keep ≤ 0.2 |
| Training steps | More steps with weak anchor doesn't help | Fix anchor first |
| Dense reward | No measurable benefit | Not worth pursuing |
| Frozen PRM | Negligible effect | Doesn't matter |

---

## Where Things Stand

The communication-during-generation approach is struggling against a fundamental constraint: the model's latent space is too fragile to tolerate mid-generation intervention. Increasing the anchor helps but doesn't solve the problem — it just makes the module more conservative, which limits what it can learn.

**The best result (18%)** is still well below the no-communication baseline (43%), meaning the current approach is actively harmful relative to doing nothing.

### Potential Next Steps

1. **Post-hoc aggregation:** let all 8 trajectories finish independently, then learn a composition or selection module that operates on completed latent sequences. No mid-generation interference.

2. **Latent tree search:** use the PRM to do structured search in latent space at each step (beam search over latent tokens) rather than training a communication module.

3. **Sequential self-refinement:** train the generator to condition on a failed trajectory and produce a better one, keeping the intervention at the input level where the model is designed to accept conditioning.

4. **Curriculum warmup:** start with an effectively identity-initialized module and very slowly reduce the anchor weight over training, giving the generator time to adapt.
