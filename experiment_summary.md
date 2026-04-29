# RL Experiment Summary: Generator-Side Latent Communication

## Background: How the Coconut Model Works

The **coconut model** is a language model that reasons in a hidden "latent" space before writing its final answer. Concretely, instead of generating visible chain-of-thought tokens like "First, let me add 3 + 4...", it generates a sequence of **latent thought vectors** — dense, high-dimensional floating-point arrays that are never decoded into text. These vectors are computed one at a time, each one attending to all previous latent vectors, and together they form the model's internal reasoning trace. Only after all latent thoughts are generated does the model produce a visible answer.

This latent representation is tightly coupled to the model's weights — the model was trained end-to-end to use these specific vector representations in specific ways. Modifying them from the outside, or introducing anything unexpected into that sequence, causes the model to produce garbage because it has no experience processing anything other than its own outputs at each step.

---

## What We're Trying to Do

At inference time, we sample **8 independent trajectories** from the coconut model. A **trajectory** is one complete run: the model generates its full sequence of latent thoughts for a given problem and then produces a final answer. With 8 independent trajectories, we get 8 different answers (due to randomness in sampling), and we pick the best one using a **process reward model (PRM)**.

The **PRM** is a separate learned model that scores how good each trajectory is. Rather than only checking whether the final answer is correct, a process reward model evaluates *each latent step* of the reasoning trace, assigning a score to each one. To rank the 8 trajectories, we sum the per-step scores for each trajectory and pick the one with the highest total. This is called **best-of-N** (here, best-of-8).

**The baseline:** using best-of-8 with no other intervention achieves ~43% accuracy on the GSM math validation set.

**The hypothesis:** the 8 trajectories are currently fully independent — each one reasons completely on its own, with no awareness of what the others are doing. If the trajectories could *share information mid-generation* — specifically, if a trajectory could observe which reasoning directions the others are taking and steer toward more promising ones — then each individual trajectory might be better, and we'd beat 43%.

To test this, we attach a small **cross-path attention module** between the 8 trajectories. At each latent step, instead of each trajectory just passing its latent vector forward, it first runs cross-attention: each trajectory's latent vector attends to the latent vectors of all 8 trajectories at that step, and the result is a blended vector that incorporates information from the group. Think of it like each reasoning chain briefly looking at what the other 7 chains are thinking at the same point in the reasoning process, before continuing.

**`communication_every`** controls how often this happens. `communication_every=1` means it runs at every latent step; `communication_every=2` means it runs every other step (the odd steps pass through unchanged). More frequent communication means more opportunity to coordinate, but also more opportunities to corrupt the latent states.

We then train this module with reinforcement learning: if a trajectory leads to a correct final answer, we reinforce the latent modifications that produced it; if it leads to a wrong answer, we discourage them.

**The baseline we're trying to beat:** ~43% accuracy on GSM validation (best-of-8, no communication).

---

## Key Metrics

Two metrics matter here, and they're easy to confuse:

- **`eval_selected_accuracy`** (accuracy): Of the 8 trajectories, we pick one using the PRM. Was that chosen trajectory correct? This is the final metric — it's what matters for real-world use. All "Best Eval Acc" numbers in the table below are this.

- **Coverage**: Of the 8 trajectories, did *at least one* get the right answer? This is the ceiling on accuracy — if no trajectory is correct, selection doesn't matter.

The relationship: `accuracy ≤ coverage`. If coverage is 43%, we can't do better than 43% no matter how smart our selector is. If coverage collapses to 10%, accuracy will follow it down regardless of training.

---

## The Training Objective

The total loss has three components:

| Term | What it does | Weight |
|------|-------------|--------|
| **Policy loss** | RL signal — rewards correct trajectories, penalizes wrong ones | implicit |
| **Anchor loss** | Keeps the communication module from drifting too far from "do nothing" | `anchor_weight` |
| **Diversity loss** | Penalizes trajectories for being too similar to each other | `diversity_penalty_weight` |

### Policy Loss (REINFORCE)

The policy gradient (REINFORCE) algorithm works by treating the communication module's outputs as actions and the correctness of the final answer as the reward. For each trajectory in a batch, we compute a log-probability under the module's output distribution, multiply it by the advantage (how much better or worse than average this trajectory did), and take the gradient. Trajectories that scored better than average get reinforced; those that scored worse get suppressed.

**Advantages and the leave-one-out baseline:** a raw binary reward (1 = correct, 0 = wrong) is hard to learn from because it doesn't tell you *how much better* you did. To help, we subtract a baseline from each trajectory's reward before using it as the gradient signal. The **leave-one-out baseline** for trajectory `i` is the mean reward of the other 7 trajectories in the same group. So if 6 out of 8 trajectories got it right, the 2 that were wrong get a large negative advantage and are suppressed — even though they aren't "zero reward" in absolute terms.

**Noise-based trajectory diversity:** rather than sampling from a discrete distribution, we generate diverse trajectories by adding Gaussian noise to the latent vectors (`noise_std=0.1`). The policy gradient then asks: which noise realizations led to correct answers? The log-probability of a noise sample under its Gaussian distribution is differentiable and provides the gradient signal. This lets us apply REINFORCE to a continuous latent space.

**Dense credit (`use_dense_credit`):** standard REINFORCE gives every latent step in a trajectory the same credit — if the trajectory was good overall, every step is equally reinforced. Dense credit tries to attribute reward more precisely. It gives more credit to latent steps where the attention module made sharper, more selective routing decisions (i.e., steps where it clearly preferred some trajectories over others), and down-weights steps where attention was diffuse. The idea is that sharp attention = the module was doing something meaningful at that step, so it deserves more of the credit/blame. In practice (run 10), this did not improve results.

### Anchor Loss

The anchor loss is a mean-squared-error penalty between the communication module's output (the modified latent vector) and its input (the original latent vector). It acts like a rubber band pulling the module's outputs back toward "don't change anything." Without this constraint, a randomly initialized module will make large, arbitrary modifications to the latent vectors before it has learned to make *useful* ones — corrupting the generator's reasoning immediately.

A large `anchor_weight` means the rubber band is tight: the module can only make small changes. This preserves generation quality but limits how much the module can actually coordinate the trajectories.

### Diversity Loss

This penalizes trajectories for having similar post-communication latent vectors. The motivation: if all 8 trajectories converge to the same reasoning path, best-of-8 gives no benefit over best-of-1. We want the trajectories to explore different directions so the group collectively covers more of the answer space.

The problem: on hard problems, *most* directions are wrong. Forcing diversity means some trajectories are pushed toward bad directions on purpose, which hurts coverage. This is why high diversity penalty (runs 7, 8) is catastrophic.

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

With `anchor_weight=0.01`, the module is essentially unconstrained. Early in training, before the module has learned anything useful, it makes large random modifications to the latent vectors. This is immediately catastrophic: the coconut model has never been trained to receive externally modified latent vectors, so even small perturbations compound across the latent steps into completely broken reasoning. **Coverage collapses** from ~43% to ~5–10%, meaning nearly every problem now has zero correct trajectories among the 8.

With collapsed coverage, almost every trajectory in the training batch gets a reward of zero. The REINFORCE gradient signal requires some correct and some incorrect trajectories in each group to compute a meaningful advantage — if everything is wrong, the advantages are all zero and nothing gets learned. The model is stuck in a hole it can't escape.

Changing communication frequency from every-1 to every-2 steps doesn't help because the bottleneck is the weak anchor, not how often communication runs.

> **Key insight:** the coconut latent space is highly sensitive to modification. A randomly initialized communication module acts like noise injection, not coordination.

### Phase 2: High Anchor (runs 5, 9)

Raising `anchor_weight` to 0.5 — a 50x increase — forces the module to stay close to "do nothing." We can verify this is working: the anchor loss settles to ~0.04, meaning the module's outputs are very close to its inputs. With this constraint active, generation quality is mostly preserved, coverage holds at ~21% on eval, and the model does learn: training accuracy rises from ~29% → ~34% over 1,000 steps.

**Best eval accuracy: 18%**, the highest we've seen — but still less than half the baseline.

Freezing the PRM during training (run 9) gives nearly identical results (16.8%). This matters because it rules out an alternative hypothesis: maybe the PRM was drifting during training and giving bad scores, causing the poor performance. If freezing the PRM doesn't help, the problem is elsewhere.

> **Key insight:** the anchor is necessary but not sufficient. Even with `anchor_weight=0.5`, the trained module still cuts eval coverage from 43% to ~21%. The communication is still hurting generation, just less catastrophically than before.

### Phase 3: Longer Training (runs 6, 7, 8)

With medium anchor (0.05), extending to 10,000 steps gives 13.3% — *worse* than the 1,000-step high-anchor run. More gradient steps don't help if the initial collapse isn't recovered. The model trains on bad data (collapsed coverage) and learns to operate in that bad regime.

Very high diversity penalty (weight=1.0) catastrophically collapses both runs (runs 7, 8) to ~2.5%. The mechanism: forcing the 8 trajectories to be maximally different from each other means pushing them in different directions, many of which are wrong. Since coverage was already fragile, this additional pressure drops it to near zero, and the RL signal dies again.

> **Key insight:** diversity and accuracy are in direct conflict when coverage is already low. More exploration is only useful if there's something to explore toward — but when coverage is 5%, almost every direction is a dead end.

### Phase 4: Dense Credit + High Anchor (run 10 — most recent)

This run used the best anchor setting (0.5) combined with dense credit assignment and 10,000 training steps. Dense credit should, in theory, give more informative gradients by attributing reward to specific latent steps rather than spreading it equally across all steps in a trajectory. The run shows a gentle upward training curve (25% → 30%) but eval accuracy peaked at **13.3%** — lower than the simpler 1,000-step high-anchor run.

Dense credit didn't help because the bottleneck is not gradient informativeness. The module is learning, but what it's learning to do still disrupts the generator's latent representations enough that coverage stays depressed. Better credit assignment can't fix the fundamental problem of out-of-distribution interference.

---

## The Core Problem

Every experiment shows the same pattern: **the trained communication module hurts the generator more than it helps it coordinate.**

```
Baseline (no comm):  43% accuracy,  ~43% coverage
Best trained run:    18% accuracy,  ~21% coverage
```

The coverage number tells the story. Before any communication, the 8 independent trajectories collectively solve ~43% of the problems (at least one trajectory gets it right). After training the communication module, that drops to ~21%. We're losing half our coverage to interference before selection even runs.

Why does this happen? The coconut model was trained with a specific, highly tuned computation: each latent step attends to all previous latent steps using a fixed distribution of attention weights, and the resulting vectors feed into the next step. This is an extremely tight loop. Injecting any external modification — even a small one — creates a **distribution shift**: the modified latent vector doesn't look like what the model learned to expect as input to the next step. This error doesn't stay small — each subsequent step attends to the modified vector, its output shifts slightly, and that shift feeds into the next step, and so on. Over 6 latent steps, small perturbations compound into large deviations from the model's trained operating regime, and reasoning breaks down.

The anchor loss fights this by limiting how much the module can modify each vector, but it can't eliminate the problem entirely — any modification is some shift, and the model has zero robustness to it.

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
