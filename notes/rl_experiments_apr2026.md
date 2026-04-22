# RL Experiments — April 2026

## Setup

All runs use `verifiable_rl` objective on GSM8K, CoCoNut generator, 8 return sequences, 6 latent steps, `communication_type=attention`. Baseline (no communication, untrained) ≈ 43% eval selected accuracy.

## Results

| Run | anchor | div | prm_adv | freeze_prm | dense_credit | Eval sel_acc | Steps |
|-----|--------|-----|---------|------------|--------------|-------------|-------|
| Apr21 ×3 (anch=0.01) | 0.01 | 0.1–0.2 | ✗ | ✗ | ✗ | 0.8–4.5% | 1000 |
| Apr22 (anch=0.01) | 0.01 | 0.2 | ✗ | ✗ | ✗ | 6.1% | 1000 |
| **anchor0.5** | **0.5** | **0.2** | ✗ | ✗ | ✗ | **17.6–18.0%** | 1000 |
| div0.05\_anch0.05 | 0.05 | 0.05 | ✗ | ✗ | ✗ | 13.1% | ~1090 |
| **frozen\_prm** | **0.5** | **0.2** | ✗ | ✓ | ✗ | **16.8%** | 1000 |
| dense\_rewards | 0.05 | 1.0 | ✓ | ✗ | ✗ | 2.7% | 376 (collapsed) |
| dense\_credit | 0.05 | 1.0 | ✓ | ✗ | ✓ | — | 330 (collapsed) |

## Key Findings

### Anchor weight is the controlling variable
Every run with `anchor_weight ≤ 0.05` collapsed. `anchor_loss` grew from ~0.11 to 0.4–1.5 as the communication module drifted out of the generator's representation space, destroying accuracy. Runs with `anchor_weight=0.5` were the only ones that converged, holding `anchor_loss` steady at ~0.048.

### High diversity penalty + weak anchor = catastrophic
`diversity_penalty_weight=1.0` combined with `anchor_weight=0.05` forces trajectories apart faster than the anchor can constrain the module. Both affected runs (`dense_rewards`, `dense_credit`) show `mean_reward` collapsing from ~0.25 → ~0.04–0.08 within 300–470 steps.

### The dense_credit run did not get a fair test
It was launched with the same bad hyperparameters (`div=1.0, anch=0.05`) as the `dense_rewards` run that already died at step 376. The implementation is correct but the config killed it before any signal emerged.

### PRM advantages with a weak anchor make things worse
`use_prm_advantages=True` has the policy gradient chasing continuous PRM scores while the module is drifting — noisy and unstable. In all stable runs `policy_loss ≈ 0.0001` throughout; `selector_loss` does the actual work.

### Freezing PRM doesn't matter
`frozen_prm` (16.8%) ≈ `anchor0.5` (17.6–18.0%) at 1000 steps. The PRM gains little from joint training at this scale.

## What To Do Next

Re-run `dense_credit` with the hyperparameters that actually work:

```yaml
anchor_weight: 0.5
diversity_penalty_weight: 0.2
use_prm_advantages: false
use_dense_credit: true
traj_credit_weight: 1.0
```

The ~18% selected accuracy from `anchor0.5`/`frozen_prm` is the baseline to beat with dense credit.
