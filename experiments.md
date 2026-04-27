# Experiment Results

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260421_101140-lrib5np2
- **Started:** 2026-04-21T14:11:40.061388Z
- **Git commit:** e35aee11f239ee3a2c3f5e047cf1fd4edd4a06b0
- **Host:** gpu-b10-1.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `1` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `1000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.1` |
| `anchor_weight` | `0.01` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

_No final metrics recorded._

### Training Curve Summary

_No training history in output.log._

### Eval Curve Summary

_No eval history._

---

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260421_102304-shkt5h6x
- **Started:** 2026-04-21T14:23:04.178178Z
- **Git commit:** 05afd02121b861fac61983e7a3dd660e802e4a9a
- **Host:** gpu-b10-1.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `1` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `1000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.1` |
| `anchor_weight` | `0.01` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.695162` |
| `eval_coverage` | `0.044922` |
| `eval_diversity_loss` | `0.929138` |
| `eval_loss` | `0.099854` |
| `eval_mean_reward` | `0.009766` |
| `eval_policy_loss` | `1.2e-05` |
| `eval_selected_accuracy` | `0.007812` |
| `eval_voting_accuracy` | `0.007812` |
| `lr` | `9.4e-05` |
| `train_anchor_loss` | `3.314373` |
| `train_coverage` | `0.1875` |
| `train_diversity_loss` | `3.349219` |
| `train_loss` | `0.368001` |
| `train_mean_reward` | `0.034375` |
| `train_policy_loss` | `-3.7e-05` |
| `train_selected_accuracy` | `0.025` |
| `train_voting_accuracy` | `0.025` |

### Training Curve Summary

- Steps recorded: 100 (step 10 → 1000)
- `train_mean_reward`: 1.189062 → 0.034375
- `train_selected_accuracy`: 1.225 → 0.025
- `train_loss`: 0.408857 → 0.368001

### Eval Curve Summary

- Eval snapshots: 2
- Best `eval_selected_accuracy`: 0.011719

---

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260421_113312-am6or4po
- **Started:** 2026-04-21T15:33:12.924507Z
- **Git commit:** a62c573a601d02b1b73c84ccd02b3fa93b5447a9
- **Host:** gpu-b10-2.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `1` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `1000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.1` |
| `anchor_weight` | `0.01` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.295048` |
| `eval_coverage` | `0.103516` |
| `eval_diversity_loss` | `0.996887` |
| `eval_loss` | `0.058012` |
| `eval_mean_reward` | `0.04126` |
| `eval_policy_loss` | `1.8e-05` |
| `eval_selected_accuracy` | `0.044922` |
| `eval_selector_loss` | `-0.044673` |
| `eval_voting_accuracy` | `0.044922` |
| `lr` | `9.4e-05` |
| `train_anchor_loss` | `1.140676` |
| `train_coverage` | `0.875` |
| `train_diversity_loss` | `3.990918` |
| `train_loss` | `-0.089641` |
| `train_mean_reward` | `0.451562` |
| `train_policy_loss` | `0.000124` |
| `train_selected_accuracy` | `0.5125` |
| `train_selector_loss` | `-0.500452` |
| `train_voting_accuracy` | `0.4625` |

### Training Curve Summary

- Steps recorded: 100 (step 10 → 1000)
- `train_mean_reward`: 1.185938 → 0.451562
- `train_selected_accuracy`: 1.1875 → 0.5125
- `train_loss`: -0.779102 → -0.089641

### Eval Curve Summary

- Eval snapshots: 2
- Best `eval_selected_accuracy`: 0.046875

---

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260421_142608-y4b9fwez
- **Started:** 2026-04-21T18:26:08.907649Z
- **Git commit:** b2e0326aae2de7f446057318f90f5fb6b186e765
- **Host:** gpu-b9-6.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `2` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `1000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.2` |
| `anchor_weight` | `0.01` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.550498` |
| `eval_coverage` | `0.082031` |
| `eval_diversity_loss` | `0.683523` |
| `eval_loss` | `0.114263` |
| `eval_mean_reward` | `0.020508` |
| `eval_policy_loss` | `-4e-06` |
| `eval_selected_accuracy` | `0.03125` |
| `eval_selector_loss` | `-0.02792` |
| `eval_voting_accuracy` | `0.027344` |
| `lr` | `9.4e-05` |
| `train_anchor_loss` | `0.819455` |
| `train_coverage` | `0.11875` |
| `train_diversity_loss` | `0.432706` |
| `train_loss` | `0.070514` |
| `train_mean_reward` | `0.021875` |
| `train_policy_loss` | `1.6e-05` |
| `train_selected_accuracy` | `0.025` |
| `train_selector_loss` | `-0.024241` |
| `train_voting_accuracy` | `0.00625` |

### Training Curve Summary

- Steps recorded: 100 (step 10 → 1000)
- `train_mean_reward`: 0.289844 → 0.021875
- `train_selected_accuracy`: 0.2875 → 0.025
- `train_loss`: -0.088331 → 0.070514

### Eval Curve Summary

- Eval snapshots: 2
- Best `eval_selected_accuracy`: 0.021484

---

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260422_113737-x9rat5mg
- **Started:** 2026-04-22T15:37:37.739521Z
- **Git commit:** 17a56f42e6bb2f0bfa096a3441a1e085968f8ed7
- **Host:** gpu-b9-3.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `2` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `1000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.2` |
| `anchor_weight` | `0.01` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.298029` |
| `eval_coverage` | `0.095703` |
| `eval_diversity_loss` | `0.998459` |
| `eval_loss` | `0.210336` |
| `eval_mean_reward` | `0.060303` |
| `eval_policy_loss` | `0.067197` |
| `eval_selected_accuracy` | `0.060547` |
| `eval_selector_loss` | `-0.059652` |
| `eval_voting_accuracy` | `0.064453` |
| `lr` | `9.4e-05` |
| `train_anchor_loss` | `0.288911` |
| `train_coverage` | `0.284375` |
| `train_diversity_loss` | `0.999463` |
| `train_loss` | `0.095868` |
| `train_mean_reward` | `0.210156` |
| `train_policy_loss` | `0.117102` |
| `train_selected_accuracy` | `0.225` |
| `train_selector_loss` | `-0.224184` |
| `train_voting_accuracy` | `0.21875` |

### Training Curve Summary

- Steps recorded: 100 (step 10 → 1000)
- `train_mean_reward`: 0.287891 → 0.210156
- `train_selected_accuracy`: 0.28125 → 0.225
- `train_loss`: 0.013089 → 0.095868

### Eval Curve Summary

- Eval snapshots: 2
- Best `eval_selected_accuracy`: 0.068359

---

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260422_141338-96092rjh
- **Started:** 2026-04-22T18:13:38.438681Z
- **Git commit:** c288c25129273da4670d51a952fff0721532ae77
- **Host:** gpu-b9-6.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `2` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `1000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.2` |
| `anchor_weight` | `0.5` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.049357` |
| `eval_coverage` | `0.212891` |
| `eval_diversity_loss` | `0.999115` |
| `eval_loss` | `0.053956` |
| `eval_mean_reward` | `0.169434` |
| `eval_policy_loss` | `-8.7e-05` |
| `eval_selected_accuracy` | `0.175781` |
| `eval_selector_loss` | `-0.170609` |
| `eval_voting_accuracy` | `0.166016` |
| `lr` | `9.4e-05` |
| `train_anchor_loss` | `0.041616` |
| `train_coverage` | `0.4` |
| `train_diversity_loss` | `0.999536` |
| `train_loss` | `-0.121208` |
| `train_mean_reward` | `0.342578` |
| `train_policy_loss` | `-4e-06` |
| `train_selected_accuracy` | `0.340625` |
| `train_selector_loss` | `-0.342092` |
| `train_voting_accuracy` | `0.346875` |

### Training Curve Summary

- Steps recorded: 100 (step 10 → 1000)
- `train_mean_reward`: 0.289844 → 0.342578
- `train_selected_accuracy`: 0.290625 → 0.340625
- `train_loss`: -0.04245 → -0.121208

### Eval Curve Summary

- Eval snapshots: 2
- Best `eval_selected_accuracy`: 0.179688

---

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260422_160120-5vygi5jf
- **Started:** 2026-04-22T20:01:20.258866Z
- **Git commit:** 7ceb499fc26e6b0668518439c07fdc93da2b7742
- **Host:** gpu-b9-7.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `2` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `16` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `10000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.05` |
| `anchor_weight` | `0.05` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.070489` |
| `eval_coverage` | `0.167969` |
| `eval_diversity_loss` | `0.999115` |
| `eval_loss` | `-0.079354` |
| `eval_mean_reward` | `0.133301` |
| `eval_policy_loss` | `3.9e-05` |
| `eval_selected_accuracy` | `0.128906` |
| `eval_selector_loss` | `-0.132912` |
| `eval_voting_accuracy` | `0.136719` |
| `lr` | `5.8e-05` |
| `train_anchor_loss` | `0.050796` |
| `train_coverage` | `0.348438` |
| `train_diversity_loss` | `0.99978` |
| `train_loss` | `-0.230694` |
| `train_mean_reward` | `0.283008` |
| `train_policy_loss` | `2.3e-05` |
| `train_selected_accuracy` | `0.285938` |
| `train_selector_loss` | `-0.283292` |
| `train_voting_accuracy` | `0.289062` |

### Training Curve Summary

- Steps recorded: 120 (step 10 → 1200)
- `train_mean_reward`: 0.251172 → 0.213867
- `train_selected_accuracy`: 0.245312 → 0.20625
- `train_loss`: -0.188944 → -0.156646

### Eval Curve Summary

- Eval snapshots: 3
- Best `eval_selected_accuracy`: 0.132812

---

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260422_163207-1lb2hkdi
- **Started:** 2026-04-22T20:32:07.025989Z
- **Git commit:** 9aee1e2351645ba37fab29b06502c4dc650b1993
- **Host:** gpu-b10-5.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `2` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `16` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `10000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `1.0` |
| `anchor_weight` | `0.05` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.622601` |
| `eval_coverage` | `0.119141` |
| `eval_diversity_loss` | `0.746311` |
| `eval_loss` | `0.750316` |
| `eval_mean_reward` | `0.024902` |
| `eval_policy_loss` | `-0.0002` |
| `eval_selected_accuracy` | `0.027344` |
| `eval_selector_loss` | `-0.026926` |
| `eval_voting_accuracy` | `0.027344` |
| `lr` | `9.5e-05` |
| `train_anchor_loss` | `1.07765` |
| `train_coverage` | `0.24375` |
| `train_diversity_loss` | `0.387518` |
| `train_loss` | `0.387119` |
| `train_mean_reward` | `0.060352` |
| `train_policy_loss` | `0.000195` |
| `train_selected_accuracy` | `0.05625` |
| `train_selector_loss` | `-0.054475` |
| `train_voting_accuracy` | `0.067187` |

### Training Curve Summary

- Steps recorded: 49 (step 10 → 490)
- `train_mean_reward`: 0.252148 → 0.05293
- `train_selected_accuracy`: 0.24375 → 0.053125
- `train_loss`: 0.761646 → 0.35966

### Eval Curve Summary

- Eval snapshots: 1
- Best `eval_selected_accuracy`: 0.027344

---

## coconut-generator-interaction-verifiable-rl-frozen-prm

- **Run ID:** offline-run-20260422_163207-j00wvmea
- **Started:** 2026-04-22T20:32:07.033873Z
- **Git commit:** 9aee1e2351645ba37fab29b06502c4dc650b1993
- **Host:** gpu-b10-4.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `2` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `1000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.2` |
| `anchor_weight` | `0.5` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.048356` |
| `eval_coverage` | `0.214844` |
| `eval_diversity_loss` | `0.999329` |
| `eval_loss` | `0.054692` |
| `eval_mean_reward` | `0.165771` |
| `eval_policy_loss` | `1e-06` |
| `eval_selected_accuracy` | `0.173828` |
| `eval_selector_loss` | `-0.169515` |
| `eval_voting_accuracy` | `0.167969` |
| `lr` | `9.4e-05` |
| `step` | `1000` |
| `train_anchor_loss` | `0.040559` |
| `train_coverage` | `0.384375` |
| `train_diversity_loss` | `0.999609` |
| `train_loss` | `-0.108527` |
| `train_mean_reward` | `0.327344` |
| `train_policy_loss` | `0.000122` |
| `train_selected_accuracy` | `0.3375` |
| `train_selector_loss` | `-0.329027` |
| `train_voting_accuracy` | `0.3375` |

### Training Curve Summary

- Steps recorded: 100 (step 10 → 1000)
- `train_mean_reward`: 0.294531 → 0.327344
- `train_selected_accuracy`: 0.290625 → 0.3375
- `train_loss`: -0.040414 → -0.108527

### Eval Curve Summary

- Eval snapshots: 2
- Best `eval_selected_accuracy`: 0.167969

---

## coconut-generator-interaction-verifiable-rl

- **Run ID:** offline-run-20260422_173615-vbtc349h
- **Started:** 2026-04-22T21:36:15.741427Z
- **Git commit:** 6383412d176bb1c282a10dc0f05fb24697e47960
- **Host:** gpu-b9-6.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `2` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `16` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `10000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `1.0` |
| `anchor_weight` | `0.05` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.397056` |
| `eval_coverage` | `0.095703` |
| `eval_diversity_loss` | `0.721054` |
| `eval_loss` | `0.73036` |
| `eval_mean_reward` | `0.015869` |
| `eval_policy_loss` | `0.014302` |
| `eval_selected_accuracy` | `0.025391` |
| `eval_selector_loss` | `-0.024849` |
| `eval_voting_accuracy` | `0.009766` |
| `lr` | `9.7e-05` |
| `train_anchor_loss` | `0.78143` |
| `train_coverage` | `0.296875` |
| `train_diversity_loss` | `0.394289` |
| `train_loss` | `0.391773` |
| `train_mean_reward` | `0.066211` |
| `train_policy_loss` | `0.032107` |
| `train_selected_accuracy` | `0.075` |
| `train_selector_loss` | `-0.073695` |
| `train_voting_accuracy` | `0.076563` |

### Training Curve Summary

- Steps recorded: 38 (step 10 → 380)
- `train_mean_reward`: 0.251367 → 0.036523
- `train_selected_accuracy`: 0.24375 → 0.054688
- `train_loss`: 0.760926 → 0.44731

### Eval Curve Summary

- Eval snapshots: 1
- Best `eval_selected_accuracy`: 0.025391

---

## coconut-generator-interaction-verifiable-rl-dense-credit

- **Run ID:** offline-run-20260422_191937-qpxxv6m4
- **Started:** 2026-04-22T23:19:37.900087Z
- **Git commit:** 7f0688e9daa01a960eb9b0d27a69ead845cad240
- **Host:** gpu-b9-6.zaratan.umd.edu  (4x NVIDIA A100-SXM4-40GB)

### Config

| Key | Value |
|-----|-------|
| `objective` | `verifiable_rl` |
| `communication_type` | `attention` |
| `communication_every` | `2` |
| `sampling_by` | `noise` |
| `noise_std` | `0.1` |
| `dropout_p` | `0.2` |
| `num_return_sequences` | `8` |
| `latent_length` | `6` |
| `per_device_train_batch_size` | `16` |
| `gradient_accumulation_steps` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `num_train_epochs` | `1` |
| `max_train_steps` | `10000` |
| `max_grad_norm` | `1.0` |
| `score_temperature` | `0.25` |
| `reward_baseline` | `leave_one_out` |
| `diversity_penalty_weight` | `0.2` |
| `anchor_weight` | `0.5` |
| `communication_attention_heads` | `4` |
| `communication_topk` | `2` |
| `init_communication_from` | `` |
| `model_id` | `checkpoints/coconut` |
| `prm_id` | `checkpoints/latentRM` |
| `metric_for_best_model` | `eval_selected_accuracy` |

### Final Metrics

| Metric | Value |
|--------|-------|
| `eval_anchor_loss` | `0.051637` |
| `eval_coverage` | `0.162109` |
| `eval_diversity_loss` | `0.998734` |
| `eval_loss` | `0.0995` |
| `eval_mean_reward` | `0.124268` |
| `eval_policy_loss` | `0.00035` |
| `eval_selected_accuracy` | `0.130859` |
| `eval_selector_loss` | `-0.126552` |
| `eval_voting_accuracy` | `0.123047` |
| `lr` | `9.9e-05` |
| `train_anchor_loss` | `0.037598` |
| `train_coverage` | `0.342187` |
| `train_diversity_loss` | `0.99978` |
| `train_loss` | `-0.055374` |
| `train_mean_reward` | `0.267969` |
| `train_policy_loss` | `0.000277` |
| `train_selected_accuracy` | `0.279687` |
| `train_selector_loss` | `-0.27459` |
| `train_voting_accuracy` | `0.273438` |

### Training Curve Summary

- Steps recorded: 501 (step 10 → 5020)
- `train_mean_reward`: 0.250781 → 0.296875
- `train_selected_accuracy`: 0.245312 → 0.30625
- `train_loss`: 0.010221 → -0.088206

### Eval Curve Summary

- Eval snapshots: 13
- Best `eval_selected_accuracy`: 0.132812

---
