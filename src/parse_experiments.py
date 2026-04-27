"""
Parse wandb offline run directories into structured, AI-readable JSON/Markdown.

Usage:
    python -m src.parse_experiments [--wandb_dir wandb] [--output experiments.json] [--format json|md|both]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone


# Hyperparameter keys worth surfacing (skip verbose/infrastructure keys)
INTERESTING_CONFIG_KEYS = [
    "run_name", "objective", "communication_type", "communication_every",
    "sampling_by", "noise_std", "dropout_p", "num_return_sequences",
    "latent_length", "per_device_train_batch_size", "gradient_accumulation_steps",
    "learning_rate", "weight_decay", "warmup_ratio", "lr_scheduler_type",
    "num_train_epochs", "max_train_steps", "max_grad_norm",
    "score_temperature", "reward_baseline", "diversity_penalty_weight", "anchor_weight",
    "communication_attention_heads", "communication_topk", "init_communication_from",
    "model_id", "prm_id", "metric_for_best_model",
]

# Metrics to include in per-step history
TRAIN_METRICS = [
    "train_loss", "train_policy_loss", "train_selector_loss",
    "train_diversity_loss", "train_anchor_loss",
    "train_mean_reward", "train_coverage",
    "train_voting_accuracy", "train_selected_accuracy", "lr",
]
EVAL_METRICS = [
    "eval_loss", "eval_policy_loss", "eval_selector_loss",
    "eval_diversity_loss", "eval_anchor_loss",
    "eval_mean_reward", "eval_coverage",
    "eval_voting_accuracy", "eval_selected_accuracy",
]


def parse_config(config_path: Path) -> dict:
    """Parse config.yaml, returning only interesting keys with their values."""
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    config = {}
    for key in INTERESTING_CONFIG_KEYS:
        if key in raw:
            val = raw[key]
            # wandb wraps values in {"desc": ..., "value": ...}
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            config[key] = val
    return config


def parse_output_log(log_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Parse output.log, returning (train_steps, eval_steps).
    Lines are JSON objects; eval blobs lack a "step" key.
    """
    train_steps = []
    eval_steps = []
    with open(log_path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in obj and any(k.startswith("train_") for k in obj):
                entry = {"step": obj["step"]}
                for k in TRAIN_METRICS:
                    if k in obj:
                        entry[k] = round(obj[k], 6) if isinstance(obj[k], float) else obj[k]
                train_steps.append(entry)
            elif any(k.startswith("eval_") for k in obj):
                entry = {}
                for k in EVAL_METRICS:
                    if k in obj:
                        entry[k] = round(obj[k], 6) if isinstance(obj[k], float) else obj[k]
                if "step" in obj:
                    entry["step"] = obj["step"]
                eval_steps.append(entry)
    return train_steps, eval_steps


def parse_run(run_dir: Path) -> dict | None:
    files_dir = run_dir / "files"
    if not files_dir.exists():
        return None

    metadata_path = files_dir / "wandb-metadata.json"
    config_path = files_dir / "config.yaml"
    summary_path = files_dir / "wandb-summary.json"
    log_path = files_dir / "output.log"

    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    run = {
        "run_id": run_dir.name,
        "run_name": None,
        "started_at": metadata.get("started_at"),
        "git_commit": metadata.get("git", {}).get("commit"),
        "host": metadata.get("host"),
        "gpu": f"{metadata.get('gpu_count', '?')}x {metadata.get('gpu_type', '?')}",
        "config": {},
        "final_metrics": {},
        "train_history": [],
        "eval_history": [],
    }

    # Config
    if config_path.exists():
        try:
            run["config"] = parse_config(config_path)
            run["run_name"] = run["config"].pop("run_name", None)
        except Exception:
            pass

    # Summary (final metrics)
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        run["final_metrics"] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in summary.items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        }

    # Step history from output.log
    if log_path.exists():
        run["train_history"], run["eval_history"] = parse_output_log(log_path)

    return run


def runs_to_markdown(runs: list[dict]) -> str:
    lines = ["# Experiment Results\n"]
    for r in runs:
        name = r["run_name"] or r["run_id"]
        lines.append(f"## {name}\n")
        lines.append(f"- **Run ID:** {r['run_id']}")
        lines.append(f"- **Started:** {r['started_at']}")
        lines.append(f"- **Git commit:** {r['git_commit']}")
        lines.append(f"- **Host:** {r['host']}  ({r['gpu']})")

        lines.append("\n### Config\n")
        lines.append("| Key | Value |")
        lines.append("|-----|-------|")
        for k, v in r["config"].items():
            lines.append(f"| `{k}` | `{v}` |")

        lines.append("\n### Final Metrics\n")
        if r["final_metrics"]:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in sorted(r["final_metrics"].items()):
                lines.append(f"| `{k}` | `{v}` |")
        else:
            lines.append("_No final metrics recorded._")

        lines.append("\n### Training Curve Summary\n")
        if r["train_history"]:
            first = r["train_history"][0]
            last = r["train_history"][-1]
            lines.append(f"- Steps recorded: {len(r['train_history'])} (step {first['step']} → {last['step']})")
            for metric in ["train_mean_reward", "train_selected_accuracy", "train_loss"]:
                if metric in first and metric in last:
                    lines.append(f"- `{metric}`: {first[metric]} → {last[metric]}")
        else:
            lines.append("_No training history in output.log._")

        lines.append("\n### Eval Curve Summary\n")
        if r["eval_history"]:
            lines.append(f"- Eval snapshots: {len(r['eval_history'])}")
            best_acc = max((e.get("eval_selected_accuracy", 0) for e in r["eval_history"]), default=None)
            if best_acc is not None:
                lines.append(f"- Best `eval_selected_accuracy`: {best_acc}")
        else:
            lines.append("_No eval history._")

        lines.append("\n---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse wandb runs into AI-readable format")
    parser.add_argument("--wandb_dir", default="wandb", help="Path to wandb directory")
    parser.add_argument("--output", default="experiments", help="Output file stem (no extension)")
    parser.add_argument("--format", choices=["json", "md", "both"], default="both")
    args = parser.parse_args()

    wandb_dir = Path(args.wandb_dir)
    if not wandb_dir.exists():
        print(f"Error: {wandb_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(
        [d for d in wandb_dir.iterdir() if d.is_dir() and "run-" in d.name],
        key=lambda d: d.name,
    )

    runs = []
    for run_dir in run_dirs:
        parsed = parse_run(run_dir)
        if parsed:
            runs.append(parsed)
            name = parsed["run_name"] or parsed["run_id"]
            steps = len(parsed["train_history"])
            print(f"  parsed: {name} ({steps} train steps)")

    print(f"\nTotal runs parsed: {len(runs)}")

    if args.format in ("json", "both"):
        out_path = Path(args.output + ".json")
        with open(out_path, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"Wrote {out_path}")

    if args.format in ("md", "both"):
        out_path = Path(args.output + ".md")
        with open(out_path, "w") as f:
            f.write(runs_to_markdown(runs))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
