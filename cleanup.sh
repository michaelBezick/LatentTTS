#!/usr/bin/env bash
# cleanup.sh — free disk space in the LatentTTS repo
#
# Levels:
#   ./cleanup.sh          — safe: wandb runs, SLURM logs, __pycache__, .ruff_cache
#   ./cleanup.sh --outputs — also wipe outputs/ (trained checkpoints)
#   ./cleanup.sh --all     — everything above
#
# Always prints sizes before/after. Dry-run with --dry-run.

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"

DRY=0
CLEAN_OUTPUTS=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY=1 ;;
    --outputs)   CLEAN_OUTPUTS=1 ;;
    --all)       CLEAN_OUTPUTS=1 ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

rm_dir() {
  local path="$1"
  if [[ ! -e "$path" ]]; then return; fi
  local size
  size=$(du -sh "$path" 2>/dev/null | cut -f1)
  echo "  removing $path  ($size)"
  if [[ $DRY -eq 0 ]]; then rm -rf "$path"; fi
}

rm_find() {
  local label="$1"; shift
  local count
  count=$(find "$@" 2>/dev/null | wc -l)
  if [[ $count -eq 0 ]]; then return; fi
  echo "  removing $count $label"
  if [[ $DRY -eq 0 ]]; then find "$@" -exec rm -rf {} + 2>/dev/null || true; fi
}

echo "=== LatentTTS cleanup ==="
[[ $DRY -eq 1 ]] && echo "(dry-run — nothing will be deleted)"
echo

before=$(du -sh "$REPO" --exclude=.git 2>/dev/null | cut -f1)
echo "Before: $before"
echo

# --- Always safe ---
echo "[safe] wandb offline runs"
# keep latest-run symlink and any synced run-* dirs; only remove offline-run-* dirs
for d in "$REPO"/wandb/offline-run-*/; do
  rm_dir "$d"
done

echo "[safe] SLURM logs"
rm_dir "$REPO/logs"

echo "[safe] Python bytecode / tool caches"
rm_find "__pycache__ dirs" "$REPO/src" -type d -name "__pycache__"
rm_dir "$REPO/.ruff_cache"

# --- Optional: outputs ---
if [[ $CLEAN_OUTPUTS -eq 1 ]]; then
  echo "[--outputs] trained checkpoints"
  rm_dir "$REPO/outputs"
fi

echo
after=$(du -sh "$REPO" --exclude=.git 2>/dev/null | cut -f1)
echo "After:  $after"
echo "Done."
