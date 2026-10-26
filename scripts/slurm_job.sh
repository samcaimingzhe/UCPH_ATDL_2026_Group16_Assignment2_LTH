#!/usr/bin/env bash
# Example SLURM array job: one task per seed (source phase, then all targets).
#   sbatch --array=0-2 scripts/slurm_job.sh
#SBATCH --job-name=lth-transfer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PRESET=${PRESET:-paper}
SEED=${SLURM_ARRAY_TASK_ID:-0}
python run_transfer.py --preset "$PRESET" --seeds "$SEED" --download --amp
