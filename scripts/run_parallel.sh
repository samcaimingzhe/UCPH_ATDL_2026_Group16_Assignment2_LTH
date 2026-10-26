#!/usr/bin/env bash
# Run the full experiment with one process per (seed, phase) spread over GPUs.
#
#   bash scripts/run_parallel.sh                      # preset=paper, seeds 0 1 2, all visible GPUs
#   PRESET=quick SEEDS="0" GPUS="0" bash scripts/run_parallel.sh
#   JOBS_PER_GPU=2 bash scripts/run_parallel.sh        # Conv-4 is small; 2-3 jobs per GPU is fine
#
# Phase 1: source IMP for every seed (in parallel).
# Phase 2: one process per (seed, target) (in parallel).
# Logs go to logs/<exp>/. Re-running resumes from where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

PRESET=${PRESET:-paper}
EXP=${EXP:-$PRESET}
SEEDS=${SEEDS:-"0 1 2"}
TARGETS=${TARGETS:-"svhn cifar100 fashionmnist"}
EXTRA=${EXTRA:-"--amp"}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
if [[ -z "${GPUS:-}" ]]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' || echo "0")
fi
read -r -a GPU_ARR <<< "$GPUS"
SLOTS=$(( ${#GPU_ARR[@]} * JOBS_PER_GPU ))
LOG=logs/$EXP; mkdir -p "$LOG"

echo "preset=$PRESET exp=$EXP seeds=[$SEEDS] targets=[$TARGETS] gpus=[$GPUS] slots=$SLOTS"

# download all datasets once (sequential, avoids races)
python -c "
from lth.data import load_raw
for d in ['cifar10', *'$TARGETS'.split()]:
    load_raw(d, 'data', download=True); print('ok', d)
"

run_batch () {   # run_batch "<cmd1>" "<cmd2>" ... with at most $SLOTS concurrent jobs
  local i=0
  for cmd in "$@"; do
    local gpu=${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}
    local name; name=$(echo "$cmd" | sed -E 's/.*--seeds ([0-9]+).*--phase ([a-z]+)( --targets ([a-z0-9]+))?.*/s\1_\2_\4/')
    echo "[gpu $gpu] $cmd"
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$LOG/$name.log" 2>&1 &
    i=$(( i + 1 ))
    if (( i % SLOTS == 0 )); then wait; fi
  done
  wait
}

cmds=()
for s in $SEEDS; do
  cmds+=("python run_transfer.py --preset $PRESET --exp $EXP --seeds $s --phase source $EXTRA")
done
run_batch "${cmds[@]}"

cmds=()
for s in $SEEDS; do
  for t in $TARGETS; do
    cmds+=("python run_transfer.py --preset $PRESET --exp $EXP --seeds $s --phase target --targets $t $EXTRA")
  done
done
run_batch "${cmds[@]}"

python analyze.py --exp "$EXP"
echo "all done -> results/$EXP/analysis"
