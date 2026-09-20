#!/bin/bash
# Sequential warm-started TQC curriculum P1 -> P2 -> P3 -> P4 -> P5, then a
# PPO baseline on P1 for the report's algorithm comparison.
#
#   PYTHONPATH=. nohup bash scripts/run_curriculum.sh > runs/curriculum.log 2>&1 &
#
# Best-effort: if a phase underperforms the script still moves on (the PS
# explicitly rewards a partial, honestly-reported result over nothing).

set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="$ROOT"
PY="$ROOT/.venv/bin/python"
mkdir -p runs

# Trimmed budgets for a ~3-4h CPU pass at gradient_steps=2 (UTD 0.25, ~230 fps).
# The configs carry larger "intended" values; these override for the overnight run.
declare -A STEPS=( [1]=200000 [2]=550000 [3]=450000 [4]=900000 [5]=550000 )
declare -A NENVS=( [1]=8 [2]=8 [3]=8 [4]=8 [5]=8 )

prev=""
for ph in 1 2 3 4 5; do
  out="runs/phase${ph}_tqc"
  echo "=============================================================="
  echo ">>> $(date +%H:%M:%S)  PHASE $ph  warm_start='${prev}'"
  echo "=============================================================="
  args=(--config "configs/phase${ph}.yaml"
        --n-envs "${NENVS[$ph]}" --out-dir "$out" --eval-freq 50000 --seed 0)
  [ -n "${STEPS[$ph]:-}" ] && args+=(--timesteps "${STEPS[$ph]}")
  if [ -n "$prev" ]; then
    args+=(--warm-start "$prev")
    prev_norm="$(dirname "$prev")/vecnormalize.pkl"
    [ -f "$prev_norm" ] && args+=(--warm-start-norm "$prev_norm")
  fi
  "$PY" train.py "${args[@]}" > "${out}.log" 2>&1
  rc=$?
  echo ">>> phase $ph finished rc=$rc"
  if [ -f "$out/model.zip" ]; then
    prev="$out/model.zip"
  else
    echo "!!! phase $ph produced no model.zip; keeping previous warm-start"
  fi
done

echo ">>> $(date +%H:%M:%S)  PPO baseline on phase 1"
"$PY" train.py --config configs/ppo_baseline.yaml --timesteps 1500000 --n-envs 16 \
  --out-dir runs/phase1_ppo --eval-freq 100000 --seed 0 > runs/phase1_ppo.log 2>&1
echo ">>> PPO baseline finished rc=$?"

echo ">>> $(date +%H:%M:%S)  CURRICULUM DONE"
ls -la runs/*/model.zip 2>/dev/null
