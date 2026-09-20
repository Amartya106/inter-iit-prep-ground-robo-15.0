#!/bin/bash
# PRIMARY curriculum: sequential warm-started PPO, P1 -> P2 -> P3 -> P4 -> P5.
#
#   setsid bash -c 'bash scripts/run_ppo_curriculum.sh > runs/ppo_curriculum.log 2>&1' </dev/null & disown
#
# Best-effort: a phase that underperforms does not stop the run.
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; export PYTHONPATH="$ROOT"
PY="$ROOT/.venv/bin/python"
mkdir -p runs

prev=""
for ph in 1 2 3 4 5; do
  out="runs/phase${ph}_ppo"
  echo "=============================================================="
  echo ">>> $(date +%H:%M:%S)  PPO PHASE $ph  warm_start='${prev}'"
  echo "=============================================================="
  args=(--config "configs/ppo_phase${ph}.yaml" --out-dir "$out"
        --eval-freq 200000 --seed 0)
  if [ -n "$prev" ]; then
    args+=(--warm-start "$prev")
    pn="$(dirname "$prev")/vecnormalize.pkl"
    [ -f "$pn" ] && args+=(--warm-start-norm "$pn")
  fi
  "$PY" train.py "${args[@]}" > "${out}.log" 2>&1
  echo ">>> PPO phase $ph rc=$? $(date +%H:%M:%S)"
  [ -f "$out/model.zip" ] && prev="$out/model.zip" || echo "!!! no model.zip for phase $ph"
done
echo ">>> $(date +%H:%M:%S)  PPO CURRICULUM DONE"
ls -la runs/phase*_ppo/model.zip 2>/dev/null
