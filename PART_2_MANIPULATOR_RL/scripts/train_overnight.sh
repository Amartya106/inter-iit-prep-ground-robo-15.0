#!/bin/bash
# Unattended overnight training for Part 2.
#
#   setsid bash -c 'bash scripts/train_overnight.sh > runs/overnight.log 2>&1' </dev/null & disown
#
# Sequential, checkpointing, best-effort (a failed phase does not abort the rest):
#   1. TQC phase 1 (reach)            -> runs/phase1_tqc      [TQC warm-start base + comparison number]
#   2. TQC phase 2 (pick&place)       -> runs/phase2_tqc      [off-policy solve attempt]
#   3. PPO phase 2 (full 3M budget)   -> runs/phase2_ppo      [workhorse; carries phases 3-5]
#   4. eval phase2 {tqc, ppo}         -> results/eval_phase2*.csv
#   5. PPO phase 3 (warm <- ppo p2)   -> runs/phase3_ppo ; eval
#   6. PPO phase 4 (warm <- ppo p3)   -> runs/phase4_ppo ; eval
#   7. PPO phase 5 (warm <- ppo p4)   -> runs/phase5_ppo ; eval
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; export PYTHONPATH="$ROOT"
PY="$ROOT/.venv/bin/python"
export MANIPULARL_TORCH_THREADS="${MANIPULARL_TORCH_THREADS:-10}"
mkdir -p runs results logs
ts() { date +%H:%M:%S; }
log() { echo ">>> $(ts)  $*"; }

# keep the interrupted 800k PPO phase-2 model around
if [ -d runs/phase2_ppo ] && [ ! -d runs/phase2_ppo_800k ]; then
  mv runs/phase2_ppo runs/phase2_ppo_800k
  mv runs/phase2_ppo.log runs/phase2_ppo_800k.log 2>/dev/null || true
fi

run_train() {  # <label> <config> <out> [extra args...]
  local label=$1 cfg=$2 out=$3; shift 3
  log "TRAIN $label  ($cfg -> $out)"
  "$PY" train.py --config "$cfg" --out-dir "$out" --eval-freq 200000 --seed 0 "$@" \
      > "${out}.log" 2>&1
  log "TRAIN $label  rc=$?  $( [ -f "$out/model.zip" ] && echo 'model.zip OK' || echo 'NO MODEL' )"
}

run_eval() {  # <run-dir> <phase> <algo> <out-csv>
  local run=$1 ph=$2 algo=$3 out=$4
  [ -f "$run/model.zip" ] || { log "EVAL skip $run (no model)"; return; }
  log "EVAL $run  phase=$ph algo=$algo"
  "$PY" evaluate.py --run "$run" --phase "$ph" --algo "$algo" --episodes 200 \
      --out "$out" >> "${run}.log" 2>&1
  log "EVAL $run  rc=$?"
  [ -f "$out" ] && tail -1 "$out"
}

log "=== OVERNIGHT START  torch_threads=$MANIPULARL_TORCH_THREADS ==="

# 1. TQC phase 1  (reach; fast to solve, gives TQC a warm-start base)
run_train "TQC-p1" configs/phase1.yaml runs/phase1_tqc --timesteps 600000
run_eval runs/phase1_tqc 1 tqc results/eval_phase1_tqc.csv

# 2. TQC phase 2  (warm from TQC p1 if it produced a model)
P1T_ARGS=()
[ -f runs/phase1_tqc/model.zip ] && P1T_ARGS=(--warm-start runs/phase1_tqc/model.zip \
    --warm-start-norm runs/phase1_tqc/vecnormalize.pkl)
run_train "TQC-p2" configs/phase2.yaml runs/phase2_tqc "${P1T_ARGS[@]}"
run_eval runs/phase2_tqc 2 tqc results/eval_phase2_tqc.csv

# 3. PPO phase 2  (fresh, full budget, warm from the solved PPO phase 1)
run_train "PPO-p2" configs/ppo_phase2.yaml runs/phase2_ppo \
    --warm-start runs/phase1_ppo/model.zip --warm-start-norm runs/phase1_ppo/vecnormalize.pkl
run_eval runs/phase2_ppo 2 ppo results/eval_phase2_ppo.csv
cp -f results/eval_phase2_ppo.csv results/eval_phase2.csv 2>/dev/null || true

# 4-7. PPO curriculum phases 3 -> 5, each warm-started from the previous PPO phase.
prev="runs/phase2_ppo"
for ph in 3 4 5; do
  out="runs/phase${ph}_ppo"
  ws=()
  [ -f "$prev/model.zip" ] && ws=(--warm-start "$prev/model.zip" \
      --warm-start-norm "$prev/vecnormalize.pkl")
  run_train "PPO-p${ph}" "configs/ppo_phase${ph}.yaml" "$out" "${ws[@]}"
  run_eval "$out" "$ph" ppo "results/eval_phase${ph}.csv"
  [ -f "$out/model.zip" ] && prev="$out"
done

log "=== OVERNIGHT DONE ==="
log "models:"; ls -la runs/phase*/model.zip 2>/dev/null
log "evals:";  for f in results/eval_phase*.csv; do echo "--- $f"; cat "$f"; done
