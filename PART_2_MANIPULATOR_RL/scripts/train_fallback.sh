#!/bin/bash
# Fallback-plan iteration 1 (unattended). Run after the TQC/PPO gamble was lost.
#
#   setsid bash -c 'bash scripts/train_fallback.sh > runs/fallback.log 2>&1' </dev/null & disown
#
#  1. PPO phase 2 (3M, warm <- phase1_ppo) with the rewards.py shaping fixes +
#     annealed grasp curriculum + higher ent_coef        -> runs/phase2_ppo
#  2. eval phase 2 (200 ep, eval seed split)             -> results/eval_phase2.csv
#  3. diagnose phase 2 (cold-start + pre-grasped)        -> results/diagnose_phase2_*.csv
#  4. IF phase-2 eval success >= GATE: chain PPO phase 3 -> 4 -> 5, eval each.
#     ELSE: stop, leave a marker for manual review.
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; export PYTHONPATH="$ROOT"
PY="$ROOT/.venv/bin/python"
export MANIPULARL_TORCH_THREADS="${MANIPULARL_TORCH_THREADS:-12}"
GATE="${GATE:-0.30}"
mkdir -p runs results
ts() { date +%H:%M:%S; }
log() { echo ">>> $(ts)  $*"; }

run_train() {  # <label> <config> <out> [extra args...]
  local label=$1 cfg=$2 out=$3; shift 3
  log "TRAIN $label  ($cfg -> $out)"
  "$PY" train.py --config "$cfg" --out-dir "$out" --eval-freq 200000 --seed 0 "$@" \
      > "${out}.log" 2>&1
  log "TRAIN $label  rc=$?  $( [ -f "$out/model.zip" ] && echo 'model.zip OK' || echo 'NO MODEL' )"
}
run_eval() {  # <run-dir> <phase> <out-csv>
  local run=$1 ph=$2 out=$3
  [ -f "$run/model.zip" ] || { log "EVAL skip $run (no model)"; return; }
  log "EVAL $run  phase=$ph"
  "$PY" evaluate.py --run "$run" --phase "$ph" --algo ppo --episodes 200 --out "$out" \
      >> "${run}.log" 2>&1
  log "EVAL $run  rc=$?  $( [ -f "$out" ] && tail -1 "$out" )"
}
succ_of() {  # <csv>  -> prints success_rate of the first data row
  [ -f "$1" ] && awk -F, 'NR==2{print $4}' "$1" || echo 0
}

log "=== FALLBACK ITER 1 START  torch_threads=$MANIPULARL_TORCH_THREADS  gate=$GATE ==="

# keep the interrupted TQC p2 evidence but start p2 PPO fresh
[ -d runs/phase2_ppo ] && mv runs/phase2_ppo "runs/phase2_ppo.superseded.$(date +%s)"

run_train "PPO-p2" configs/ppo_phase2.yaml runs/phase2_ppo \
    --warm-start runs/phase1_ppo/model.zip --warm-start-norm runs/phase1_ppo/vecnormalize.pkl
run_eval runs/phase2_ppo 2 results/eval_phase2.csv

log "DIAGNOSE phase 2"
"$PY" scripts/diagnose_phase2.py --run runs/phase2_ppo --mode coldstart  --episodes 150 \
    >> runs/phase2_ppo.log 2>&1
"$PY" scripts/diagnose_phase2.py --run runs/phase2_ppo --mode pregrasped --episodes 150 \
    >> runs/phase2_ppo.log 2>&1
grep -aA10 "=== diagnose_phase2" runs/phase2_ppo.log | tail -30

S2=$(succ_of results/eval_phase2.csv)
log "phase-2 eval success_rate = ${S2}  (gate ${GATE})"
if awk "BEGIN{exit !(${S2:-0} >= ${GATE})}"; then
  log "gate PASSED -> chaining phases 3 -> 5"
  prev="runs/phase2_ppo"
  for ph in 3 4 5; do
    out="runs/phase${ph}_ppo"
    ws=(); [ -f "$prev/model.zip" ] && ws=(--warm-start "$prev/model.zip" \
        --warm-start-norm "$prev/vecnormalize.pkl")
    run_train "PPO-p${ph}" "configs/ppo_phase${ph}.yaml" "$out" "${ws[@]}"
    run_eval "$out" "$ph" "results/eval_phase${ph}.csv"
    [ -f "$out/model.zip" ] && prev="$out"
  done
else
  log "gate FAILED -> stopping for manual review (see diagnose output above)"
  touch runs/FALLBACK_ITER1_NEEDS_REVIEW
fi

log "=== FALLBACK ITER 1 DONE ==="
ls -la runs/phase*_ppo/model.zip 2>/dev/null
for f in results/eval_phase*.csv; do echo "--- $f"; cat "$f"; done
