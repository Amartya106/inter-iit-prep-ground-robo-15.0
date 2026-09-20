#!/bin/bash
# Wait for the from-scratch phase-2 run to finish, then chain PPO phases 3->4->5
# warm-started from the previous phase, evaluating each. Best-effort.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" MANIPULARL_TORCH_THREADS=12
PY=.venv/bin/python
ts(){ date +%H:%M:%S; }

# wait for phase-2 training + its eval/diagnose to complete
while pgrep -f "train.py --config configs/ppo_phase2_scratch" >/dev/null; do sleep 30; done
echo ">>> $(ts) phase-2 training finished; waiting for its eval/diagnose"
while pgrep -f "evaluate.py --run runs/phase2_ppo|diagnose_phase2.py --run runs/phase2_ppo" >/dev/null; do sleep 20; done
echo ">>> $(ts) phase-2 eval:"; tail -1 results/eval_phase2.csv

prev="runs/phase2_ppo"
for ph in 3 4 5; do
  out="runs/phase${ph}_ppo"
  [ -d "$out" ] && mv "$out" "${out}.old.$(date +%s)"
  ws=(); [ -f "$prev/model.zip" ] && ws=(--warm-start "$prev/model.zip" --warm-start-norm "$prev/vecnormalize.pkl")
  echo ">>> $(ts) TRAIN PPO-p${ph}  warm<-${prev}"
  $PY train.py --config "configs/ppo_phase${ph}.yaml" --out-dir "$out" --eval-freq 200000 --seed 0 "${ws[@]}" > "${out}.log" 2>&1
  echo ">>> $(ts) PPO-p${ph} rc=$?  $([ -f $out/model.zip ] && echo OK || echo NOMODEL)"
  $PY evaluate.py --run "$out" --phase "$ph" --algo ppo --episodes 200 --out "results/eval_phase${ph}.csv" >> "${out}.log" 2>&1
  echo ">>> $(ts) eval p${ph}: $(tail -1 results/eval_phase${ph}.csv)"
  [ -f "$out/model.zip" ] && prev="$out"
done
echo ">>> $(ts) CHAIN DONE"
for f in results/eval_phase*.csv; do echo "--- $f"; cat "$f"; done
