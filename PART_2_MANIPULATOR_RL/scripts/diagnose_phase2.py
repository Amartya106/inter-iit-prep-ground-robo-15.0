#!/usr/bin/env python3
"""
diagnose_phase2.py -- localise WHICH sub-task of pick-and-place a trained
policy fails at, so reward/curriculum fixes are targeted, not blind.

Rolls the policy for N episodes and, per episode, records how far it got
along the chain:  approach -> grasp -> carry -> reach-target -> release.

    python scripts/diagnose_phase2.py --run runs/phase2_ppo_800k --mode coldstart
    python scripts/diagnose_phase2.py --run runs/phase2_ppo_800k --mode pregrasped

Modes
  coldstart   eval-split layouts, normal start (peg on the table).      -> tests the whole chain
  pregrasped  train-split layouts, EVERY episode starts peg-in-hand.    -> tests carry + release only
              (the pre-grasp curriculum only fires for split=="train")
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from manipularl.env import ManipulaRLEnv
from manipularl.configs import get_phase, EVAL_SEED_LO, TRAIN_SEED_LO


def load_policy(run_dir, algo):
    run = Path(run_dir)
    mp = run / "model.zip"
    if not mp.exists():
        mp = run / "best" / "best_model.zip"
    Model = TQC if algo == "tqc" else PPO
    return Model.load(str(mp), device="cpu"), str(run / "vecnormalize.pkl")


def build_env(mode, n_stack, norm_path):
    if mode == "coldstart":
        base = ManipulaRLEnv(phase=2, split="eval", seed=EVAL_SEED_LO, grasp_curriculum=0.0)
    else:  # pregrasped
        base = ManipulaRLEnv(phase=2, split="train", seed=TRAIN_SEED_LO, grasp_curriculum=1.0)
    venv = DummyVecEnv([lambda: base])
    if n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack)
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv, base


def rollout(model, venv, base, episodes, place_tol):
    rows = []
    for ep in range(episodes):
        obs = venv.reset()
        done = False
        first_step = True
        pregrasp_ok = False
        min_ee_peg = np.inf
        grasp_formed = False
        t_first_grasp = -1
        max_peg_z = -np.inf
        min_pg_grasped = np.inf
        reached_target = False
        released_after_reach = False
        t = 0
        last_info = {}
        while not done:
            s = base.unwrapped._state_dict()
            ee, peg, goal = s["ee_pos"], s["peg_pos"], s["goal_pos"]
            grasped = bool(s["grasped"])
            d_ee_peg = float(np.linalg.norm(ee - peg))
            d_pg = float(np.linalg.norm(peg - goal))
            if first_step:
                pregrasp_ok = grasped
                first_step = False
            min_ee_peg = min(min_ee_peg, d_ee_peg)
            max_peg_z = max(max_peg_z, float(peg[2]))
            if grasped:
                if not grasp_formed:
                    grasp_formed, t_first_grasp = True, t
                min_pg_grasped = min(min_pg_grasped, d_pg)
                if d_pg < place_tol:
                    reached_target = True
            elif reached_target:
                released_after_reach = True

            action, _ = model.predict(obs, deterministic=True)
            obs, _r, dones, infos = venv.step(action)
            done = bool(dones[0]); last_info = infos[0]; t += 1
        rows.append(dict(
            pregrasp_ok=int(pregrasp_ok),
            min_ee_peg=min_ee_peg,
            grasp_formed=int(grasp_formed),
            t_first_grasp=t_first_grasp,
            max_peg_z=max_peg_z,
            min_pg_grasped=(min_pg_grasped if np.isfinite(min_pg_grasped) else np.nan),
            reached_target=int(reached_target),
            released_after_reach=int(released_after_reach),
            success=int(bool(last_info.get("success", False))),
            steps=t,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase2_ppo_800k")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "tqc"])
    ap.add_argument("--mode", default="coldstart", choices=["coldstart", "pregrasped"])
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, norm_path = load_policy(args.run, args.algo)
    place_tol = get_phase(2).place_success_dist
    venv, base = build_env(args.mode, args.frame_stack, norm_path)
    rows = rollout(model, venv, base, args.episodes, place_tol)
    venv.close()

    a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    n = len(rows)

    def pct(x):
        return 100.0 * np.nanmean(x)

    # condition carry stats on "peg actually in hand at some point"
    held = a["grasp_formed"] > 0.5 if args.mode == "coldstart" else a["pregrasp_ok"] > 0.5

    print(f"\n=== diagnose_phase2  run={args.run}  mode={args.mode}  n={n} ===")
    print(f"  approach:  min EE-peg dist   median {np.median(a['min_ee_peg'])*100:5.1f} cm   "
          f"(<9cm trigger reached: {pct(a['min_ee_peg'] < 0.09):4.1f}%)")
    if args.mode == "pregrasped":
        print(f"  pre-grasp took (IK ok):   {pct(a['pregrasp_ok']):5.1f}%")
    print(f"  grasp formed (ever):      {pct(a['grasp_formed']):5.1f}%   "
          f"median t_first_grasp {np.median(a['t_first_grasp'][a['grasp_formed']>0.5]) if held.any() else float('nan'):.0f}")
    print(f"  carry:   max peg height    median {np.median(a['max_peg_z'])*100:5.1f} cm   "
          f"(lifted >2cm off table: {pct(a['max_peg_z'] > 0.07):4.1f}%)")
    if held.any():
        print(f"  carry:   min |peg-goal| while grasped   median "
              f"{np.nanmedian(a['min_pg_grasped'][held])*100:5.1f} cm   "
              f"(<{place_tol*100:.0f}cm target: {100.0*np.nanmean(a['reached_target'][held]):4.1f}%)")
        print(f"  release: released after reaching target (of those that reached): "
              f"{100.0*np.nanmean(a['released_after_reach'][a['reached_target']>0.5]) if (a['reached_target']>0.5).any() else 0.0:4.1f}%")
    print(f"  SUCCESS:                  {pct(a['success']):5.1f}%   mean steps {a['steps'].mean():.0f}")
    print()

    out = Path(args.out or f"results/diagnose_phase2_{args.mode}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[diagnose] wrote {out}")


if __name__ == "__main__":
    main()
