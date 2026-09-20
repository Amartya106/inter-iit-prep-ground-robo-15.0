#!/usr/bin/env python3
"""
collect_demos_mpc.py -- roll a CEM planner (manipularl/mpc.py) over the
learned dynamics model through the REAL environment, saving (obs, action)
pairs the same way scripts/collect_demos.py does, for behaviour cloning.

User-directed: "is there a better way to get expert demos than IK" ->
"Try 1. And 3." This is 3 -- MPC, a genuinely different KIND of demonstrator
from scripts/scripted_expert.py's memoryless kinematic IK: at every real
step it samples many candidate action sequences, rolls each one forward
through a LEARNED model of the environment's own dynamics, scores the
predicted outcomes, and only then commits to the best first action --
planning ahead using a model of consequences, not reacting to a static
target. The actions actually EXECUTED and RECORDED are always real
env.step() results (the model is only used for planning/scoring inside
CEMPlanner.plan(), never to fabricate a transition), so this is a genuine
demonstration source, not synthetic/imagined data.

Verified before building this (see EXPERIMENTS.md E59): random search over
(horizon, n_candidates, n_iters) found n_iters (CEM refinement passes) matters
far more than horizon length or candidate count -- more iterations lets the
search distribution narrow onto good solutions despite the model's own
imperfection, while a longer horizon just compounds that same imperfection
faster than it helps.

    PYTHONPATH=. .venv/bin/python scripts/collect_demos_mpc.py \
        --episodes 200 --out runs/demos_phase4_mpc
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from manipularl.env import ManipulaRLEnv
from manipularl.configs import TRAIN_SEED_LO, get_phase
from manipularl.world_model import DynamicsModel
from manipularl.mpc import CEMPlanner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--seed", type=int, default=TRAIN_SEED_LO)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--min-depth-keep", type=float, default=0.0,
                    help="keep episodes reaching at least this real insertion "
                         "depth even without a full held success -- see "
                         "collect_demos.py's own flag for the rationale "
                         "(this task's success rate is low enough that "
                         "success-only would keep almost nothing).")
    ap.add_argument("--dynamics-model", default="runs/dynamics_model_full4.pt")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--n-candidates", type=int, default=256)
    ap.add_argument("--n-iters", type=int, default=8,
                    help="CEM refinement passes -- the dominant lever, "
                         "verified directly (E59): more iterations beats "
                         "more candidates or a longer horizon for this "
                         "model's accuracy level.")
    ap.add_argument("--n-elites", type=int, default=None,
                    help="default: n_candidates // 8")
    args = ap.parse_args()

    out_default = f"runs/demos_phase{args.phase}_mpc"
    max_steps_default = get_phase(args.phase).max_steps
    args.out = args.out or out_default
    args.max_steps = args.max_steps or max_steps_default
    n_elites = args.n_elites or max(16, args.n_candidates // 8)

    ckpt = torch.load(args.dynamics_model, map_location="cpu", weights_only=False)
    model = DynamicsModel(obs_dim=ckpt["obs_mean"].shape[0], action_dim=ckpt["act_mean"].shape[0])
    model.load_state_dict(ckpt["state_dict"])
    planner = CEMPlanner(model, ckpt["obs_mean"], ckpt["obs_std"], ckpt["act_mean"], ckpt["act_std"],
                         action_dim=ckpt["act_mean"].shape[0], horizon=args.horizon,
                         n_candidates=args.n_candidates, n_elites=n_elites, n_iters=args.n_iters)
    print(f"[collect_mpc] planner: horizon={args.horizon} n_candidates={args.n_candidates} "
          f"n_elites={n_elites} n_iters={args.n_iters}, model={args.dynamics_model}")

    base = ManipulaRLEnv(phase=args.phase, split="train", seed=args.seed, grasp_curriculum=0.0)
    venv = DummyVecEnv([lambda: base])
    if args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, training=True)

    obs = venv.reset()
    cur_obs, cur_act = [], []
    kept_obs, kept_act = [], []
    n_done = n_success = 0
    steps_in_ep = 0
    t_start = time.time()

    while n_done < args.episodes:
        raw_obs = base._assemble_obs()   # the planner's own native (unstacked, unnormalized) space
        a = planner.plan(raw_obs).reshape(1, -1)
        cur_obs.append(np.asarray(obs[0], np.float32))
        cur_act.append(np.asarray(a[0], np.float32))
        obs, _r, dones, infos = venv.step(a)
        steps_in_ep += 1
        if dones[0] or steps_in_ep >= args.max_steps:
            n_done += 1
            success = bool(infos[0].get("success"))
            keep = success or infos[0].get("max_depth", 0.0) > args.min_depth_keep
            if success:
                n_success += 1
            if keep:
                kept_obs.extend(cur_obs)
                kept_act.extend(cur_act)
            cur_obs, cur_act = [], []
            steps_in_ep = 0
            if not dones[0]:
                obs = venv.reset()
            if n_done % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  {n_done}/{args.episodes} episodes, {n_success} success, "
                      f"{len(kept_obs)} transitions kept, "
                      f"{elapsed/n_done:.1f}s/ep, ETA {(args.episodes-n_done)*elapsed/n_done/60:.1f}min")

    print(f"\n[collect_mpc] CEM planner {n_success}/{n_done} = {100*n_success/n_done:.0f}% success")
    if not kept_obs:
        print(f"[collect_mpc] kept 0 transitions -- nothing to save. "
              f"Try more --episodes or --min-depth-keep.")
        venv.close()
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    O = np.asarray(kept_obs, np.float32); A = np.asarray(kept_act, np.float32)
    np.savez_compressed(out / "demos.npz", obs=O, act=A)
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"[collect_mpc] kept {O.shape[0]} transitions (obs dim {O.shape[1]}) -> {out}/demos.npz")
    venv.close()


if __name__ == "__main__":
    main()
