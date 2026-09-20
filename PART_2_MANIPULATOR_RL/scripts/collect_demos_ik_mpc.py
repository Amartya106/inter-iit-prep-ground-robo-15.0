#!/usr/bin/env python3
"""
collect_demos_ik_mpc.py -- combine E58 (compliant scripted IK) and E59
(MPC/CEM) demonstrations into ONE shared dataset, after E58/E59/E60 each
showed a real but PARTIAL result, and three independent findings converged
on the same lesson: single-source demos underperform combined sources --
E54's original finding, re-confirmed by both E59's and E60's own writeups
before I combined the two here.

Follows E54's PROVEN-correct multi-expert pattern, not a naive concatenation
of scripts/collect_demos.py's and scripts/collect_demos_mpc.py's own saved
demos.npz files -- those each fit their OWN VecNormalize independently, in
ONLINE/running-stats mode (`training=True`), so two real problems would
follow from combining them directly: (1) the two sources' saved obs arrays
sit on different, incomparable normalization scales, and (2) even within
one source, early-collected episodes were normalized under LESS-CONVERGED
running stats than later ones, so there is no single "the" stats snapshot
that exactly un-normalizes every row even from ONE file. This script instead
collects RAW (pre-normalization) observations directly from both sources
during THIS run, then fits exactly one shared, robust (percentile-clipped)
VecNormalize over the combined raw data at the end -- avoiding both issues
by construction, identical in spirit to collect_multi_expert_demos.py's own
approach for the three RL specialists.

    PYTHONPATH=. .venv/bin/python scripts/collect_demos_ik_mpc.py \
        --ik-episodes 150 --mpc-episodes 60 --out runs/demos_phase4_ik_mpc
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd

from manipularl.env import ManipulaRLEnv
from manipularl.configs import TRAIN_SEED_LO, get_phase
from manipularl.world_model import DynamicsModel
from manipularl.mpc import CEMPlanner
from scripts.scripted_expert import ScriptedManipulator


def _make_venv(seed, frame_stack):
    base = ManipulaRLEnv(phase=4, split="train", seed=seed, grasp_curriculum=0.0)
    venv = DummyVecEnv([lambda: base])
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, training=True)
    return base, venv


def collect_ik(episodes, frame_stack, min_depth_keep, seed):
    """E58's compliant scripted expert -- fast, no planning overhead."""
    base, venv = _make_venv(seed, frame_stack)
    expert = ScriptedManipulator(base, phase=4)
    max_steps = base.max_steps

    raw_obs_buf, act_buf = [], []
    n_success = 0
    obs = venv.reset()
    expert.reset()
    cur_obs, cur_act = [], []
    steps_in_ep = 0
    n_done = 0
    t0 = time.time()
    while n_done < episodes:
        s = base._state_dict()
        a = expert.act(s).reshape(1, -1)
        cur_obs.append(venv.get_original_obs()[0].copy())
        cur_act.append(a[0].copy())
        obs, _r, dones, infos = venv.step(a)
        steps_in_ep += 1
        if dones[0] or steps_in_ep >= max_steps:
            n_done += 1
            success = bool(infos[0].get("success"))
            keep = success or infos[0].get("max_depth", 0.0) > min_depth_keep
            if success:
                n_success += 1
            if keep:
                raw_obs_buf.extend(cur_obs); act_buf.extend(cur_act)
            cur_obs, cur_act = [], []
            steps_in_ep = 0
            if not dones[0]:
                obs = venv.reset()
            expert.reset()
            if n_done % 25 == 0:
                print(f"  [IK] {n_done}/{episodes} episodes, {n_success} success, "
                      f"{len(raw_obs_buf)} transitions kept, {(time.time()-t0)/n_done:.2f}s/ep")
    venv.close()
    print(f"[IK] {n_success}/{episodes} = {100*n_success/episodes:.0f}% success, "
          f"{len(raw_obs_buf)} transitions kept")
    return (np.stack(raw_obs_buf).astype(np.float32) if raw_obs_buf else np.zeros((0, 123 * frame_stack), np.float32),
            np.stack(act_buf).astype(np.float32) if act_buf else np.zeros((0, 8), np.float32))


def collect_mpc(episodes, frame_stack, min_depth_keep, seed, dynamics_model,
                horizon, n_candidates, n_iters, n_elites):
    """E59's CEM planner over the learned dynamics model."""
    ckpt = torch.load(dynamics_model, map_location="cpu", weights_only=False)
    model = DynamicsModel(obs_dim=ckpt["obs_mean"].shape[0], action_dim=ckpt["act_mean"].shape[0])
    model.load_state_dict(ckpt["state_dict"])
    planner = CEMPlanner(model, ckpt["obs_mean"], ckpt["obs_std"], ckpt["act_mean"], ckpt["act_std"],
                         action_dim=ckpt["act_mean"].shape[0], horizon=horizon,
                         n_candidates=n_candidates, n_elites=n_elites, n_iters=n_iters)

    base, venv = _make_venv(seed, frame_stack)
    max_steps = base.max_steps

    raw_obs_buf, act_buf = [], []
    n_success = 0
    obs = venv.reset()
    cur_obs, cur_act = [], []
    steps_in_ep = 0
    n_done = 0
    t0 = time.time()
    while n_done < episodes:
        raw_obs = base._assemble_obs()
        a = planner.plan(raw_obs).reshape(1, -1)
        cur_obs.append(venv.get_original_obs()[0].copy())
        cur_act.append(a[0].copy())
        obs, _r, dones, infos = venv.step(a)
        steps_in_ep += 1
        if dones[0] or steps_in_ep >= max_steps:
            n_done += 1
            success = bool(infos[0].get("success"))
            keep = success or infos[0].get("max_depth", 0.0) > min_depth_keep
            if success:
                n_success += 1
            if keep:
                raw_obs_buf.extend(cur_obs); act_buf.extend(cur_act)
            cur_obs, cur_act = [], []
            steps_in_ep = 0
            if not dones[0]:
                obs = venv.reset()
            if n_done % 10 == 0:
                print(f"  [MPC] {n_done}/{episodes} episodes, {n_success} success, "
                      f"{len(raw_obs_buf)} transitions kept, {(time.time()-t0)/n_done:.1f}s/ep")
    venv.close()
    print(f"[MPC] {n_success}/{episodes} = {100*n_success/episodes:.0f}% success, "
          f"{len(raw_obs_buf)} transitions kept")
    return (np.stack(raw_obs_buf).astype(np.float32) if raw_obs_buf else np.zeros((0, 123 * frame_stack), np.float32),
            np.stack(act_buf).astype(np.float32) if act_buf else np.zeros((0, 8), np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ik-episodes", type=int, default=150)
    ap.add_argument("--mpc-episodes", type=int, default=60)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--min-depth-keep", type=float, default=0.005)
    ap.add_argument("--dynamics-model", default="runs/dynamics_model_full4.pt")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--n-candidates", type=int, default=256)
    ap.add_argument("--n-iters", type=int, default=8)
    ap.add_argument("--n-elites", type=int, default=32)
    ap.add_argument("--out", default="runs/demos_phase4_ik_mpc")
    args = ap.parse_args()

    ik_obs, ik_act = collect_ik(args.ik_episodes, args.frame_stack, args.min_depth_keep,
                                seed=TRAIN_SEED_LO)
    mpc_obs, mpc_act = collect_mpc(args.mpc_episodes, args.frame_stack, args.min_depth_keep,
                                   seed=TRAIN_SEED_LO + 5000, dynamics_model=args.dynamics_model,
                                   horizon=args.horizon, n_candidates=args.n_candidates,
                                   n_iters=args.n_iters, n_elites=args.n_elites)

    obs = np.concatenate([ik_obs, mpc_obs], axis=0)
    act = np.concatenate([ik_act, mpc_act], axis=0)
    source = np.concatenate([np.zeros(len(ik_obs), np.int32), np.ones(len(mpc_obs), np.int32)])
    print(f"\n[combine] {obs.shape[0]} total transitions: "
          f"IK {len(ik_obs)} ({100*len(ik_obs)/max(1,obs.shape[0]):.1f}%), "
          f"MPC {len(mpc_obs)} ({100*len(mpc_obs)/max(1,obs.shape[0]):.1f}%)")

    if obs.shape[0] == 0:
        print("[combine] 0 transitions from either source -- nothing to save.")
        return

    # Shared, robust (percentile-clipped) VecNormalize -- E54's own fix for
    # collision-impact ft outliers dominating a naive variance estimate;
    # applies here too since both sources include real collision episodes.
    lo = np.percentile(obs, 0.5, axis=0); hi = np.percentile(obs, 99.5, axis=0)
    obs_clipped = np.clip(obs, lo, hi)
    obs_rms = RunningMeanStd(shape=(obs.shape[1],))
    obs_rms.update(obs_clipped)

    dummy = ManipulaRLEnv(phase=4, split="train", seed=0)
    dummy_venv = DummyVecEnv([lambda: dummy])
    if args.frame_stack > 1:
        dummy_venv = VecFrameStack(dummy_venv, n_stack=args.frame_stack)
    shared_norm = VecNormalize(dummy_venv, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, training=False)
    shared_norm.obs_rms = obs_rms
    normalized_obs = shared_norm.normalize_obs(obs).astype(np.float32)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "demos.npz", obs=normalized_obs, act=act, source=source)
    shared_norm.save(str(out / "vecnormalize.pkl"))
    print(f"[combine] saved {out}/demos.npz and {out}/vecnormalize.pkl")


if __name__ == "__main__":
    main()
