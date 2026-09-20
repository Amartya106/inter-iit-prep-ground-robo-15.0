#!/usr/bin/env python3
"""
collect_demos_her.py -- Hindsight Experience Replay-style demo relabeling.

SCOPE, disclosed upfront: naive full HER (relabel the goal to wherever the
episode happened to end up, unconditionally) is only PHYSICALLY VALID here
for the align_only sub-task. Insertion depth's "goal" is tied to a real hole
with a fixed ~3mm clearance at a fixed location -- you cannot relabel
"target position" to wherever the peg ended up in free space, because there
is no hole there; a demo of "the peg reached 2cm depth at position X" is
only a valid demonstration of insertion AT X's actual hole, not transferable
to a different X. align_only has no such constraint -- manipularl/rewards.py's
align_only branch is PURE xy+tilt matching with zero dependency on hole
geometry (see PhaseConfig.align_only's docstring: "succeed once xy/tilt-
aligned above the hole mouth ... no depth requirement"). "The peg hovered
here, well-aligned" is therefore a valid align demo regardless of what (if
anything) is physically at that position.

This script rolls a source controller (default: scripted_expert.py, which
already has ~66-70% grasp_ok and ~52-85% align_ok conditional on grasp, see
E55/E56/E58) through the FULL cold-start task, then relabels every GRASPED
segment of every trajectory using the standard HER "future" strategy: for k
sampled future timesteps within a segment, the peg's OWN achieved (x, y, z)
at that future step becomes the goal for every earlier step in the segment,
with the SAME action that was actually taken (the core HER assumption: an
action that led toward SOME state is, by construction, a correct action for
reaching that state, even if it wasn't the state originally intended).

This turns ordinary transit/carry motion -- already the RELIABLE part of the
task -- into arbitrarily many align-only demonstrations for free, INCLUDING
from episodes that never got anywhere near the real hole and would otherwise
be discarded entirely by scripts/collect_demos.py's success-only filtering:
their carry motion is still a perfectly valid "align to SOMEWHERE" demo.

    PYTHONPATH=. .venv/bin/python scripts/collect_demos_her.py \
        --episodes 150 --out runs/demos_phase4_her
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd

from manipularl.env import ManipulaRLEnv
from manipularl.configs import TRAIN_SEED_LO
from scripts.scripted_expert import ScriptedManipulator

# manipularl/env.py's _assemble_obs() layout (123-dim, single raw frame --
# see EXPERIMENTS.md's exploration report for the full index table).
_GOAL_POS = slice(46, 49)
_XY_ERR_VEC = slice(55, 57)
_HEIGHT_GAP = 57
_PEG_POS = slice(31, 34)
_GRASPED = 30


def relabel_frame(raw_obs: np.ndarray, new_goal_xyz: np.ndarray) -> np.ndarray:
    """Return a COPY of raw_obs with goal-RELATIVE fields rewritten for a new
    3D goal position. peg_pos itself and peg_depth (obs[58]) are left
    untouched -- depth is NOT goal-relabeled, it is an absolute physical
    measurement tied to the episode's own real hole, not a function of
    `new_goal_xyz` (see this module's docstring for why depth-relabeling
    isn't physically valid the way xy/height relabeling is)."""
    out = raw_obs.copy()
    out[_GOAL_POS] = new_goal_xyz
    peg_xyz = raw_obs[_PEG_POS]
    out[_XY_ERR_VEC] = new_goal_xyz[:2] - peg_xyz[:2]
    out[_HEIGHT_GAP] = new_goal_xyz[2] - peg_xyz[2]
    return out


def collect_raw_trajectory(env, expert, max_steps):
    """One episode -> list of (raw_obs, action, peg_pos_xyz, grasped)."""
    env.reset()
    expert.reset()
    traj = []
    for _ in range(max_steps):
        s = env._state_dict()
        raw_obs = env._assemble_obs().astype(np.float32)
        a = expert.act(s).astype(np.float32)
        peg_xyz = np.asarray(s["peg_pos"], dtype=np.float32).copy()
        grasped = bool(s["grasped"])
        traj.append((raw_obs, a, peg_xyz, grasped))
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return traj


def relabel_episode(traj, k_per_segment=2, min_segment_len=8):
    """HER 'future' relabeling over one episode's grasped segments.
    Returns list of (relabeled_3frame_stack[369], action) pairs -- stacking
    is done HERE (not deferred) because all 3 frames in a stack must be
    relabeled to the SAME goal for internal consistency (the goal is
    constant across a real episode; a relabeled goal must stay just as
    constant within its own 3-frame window)."""
    out = []
    n = len(traj)
    i = 0
    while i < n:
        if not traj[i][3]:
            i += 1
            continue
        j = i
        while j < n and traj[j][3]:
            j += 1
        seg_len = j - i
        if seg_len >= min_segment_len:
            candidate_ks = sorted(set(
                int(x) for x in np.linspace(i + min_segment_len - 1, j - 1, num=k_per_segment)
            ))
            for k in candidate_ks:
                goal_xyz = traj[k][2].copy()
                for t in range(i, k):
                    frames = []
                    for f in (t - 2, t - 1, t):
                        idx = max(i, f)   # pad at the segment's own start, not episode start --
                                          # frames from BEFORE the grasp aren't valid under this goal
                        frames.append(relabel_frame(traj[idx][0], goal_xyz))
                    stacked = np.concatenate(frames, axis=0)   # oldest..newest, matches
                                                               # DreamAugmentCallback's own convention
                    out.append((stacked, traj[t][1]))
        i = j
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=TRAIN_SEED_LO)
    ap.add_argument("--k-per-segment", type=int, default=2,
                    help="number of future-goal checkpoints sampled per grasped segment")
    ap.add_argument("--min-segment-len", type=int, default=8,
                    help="a grasped run shorter than this contributes no relabeled demos")
    ap.add_argument("--out", default="runs/demos_phase4_her")
    args = ap.parse_args()

    base = ManipulaRLEnv(phase=args.phase, split="train", seed=args.seed, grasp_curriculum=0.0)
    expert = ScriptedManipulator(base, phase=args.phase)
    max_steps = base.max_steps

    all_obs, all_act = [], []
    n_segments_total = 0
    for ep in range(args.episodes):
        traj = collect_raw_trajectory(base, expert, max_steps)
        pairs = relabel_episode(traj, args.k_per_segment, args.min_segment_len)
        for o, a in pairs:
            all_obs.append(o); all_act.append(a)
        n_grasped_steps = sum(1 for row in traj if row[3])
        n_segments_total += n_grasped_steps
        if (ep + 1) % 25 == 0:
            print(f"  {ep+1}/{args.episodes} episodes, {len(all_obs)} relabeled pairs so far")
    base.close()

    if not all_obs:
        print("[collect_her] 0 relabeled pairs -- no grasped segment reached "
              f"min_segment_len={args.min_segment_len} in any episode.")
        return

    obs = np.stack(all_obs).astype(np.float32)
    act = np.stack(all_act).astype(np.float32)
    print(f"\n[collect_her] {obs.shape[0]} relabeled (obs, action) pairs, "
          f"obs_dim={obs.shape[1]}, act_dim={act.shape[1]}")

    # Sanity check: xy_err recomputed from the relabeled obs must EXACTLY
    # match the norm of the rewritten xy_err_vec field -- a cheap, direct
    # verification that the relabeling math is internally consistent before
    # trusting any of this for training.
    goal_xy = obs[:, 46:48]; peg_xy = obs[:, 31:33]
    recomputed = np.linalg.norm(goal_xy - peg_xy, axis=1)
    stored = np.linalg.norm(obs[:, 55:57], axis=1)
    max_err = float(np.max(np.abs(recomputed - stored)))
    print(f"[collect_her] consistency check: max|recomputed_xy_err - stored_xy_err| = {max_err:.2e} "
          f"({'OK' if max_err < 1e-5 else 'FAILED -- relabeling bug'})")

    # Fit a fresh VecNormalize on the relabeled obs (same pattern as
    # collect_multi_expert_demos.py: robust percentile-clipped stats, since
    # a handful of frames can carry large ft/collision-impact values that
    # would otherwise dominate the naive variance estimate).
    lo = np.percentile(obs, 0.5, axis=0); hi = np.percentile(obs, 99.5, axis=0)
    obs_clipped = np.clip(obs, lo, hi)
    obs_rms = RunningMeanStd(shape=(obs.shape[1],))
    obs_rms.update(obs_clipped)

    dummy = ManipulaRLEnv(phase=args.phase, split="train", seed=0)
    dummy_venv = DummyVecEnv([lambda: dummy])
    dummy_venv = VecFrameStack(dummy_venv, n_stack=3)
    shared_norm = VecNormalize(dummy_venv, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, training=False)
    shared_norm.obs_rms = obs_rms
    normalized_obs = shared_norm.normalize_obs(obs).astype(np.float32)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "demos.npz", obs=normalized_obs, act=act)
    shared_norm.save(str(out / "vecnormalize.pkl"))
    print(f"[collect_her] saved {out}/demos.npz (normalized) and {out}/vecnormalize.pkl")


if __name__ == "__main__":
    main()
