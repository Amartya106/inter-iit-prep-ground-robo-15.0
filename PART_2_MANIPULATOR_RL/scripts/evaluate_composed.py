"""
evaluate_composed.py -- composed two-stage Phase 4 evaluation.

Runs Task 1's ("align") policy from a cold reset until its align-only
success condition fires (or a step budget elapses -> hand-off failure),
hands off via env._regrasp_with_offset() (swaps the grasp constraint from
zero-offset to the depth-enabling _GRASP_OFFSET_Z at the peg's CURRENT pose
-- no teleport), then runs Task 2's ("insert") policy for the actual
insertion attempt. Reports the same aggregate metrics evaluate.py produces,
but for the full composed pipeline -- the real test of whether the
SeqPolicy-style task split solves Phase 4 end-to-end, comparable against
the monolithic baselines (E10/E16/E17/E28/E30) and each sub-policy's own
isolated eval (E31-E34).

    python scripts/evaluate_composed.py \
        --task1-run runs/phase4_task1_align_v3 --task2-run runs/phase4_task2_insert \
        --episodes 200 --out results/eval_phase4_composed.csv

Each stage uses its OWN policy's VecNormalize stats (fit during its own
training), applied manually here rather than through a live VecEnv, since
both stages must share the SAME underlying ManipulaRLEnv instance across
the hand-off (the whole point is to carry the real simulation state -- arm
pose, peg pose, grasp constraint -- from stage 1 into stage 2, not reset
between them). VecFrameStack sits BELOW VecNormalize in the training stack
(manipularl/make_env.py) -- stack raw observations first, normalize the
stacked vector second -- replicated exactly by FrameStacker below.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from manipularl.env import ManipulaRLEnv
from manipularl.configs import EVAL_SEED_LO


def load_policy(run_dir: str):
    run = Path(run_dir)
    model_path = run / "model.zip"
    if not model_path.exists():
        model_path = run / "best" / "best_model.zip"
    return PPO.load(str(model_path), device="cpu"), str(run / "vecnormalize.pkl")


def load_norm_stats(norm_path: str):
    """Pull obs_rms mean/var/clip_obs/epsilon out of a saved VecNormalize
    without needing a live VecEnv to load it onto."""
    with open(norm_path, "rb") as fh:
        vn = pickle.load(fh)
    return vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon


def normalize(stacked_raw, mean, var, clip_obs, eps):
    x = (stacked_raw - mean) / np.sqrt(var + eps)
    return np.clip(x, -clip_obs, clip_obs).astype(np.float32)


class FrameStacker:
    """Minimal re-implementation of VecFrameStack's per-env ring buffer."""

    def __init__(self, obs_dim: int, n_stack: int):
        self.n_stack = n_stack
        self.obs_dim = obs_dim
        self.buf = np.zeros(obs_dim * n_stack, dtype=np.float32)

    def reset(self, first_obs):
        self.buf[:] = 0.0
        for i in range(self.n_stack):
            self.buf[i * self.obs_dim:(i + 1) * self.obs_dim] = first_obs

    def push(self, obs):
        self.buf[:-self.obs_dim] = self.buf[self.obs_dim:]
        self.buf[-self.obs_dim:] = obs
        return self.buf.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task1-run", required=True)
    ap.add_argument("--task2-run", required=True)
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--task1-max-steps", type=int, default=400,
                    help="step budget for stage 1's align attempt before "
                         "counting the episode as a hand-off failure")
    ap.add_argument("--handoff-mode", choices=["success", "xy_thresh"], default="success",
                    help="'success' (default): hand off when stage-1's own "
                         "info['success'] fires -- needs a policy actually "
                         "trained on cfg_overrides.align_only (e.g. Task 1). "
                         "'xy_thresh': hand off once grasped and xy_err to "
                         "the hole < --handoff-xy-tol, held for "
                         "--handoff-hold-steps, checked directly off the "
                         "base env's own state -- works for ANY stage-1 "
                         "policy, including a raw Phase-3 checkpoint that "
                         "was never trained on align_only at all (its "
                         "'goal_pos' observation feature already maps onto "
                         "the hole's location in phase 4's scene, the same "
                         "warm-start-transfer premise every Phase 4 attempt "
                         "has relied on so far).")
    ap.add_argument("--handoff-xy-tol", type=float, default=0.03,
                    help="xy_thresh mode only: xy tolerance (m) to count as aligned")
    ap.add_argument("--handoff-hold-steps", type=int, default=5,
                    help="xy_thresh mode only: consecutive steps the tolerance must hold")
    ap.add_argument("--task1-align-only", action="store_true",
                    help="apply cfg_overrides={'align_only': True} to the shared env "
                         "(only meaningful/needed for --handoff-mode success)")
    ap.add_argument("--out", default="results/eval_phase4_composed.csv")
    args = ap.parse_args()

    model1, norm1_path = load_policy(args.task1_run)
    model2, norm2_path = load_policy(args.task2_run)
    mean1, var1, clip1, eps1 = load_norm_stats(norm1_path)
    mean2, var2, clip2, eps2 = load_norm_stats(norm2_path)

    cfg_overrides = {"align_only": True} if args.task1_align_only else None
    base = ManipulaRLEnv(phase=args.phase, split="eval", seed=EVAL_SEED_LO,
                         cfg_overrides=cfg_overrides)
    obs_dim = base.obs_dim
    fs1 = FrameStacker(obs_dim, args.frame_stack)
    fs2 = FrameStacker(obs_dim, args.frame_stack)

    rows = []
    for ep in range(args.episodes):
        raw_obs, _ = base.reset()
        fs1.reset(raw_obs)
        done = False
        t = 0
        aligned = False
        hold = 0
        info = {}
        # --- stage 1: Task 1 (or a raw Phase-3 checkpoint) drives toward the hole ---
        while not done and t < args.task1_max_steps:
            stacked = fs1.buf.copy() if t == 0 else fs1.push(raw_obs)
            nobs = normalize(stacked, mean1, var1, clip1, eps1)
            action, _ = model1.predict(nobs[None, :], deterministic=True)
            raw_obs, _r, term, trunc, info = base.step(action[0])
            done = bool(term or trunc)
            t += 1
            if args.handoff_mode == "success":
                if info.get("success"):
                    aligned = True
                    break
            else:
                s = base._state_dict()
                xy_err = float(np.linalg.norm(s["peg_pos"][:2] - s["hole_xy"]))
                ok_now = bool(s["grasped"]) and xy_err < args.handoff_xy_tol
                hold = hold + 1 if ok_now else 0
                if hold >= args.handoff_hold_steps:
                    aligned = True
                    break

        collided1 = bool(info.get("episode_collided", False))
        if not aligned:
            # never aligned within budget, or the env itself ended
            # (collision/truncation) first -- hand-off fails, overall
            # failure for this episode
            rows.append(dict(success=0, collided=int(collided1), steps=t,
                             stage1_aligned=0, insert_depth=0.0,
                             insert_tilt=np.nan, grasp_failure_rate=0.0))
            continue

        # --- hand-off: swap grasp to the depth-enabling top-offset,
        # no teleport, peg stays exactly where stage 1 left it ---
        ok = base._regrasp_with_offset()
        if not ok:
            rows.append(dict(success=0, collided=int(collided1), steps=t,
                             stage1_aligned=1, insert_depth=0.0,
                             insert_tilt=np.nan, grasp_failure_rate=0.0))
            continue

        # --- stage 2: Task 2 policy attempts insertion from the hand-off state ---
        raw_obs = base._assemble_obs()
        fs2.reset(raw_obs)
        done2 = False
        t2 = 0
        info2 = {}
        while not done2:
            stacked = fs2.buf.copy() if t2 == 0 else fs2.push(raw_obs)
            nobs = normalize(stacked, mean2, var2, clip2, eps2)
            action, _ = model2.predict(nobs[None, :], deterministic=True)
            raw_obs, _r, term, trunc, info2 = base.step(action[0])
            done2 = bool(term or trunc)
            t2 += 1

        ga = info2.get("grasp_attempts", 0)
        gf = info2.get("grasp_failures", 0)
        rows.append(dict(
            success=int(bool(info2.get("success", False))),
            collided=int(bool(info2.get("episode_collided", False)) or collided1),
            steps=t + t2, stage1_aligned=1,
            insert_depth=float(info2.get("max_depth", 0.0)),
            insert_tilt=float(info2.get("min_tilt_deg", np.nan)),
            grasp_failure_rate=(gf / ga) if ga else 0.0,
        ))
        if (ep + 1) % 50 == 0:
            print(f"  {ep+1}/{args.episodes} episodes done")

    base.close()

    n = len(rows)
    succ = np.array([r["success"] for r in rows], dtype=float)
    coll = np.array([r["collided"] for r in rows], dtype=float)
    aligned_arr = np.array([r["stage1_aligned"] for r in rows], dtype=float)
    depth = np.array([r["insert_depth"] for r in rows], dtype=float)
    tilt = np.array([r["insert_tilt"] for r in rows], dtype=float)
    steps = np.array([r["steps"] for r in rows], dtype=float)
    gfr = np.array([r["grasp_failure_rate"] for r in rows], dtype=float)

    print(f"\nN={n}")
    print(f"stage1_aligned (handed off to Task 2): {aligned_arr.mean():.3f}")
    print(f"overall success: {succ.mean():.3f}")
    print(f"collision_rate: {coll.mean():.3f}")
    print(f"mean_steps: {steps.mean():.1f}")
    print(f"insert_depth_mean: {depth.mean():.5f}")
    print(f"insert_tilt_deg_mean: {np.nanmean(tilt):.2f}")
    print(f"grasp_failure_rate (stage 2 only): {gfr.mean():.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["n", "stage1_aligned", "success_rate", "collision_rate", "mean_steps",
              "insert_depth_mean", "insert_tilt_deg_mean", "grasp_failure_rate"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(n=n, stage1_aligned=aligned_arr.mean(),
                        success_rate=succ.mean(), collision_rate=coll.mean(),
                        mean_steps=steps.mean(), insert_depth_mean=depth.mean(),
                        insert_tilt_deg_mean=np.nanmean(tilt),
                        grasp_failure_rate=gfr.mean()))
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
