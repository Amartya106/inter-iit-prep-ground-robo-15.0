#!/usr/bin/env python3
"""Phase 4 diagnostic: failure-mode breakdown + per-link collision
attribution (reads info["collision_links"], populated inside env.step()
itself -- safe even though collision is terminal, same fix as
scripts/diagnose_phase3.py). Used for E28, E30 and onward; argparse-ified
(previously positional-only, PPO-only, hardcoded phase=4) to compare
before/after checkpoints across the grasp-weld fix.

    python scripts/diagnose_phase4.py --run runs/phase4_utility/best
    python scripts/diagnose_phase4.py --run runs/phase4_utility/best --episodes 200 \
        --eval-seed 7 --out results/diagnose_phase4_utility_seed7.csv
    python scripts/diagnose_phase4.py --run runs/phase4_utility/best --insert-only
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from manipularl.env import ManipulaRLEnv
from manipularl.configs import EVAL_SEED_LO


def load_policy(run_dir, algo):
    run = Path(run_dir)
    mp = run / "model.zip"
    if not mp.exists():
        mp = run / "best" / "best_model.zip"
    Model = {"tqc": TQC, "sac": SAC}.get(algo, PPO)
    return Model.load(str(mp), device="cpu"), str(run / "vecnormalize.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "tqc", "sac"])
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="independent held-out episode set, see evaluate.py --eval-seed")
    ap.add_argument("--align-only", action="store_true")
    ap.add_argument("--insert-only", action="store_true")
    ap.add_argument("--descend-only", action="store_true")
    ap.add_argument("--cfg-json", default=None,
                    help='extra PhaseConfig cfg_overrides as JSON, e.g. \'{"rigid_grasp": true}\'')
    ap.add_argument("--out", default=None, help="optional per-episode CSV, results/-style")
    args = ap.parse_args()

    cfg_overrides = {}
    if args.align_only:
        cfg_overrides["align_only"] = True
    elif args.insert_only:
        cfg_overrides["insert_only"] = True
    elif args.descend_only:
        cfg_overrides["descend_only"] = True
    if args.cfg_json:
        cfg_overrides.update(json.loads(args.cfg_json))

    model, norm_path = load_policy(args.run, args.algo)
    # grasp_curriculum/align_curriculum are direct ManipulaRLEnv constructor
    # kwargs, NOT PhaseConfig dataclass fields (align_curriculum isn't a
    # PhaseConfig field at all -- dataclasses.replace would raise) -- eval
    # should always see the real, undiluted policy, same as the original
    # script's explicit 0.0/0.0.
    base = ManipulaRLEnv(phase=args.phase, split="eval", seed=EVAL_SEED_LO,
                         index_seed=args.eval_seed, grasp_curriculum=0.0,
                         align_curriculum=0.0,
                         cfg_overrides=(cfg_overrides or None))
    venv = DummyVecEnv([lambda: base])
    venv = VecFrameStack(venv, n_stack=args.frame_stack)
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False

    rows = []
    link_hits = {}
    n_collision_steps = 0

    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ever_grasped = False
        n_grasp_events = 0
        was_grasped = False
        collided = False
        collided_before_grasp = False
        t = 0
        info = {}
        while not done:
            s = base._state_dict()
            grasped = bool(s["grasped"])
            if grasped and not was_grasped:
                n_grasp_events += 1
            if grasped:
                ever_grasped = True
            was_grasped = grasped
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = venv.step(action)
            info = infos[0]
            done = bool(dones[0])
            t += 1
            if info.get("collision", False):
                n_collision_steps += 1
                collided = True
                if not ever_grasped:
                    collided_before_grasp = True
                for name in info.get("collision_links", []):
                    link_hits[name] = link_hits.get(name, 0) + 1

        rows.append(dict(ever_grasped=ever_grasped, n_grasp_events=n_grasp_events,
                          collided=collided, collided_before_grasp=collided_before_grasp,
                          min_xy_err=info.get("min_xy_err", np.inf),
                          min_tilt_deg=info.get("min_tilt_deg", np.inf),
                          max_depth=info.get("max_depth", 0.0),
                          success=bool(info.get("success", False)), steps=t))

    venv.close()

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[diagnose_phase4] wrote {args.out}")

    n = len(rows)
    arr = np.array([(float(r["ever_grasped"]), float(r["n_grasp_events"]), float(r["collided"]),
                     float(r["collided_before_grasp"]), r["min_xy_err"], r["min_tilt_deg"],
                     r["max_depth"], float(r["success"]), r["steps"])
                    for r in rows])
    print(f"N={n}  run={args.run}  eval_seed={args.eval_seed}")
    print(f"success: {arr[:,7].mean():.3f}")
    print(f"ever_grasped: {arr[:,0].mean():.3f}")
    print(f"mean grasp events/episode (>1 = flicker): {arr[:,1].mean():.2f}")
    print(f"frac episodes with >5 grasp events (severe flicker): {(arr[:,1] > 5).mean():.2f}")
    print(f"collided (any point in episode): {arr[:,2].mean():.2f}")
    print(f"collided BEFORE ever grasping: {arr[:,3].mean():.2f}")
    print(f"mean steps: {arr[:,8].mean():.1f}")

    grasped_mask = arr[:, 0] > 0.5
    print(f"\nof grasped episodes ({int(grasped_mask.sum())}/{n}):")
    if grasped_mask.sum() > 0:
        g = arr[grasped_mask]
        tilt = g[:, 5][np.isfinite(g[:, 5])]
        print(f"  min_xy_err_to_hole: mean={g[:,4].mean()*100:.1f}cm  median={np.median(g[:,4])*100:.1f}cm  best={g[:,4].min()*100:.2f}cm")
        print(f"  frac with xy_err < 3.5cm: {(g[:,4] < 0.035).mean():.2f}")
        print(f"  frac with xy_err < 10cm: {(g[:,4] < 0.10).mean():.2f}")
        if tilt.size:
            print(f"  min_tilt_deg: mean={tilt.mean():.1f}  median={np.median(tilt):.1f}  best={tilt.min():.1f}")
        print(f"  max_depth: mean={g[:,6].mean()*100:.2f}cm  best={g[:,6].max()*100:.2f}cm")
        print(f"  collided (of grasped eps): {g[:,2].mean():.2f}")

    print(f"\ncollision-steps total: {n_collision_steps}")
    if n_collision_steps:
        print("which part of the arm/peg was in contact (a step can count toward >1 if multi-contact):")
        for name, cnt in sorted(link_hits.items(), key=lambda kv: -kv[1]):
            print(f"  {name:22s} {cnt:5d}  ({100*cnt/n_collision_steps:.1f}% of collision-steps)")


if __name__ == "__main__":
    main()
