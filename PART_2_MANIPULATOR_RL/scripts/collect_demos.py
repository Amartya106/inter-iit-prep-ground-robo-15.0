#!/usr/bin/env python3
"""
collect_demos.py -- roll the scripted IK expert (scripted_expert.py) through
the SAME wrapper stack the policy trains on (frame-stack 3 + VecNormalize) and
save the (obs, action) pairs from SUCCESSFUL episodes only, for behaviour
cloning.

    python scripts/collect_demos.py --phase 2 --episodes 400 --out runs/demos_phase2
    python scripts/collect_demos.py --phase 3 --episodes 800 --out runs/demos_phase3

Writes  <out>/demos.npz  (obs [N,D], act [N,8])  and  <out>/vecnormalize.pkl
(the normaliser stats, to hand to `train.py --warm-start-norm` for finetune).
"""

import argparse
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from manipularl.env import ManipulaRLEnv
from manipularl.configs import TRAIN_SEED_LO, get_phase
from scripts.scripted_expert import ScriptedManipulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--seed", type=int, default=TRAIN_SEED_LO)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--start-aligned", action="store_true",
                    help="peg_in_hole only: force every episode to start "
                         "pre-aligned above the hole mouth (env's own "
                         "align_curriculum=1.0), bypassing the reach+grasp+"
                         "carry legs entirely -- for collecting demos of "
                         "just the align->insert segment, which is the "
                         "actual missing skill (see Phase-4 diagnostics).")
    ap.add_argument("--min-depth-keep", type=float, default=0.0,
                    help="also keep episodes that reach at least this much "
                         "real insertion depth (info['max_depth']), even if "
                         "not a full held success -- the scripted expert's "
                         "insertion is bouncy (rim contact in the ~3mm bore) "
                         "so real partial progress is still useful "
                         "demonstration signal. 0.0 = success-only (default, "
                         "unchanged behaviour for phase 2/3).")
    args = ap.parse_args()
    out_default = f"runs/demos_phase{args.phase}"
    max_steps_default = get_phase(args.phase).max_steps
    args.out = args.out or out_default
    args.max_steps = args.max_steps or max_steps_default

    base = ManipulaRLEnv(phase=args.phase, split="train", seed=args.seed, grasp_curriculum=0.0,
                         align_curriculum=(1.0 if args.start_aligned else None))
    expert = ScriptedManipulator(base, phase=args.phase)
    venv = DummyVecEnv([lambda: base])
    if args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, training=True)

    def reset_expert():
        expert.reset()
        if args.start_aligned:
            # Skip APPROACH/DESCEND/GRASP entirely -- the peg is already
            # held (align_curriculum=1.0 teleported it above the hole mouth
            # in env.reset()). Starting the expert's state machine at
            # APPROACH would have it target peg+[0,0,0.10] as if reaching
            # for an ungrasped peg, moving the wrong direction. Also set
            # the orientation-control state manually (normally initialised
            # at the GRASP->LIFT transition -- see E58's compliant-expert
            # additions) so it isn't left unset:
            #   - _grip_quat_target/_grip_quat: same as the real transition
            #   - _reorient_t = 1.0 (already converged, not 0.0) -- the
            #     teleport start IS already vertical (env._start_pre_aligned
            #     places it with quat=identity), so there's genuinely
            #     nothing to gradually correct here. Leaving this at 0.0
            #     (reset()'s default) would permanently block ALIGN's own
            #     transition gate (`reorient_progress() >= 1.0`), since
            #     nothing in this bypass path ever advances it -- a real
            #     regression this fix specifically avoids.
            expert.state = "ALIGN"
            ee_quat = np.asarray(base._state_dict()["ee_quat"], float)
            expert._grip_quat_target = np.asarray(
                p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
            expert._grip_quat_start = ee_quat
            expert._grip_quat = ee_quat
            expert._reorient_t = 1.0

    obs = venv.reset()
    reset_expert()
    cur_obs, cur_act = [], []
    kept_obs, kept_act = [], []
    n_done = n_success = 0
    steps_in_ep = 0

    while n_done < args.episodes:
        s = base._state_dict()
        a = expert.act(s).reshape(1, -1)
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
            reset_expert()
            if n_done % 50 == 0:
                print(f"  {n_done}/{args.episodes} episodes, {n_success} success, "
                      f"{len(kept_obs)} transitions kept")

    print(f"\n[collect] scripted expert {n_success}/{n_done} = {100*n_success/n_done:.0f}% success")
    if not kept_obs:
        # Pre-existing crash risk (unrelated to any expert-behaviour change):
        # a small --episodes count against a genuinely low-success-rate task
        # (Phase 4's ~2% ceiling, see EXPERIMENTS.md E55) can legitimately
        # keep ZERO transitions -- np.asarray([]).shape[1] used to crash with
        # a confusing IndexError instead of reporting this plainly.
        print(f"[collect] kept 0 transitions -- nothing to save. "
              f"Try more --episodes or --min-depth-keep for a harder task.")
        venv.close()
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    O = np.asarray(kept_obs, np.float32); A = np.asarray(kept_act, np.float32)
    np.savez_compressed(out / "demos.npz", obs=O, act=A)
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"[collect] kept {O.shape[0]} transitions (obs dim {O.shape[1]}) -> {out}/demos.npz")
    venv.close()


if __name__ == "__main__":
    main()
