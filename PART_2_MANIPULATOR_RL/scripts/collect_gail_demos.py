"""
collect_gail_demos.py -- harvest E24's own successful Phase-3 rollouts as
GAIL expert demonstrations for Task 1 ("align").

Since the observation/action interface is IDENTICAL across all six phases
by design (manipularl/env.py's own docstring), E24's clean, non-flickering
reach+grasp+carry behavior on Phase 3 (0.765 success, zero grasp-flicker)
can be used directly as expert data for Task 1's GAIL training on Phase 4
-- no need to run the weak scripted expert (whose full-task success was
only ~2%).

Uses the EXACT same wrapper stack (VecFrameStack + VecNormalize, E24's own
saved stats) that E24's policy and Task 1's PPO generator both expect, so
the discriminator sees observations in the same space the generator does.

    python scripts/collect_gail_demos.py --episodes 80 --out runs/gail_demos_e24

Writes <out>/trajectories.npz (a pickled list of imitation.data.types.Trajectory,
via numpy's allow_pickle) and prints a summary.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from imitation.data.types import Trajectory

from manipularl.env import ManipulaRLEnv
from manipularl.configs import TRAIN_SEED_LO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", default="runs/phase3_ppo_v7_terminal",
                    help="checkpoint to harvest demonstrations from (E24 by default)")
    ap.add_argument("--phase", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=100,
                    help="target number of SUCCESSFUL episodes to keep")
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--seed", type=int, default=TRAIN_SEED_LO)
    ap.add_argument("--out", default="runs/gail_demos_e24")
    args = ap.parse_args()

    run = Path(args.source_run)
    model = PPO.load(str(run / "model.zip"), device="cpu")
    base = ManipulaRLEnv(phase=args.phase, split="train", seed=args.seed)
    venv = DummyVecEnv([lambda: base])
    if args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack)
    venv = VecNormalize.load(str(run / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False

    trajectories = []
    n_success = 0
    attempt = 0
    while n_success < args.episodes and attempt < args.max_attempts:
        attempt += 1
        obs = venv.reset()
        obs_list = [np.asarray(obs[0], dtype=np.float32)]
        act_list = []
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, dones, infos = venv.step(action)
            done = bool(dones[0])
            info = infos[0]
            obs_list.append(np.asarray(obs[0], dtype=np.float32))
            act_list.append(np.asarray(action[0], dtype=np.float32))

        success = bool(info.get("success", False))
        if success:
            n_success += 1
            traj = Trajectory(
                obs=np.stack(obs_list, axis=0),
                acts=np.stack(act_list, axis=0),
                infos=None,
                terminal=True,
            )
            trajectories.append(traj)
            if n_success % 10 == 0:
                print(f"  {n_success}/{args.episodes} successful demos "
                      f"({attempt} attempts, {n_success/attempt:.0%} hit rate)")

    venv.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trajectories.pkl", "wb") as fh:
        pickle.dump(trajectories, fh)

    lengths = [len(t.acts) for t in trajectories]
    print(f"\n[collect] {len(trajectories)}/{args.episodes} demos collected "
          f"in {attempt} attempts ({len(trajectories)/attempt:.0%} hit rate)")
    print(f"[collect] episode length: mean={np.mean(lengths):.1f} "
          f"min={min(lengths)} max={max(lengths)}")
    print(f"[collect] total transitions: {sum(lengths)}")
    print(f"[collect] wrote {out}/trajectories.pkl")


if __name__ == "__main__":
    main()
