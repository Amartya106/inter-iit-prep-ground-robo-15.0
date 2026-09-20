"""
train_gail_task1.py -- GAIL (adversarial imitation learning) for Task 1
("align"), using E24's own successful Phase-3 rollouts as expert
demonstrations (collect_gail_demos.py).

Unlike every hand-designed reward-shaping attempt so far (E31-E43),
this doesn't specify a reward function at all -- a discriminator learns to
distinguish expert (E24, non-flickering, stable grasp+carry) from learner
state-action pairs, and its output IS the reward signal PPO optimizes.
Trains via real on-policy rollouts (unlike BC, which lost to RL twice
already in this project -- Phase 2, Phase 3 E18), so the policy learns to
REACH the states an expert would occupy, not just mimic isolated actions.

    python scripts/train_gail_task1.py --demos runs/gail_demos_e24 \
        --out-dir runs/phase4_task1_gail --timesteps 1000000
"""

import argparse
import os
import pickle
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from stable_baselines3 import PPO
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

from manipularl.make_env import make_vec_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="runs/gail_demos_e24")
    ap.add_argument("--warm-start", default="runs/phase3_ppo_v7_terminal/model.zip")
    ap.add_argument("--warm-start-norm", default="runs/phase3_ppo_v7_terminal/vecnormalize.pkl")
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--demo-batch-size", type=int, default=1024)
    ap.add_argument("--disc-updates-per-round", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/phase4_task1_gail")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = str(out_dir / "tb")

    with open(Path(args.demos) / "trajectories.pkl", "rb") as fh:
        trajectories = pickle.load(fh)
    transitions = rollout.flatten_trajectories(trajectories)
    print(f"[gail] loaded {len(trajectories)} expert trajectories, "
          f"{len(transitions)} transitions")

    venv = make_vec_env(
        args.phase, n_envs=args.n_envs, split="train", seed=args.seed,
        training=True, n_stack=args.frame_stack, norm_path=args.warm_start_norm,
        subproc=True, cfg_overrides={"align_only": True},
    )

    model = PPO.load(args.warm_start, env=venv, tensorboard_log=tb_dir, seed=args.seed)
    print(f"[gail] warm-started generator from {args.warm_start}")

    reward_net = BasicRewardNet(venv.observation_space, venv.action_space,
                                normalize_input_layer=RunningNorm)

    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=args.demo_batch_size,
        venv=venv,
        gen_algo=model,
        reward_net=reward_net,
        n_disc_updates_per_round=args.disc_updates_per_round,
        allow_variable_horizon=True,
    )

    print(f"[gail] training {args.timesteps:,} generator timesteps "
          f"({args.n_envs} envs, demo_batch_size={args.demo_batch_size})")
    gail_trainer.train(total_timesteps=args.timesteps)

    model.save(str(out_dir / "model.zip"))
    venv.save(str(out_dir / "vecnormalize.pkl"))
    print(f"[gail] saved {out_dir}/model.zip and {out_dir}/vecnormalize.pkl "
          f"-- evaluable via the normal pipeline: "
          f"evaluate.py --run {out_dir} --phase 4 --algo ppo --align-only")
    venv.close()


if __name__ == "__main__":
    main()
