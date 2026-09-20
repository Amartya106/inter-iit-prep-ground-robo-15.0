"""
extract_replay_trajectory.py -- roll out E33's trained insert_only checkpoint
until a high-depth episode is found, and record every step's exact physical
state (joint angles, peg pose, hole position) so it can be teleported BACK to
exactly later, via env._start_from_replay(). No IK involved on replay -- these
are real joint angles from a real successful rollout, so there's zero
convergence risk the way there is with every other curriculum-teleport method
in this codebase.

    PYTHONPATH=. .venv/bin/python scripts/extract_replay_trajectory.py \
        --run runs/phase4_task2_insert --episodes 40 --out runs/replay_trajectory_e33.npz
"""
import argparse
import numpy as np
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from manipularl.env import ManipulaRLEnv, ARM_DOF, _EE_LINK
from manipularl.configs import EVAL_SEED_LO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase4_task2_insert")
    ap.add_argument("--episodes", type=int, default=40,
                    help="attempts to roll -- E33's own success rate is ~2.5%%, "
                         "so most attempts won't reach a high depth; keep the BEST one seen")
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--out", default="runs/replay_trajectory_e33.npz")
    args = ap.parse_args()

    model = PPO.load(f"{args.run}/model.zip", device="cpu")
    base = ManipulaRLEnv(phase=4, split="eval", seed=EVAL_SEED_LO,
                         cfg_overrides={"insert_only": True})
    venv = DummyVecEnv([lambda: base])
    venv = VecFrameStack(venv, n_stack=args.frame_stack)
    venv = VecNormalize.load(f"{args.run}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False

    best_max_depth = -1.0
    best_snapshots = None

    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        snapshots = []
        ep_max_depth = 0.0
        while not done:
            s = base.unwrapped._state_dict()
            q = np.array([p.getJointState(base.unwrapped.robot_id, j,
                                          physicsClientId=base.unwrapped.client)[0]
                         for j in range(ARM_DOF)])
            snapshots.append(dict(
                q=q.copy(),
                peg_pos=s["peg_pos"].copy(),
                peg_quat=s["peg_quat"].copy(),
                hole_xy=s["hole_xy"].copy(),
                depth=float(s["peg_depth"]),
            ))
            ep_max_depth = max(ep_max_depth, float(s["peg_depth"]))
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, dones, infos = venv.step(a)
            done = bool(dones[0])
        print(f"  ep {ep}: max_depth={ep_max_depth*100:.2f}cm steps={len(snapshots)} "
              f"success={infos[0].get('success', False)}")
        if ep_max_depth > best_max_depth:
            best_max_depth = ep_max_depth
            best_snapshots = snapshots

    venv.close()
    assert best_snapshots is not None, "no episodes rolled?"
    print(f"[extract] best episode: max_depth={best_max_depth*100:.2f}cm, "
          f"{len(best_snapshots)} steps")

    q = np.stack([s["q"] for s in best_snapshots])
    peg_pos = np.stack([s["peg_pos"] for s in best_snapshots])
    peg_quat = np.stack([s["peg_quat"] for s in best_snapshots])
    hole_xy = np.stack([s["hole_xy"] for s in best_snapshots])
    depth = np.array([s["depth"] for s in best_snapshots])

    np.savez(args.out, q=q, peg_pos=peg_pos, peg_quat=peg_quat,
             hole_xy=hole_xy, depth=depth, max_depth=best_max_depth)
    print(f"[extract] saved {len(best_snapshots)} snapshots -> {args.out}")


if __name__ == "__main__":
    main()
