"""
collect_dynamics_transitions.py -- Part 2a of the "dreaming"/model-based RL
plan. Rolls descend_only episodes (mixing the airspace-height curriculum and
the replay-seeded curriculum, at their real training-time fractions) under
RANDOM actions and records (obs, action, next_obs, reward, done) transitions
to disk, to train a dynamics model on (Part 2b). Random-action exploration is
standard practice for this -- the model just needs to learn state-transition
dynamics, not expert behavior, and a random policy gives broad, unbiased
state/action coverage (no trained descend_only policy is good enough yet to
use instead, and using one would bias coverage toward whatever that policy's
current habits are).

Uses the RAW _assemble_obs() output (pre-VecNormalize), since normalization
stats aren't stable this early in any of the parallel training runs -- the
dynamics model is trained and used consistently in this same raw space.

    PYTHONPATH=. .venv/bin/python scripts/collect_dynamics_transitions.py \
        --episodes 150 --out runs/dynamics_transitions.npz
"""
import argparse
import numpy as np

from manipularl.env import ManipulaRLEnv
from manipularl.configs import TRAIN_SEED_LO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--replay-path", default="runs/replay_trajectory_e33.npz")
    ap.add_argument("--replay-curriculum", type=float, default=0.3)
    ap.add_argument("--airspace-height-frac", type=float, default=1.0,
                    help="1.0 = full real range, matches the near-converged "
                         "training curricula's end state")
    ap.add_argument("--out", default="runs/dynamics_transitions.npz")
    ap.add_argument("--full-phase4", action="store_true",
                    help="collect from the REAL monolithic phase=4 task "
                         "(reach->grasp->transit->align->insert, cold-start "
                         "each episode, max_steps=400) instead of descend_only "
                         "-- no replay/airspace curriculum applies here, "
                         "those are descend_only-specific mechanisms")
    args = ap.parse_args()

    # Deliberately RAW single-frame observations (no VecFrameStack/VecNormalize
    # wrappers) -- the actual PPO rollout buffer stores frame-stacked+
    # normalized obs, a different representation this model does NOT target.
    # DreamAugmentCallback (callbacks.py) is responsible for reconstructing
    # the 3-frame stack and applying live VecNormalize.normalize_obs() to
    # convert this model's raw single-frame predictions into the buffer's
    # actual representation -- keeping that logic in one place (the callback)
    # rather than duplicating it here.
    if args.full_phase4:
        env = ManipulaRLEnv(phase=4, split="train", seed=42)
    else:
        env = ManipulaRLEnv(phase=4, split="train", seed=42,
                            cfg_overrides={"descend_only": True, "max_steps": 150},
                            replay_path=args.replay_path,
                            replay_curriculum=args.replay_curriculum)
        env.set_airspace_height_frac(args.airspace_height_frac)

    obs_buf, act_buf, next_obs_buf, rew_buf, done_buf = [], [], [], [], []
    total_steps = 0
    for ep in range(args.episodes):
        obs, info = env.reset(options={"episode_index": TRAIN_SEED_LO + ep})
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs_buf.append(obs.copy())
            act_buf.append(action.copy())
            next_obs_buf.append(next_obs.copy())
            rew_buf.append(float(reward))
            done_buf.append(bool(info.get("episode_collided", False)))
            obs = next_obs
            steps += 1
        total_steps += steps
        if ep % 25 == 0:
            print(f"  ep {ep}: {steps} steps, total so far {total_steps}")

    obs_arr = np.stack(obs_buf).astype(np.float32)
    act_arr = np.stack(act_buf).astype(np.float32)
    next_obs_arr = np.stack(next_obs_buf).astype(np.float32)
    rew_arr = np.array(rew_buf, dtype=np.float32)
    done_arr = np.array(done_buf, dtype=np.float32)

    np.savez(args.out, obs=obs_arr, action=act_arr, next_obs=next_obs_arr,
             reward=rew_arr, done=done_arr)
    print(f"[collect] {total_steps} transitions from {args.episodes} episodes -> {args.out}")
    print(f"[collect] obs_dim={obs_arr.shape[1]} action_dim={act_arr.shape[1]} "
          f"done_rate={done_arr.mean():.3f}")


if __name__ == "__main__":
    main()
