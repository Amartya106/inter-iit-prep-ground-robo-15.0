"""
collect_multi_expert_demos.py -- policy distillation, source-collection step.

Unlike every prior imitation-learning attempt so far (Phase 2 BC, Phase
3 E18 BC-finetune, Task 1 GAIL/E46 -- all cloning ONE narrow scripted or
single-RL source and all losing to reward-shaped RL), this collects
demonstrations from THREE complementary RL-TRAINED SPECIALISTS, each good at
a different part of the task:
  - E41 (runs/phase4_task1_align_v6, align_only): cleanest reach+grasp+align
    behavior (collision 0.30, 65% ever-grasped, 64% clean single-grasp).
  - E33 (runs/phase4_task2_insert, insert_only): cleanest fine-insertion
    threading (mean depth 0.32cm, best-case 6.24cm from an aligned start).
  - E51 (runs/phase4_full_dream, full monolithic task): best COMPLETE
    cold-start-to-finish coverage (74% ever-grasped, best-case depth
    7.89cm, the standing-best full-task policy).

Each specialist is rolled out under ITS OWN cfg_overrides and ITS OWN frozen
VecNormalize (so it produces sensible, in-distribution actions) -- but the
RAW (pre-normalization) observation is what gets saved, via VecNormalize's
own get_original_obs(). This is essential: the three specialists were each
fit to their OWN observation scale, so mixing their NORMALIZED obs directly
would be comparing apples to oranges. Saving raw obs lets one shared,
freshly-fit VecNormalize be built afterward (see the final block below,
mirroring scripts/collect_demos.py's own live-fit-during-collection pattern,
just done as an explicit RunningMeanStd.update() over the combined array
instead).

    PYTHONPATH=. .venv/bin/python scripts/collect_multi_expert_demos.py \
        --out runs/multi_expert_demos
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd

from manipularl.env import ManipulaRLEnv
from manipularl.configs import TRAIN_SEED_LO


SPECIALISTS = [
    dict(name="E41-align", run="runs/phase4_task1_align_v6",
         cfg_overrides={"align_only": True}, episodes=80),
    dict(name="E33-insert", run="runs/phase4_task2_insert",
         cfg_overrides={"insert_only": True}, episodes=80),
    dict(name="E51-full", run="runs/phase4_full_dream",
         cfg_overrides=None, episodes=150),
]


def collect_one(spec, frame_stack, seed_base):
    model = PPO.load(f"{spec['run']}/model.zip", device="cpu")
    base = ManipulaRLEnv(phase=4, split="train", seed=seed_base, cfg_overrides=spec["cfg_overrides"])
    venv = DummyVecEnv([lambda: base])
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    venv = VecNormalize.load(f"{spec['run']}/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False

    raw_obs_buf, act_buf = [], []
    total_steps = 0
    for ep in range(spec["episodes"]):
        obs = venv.reset()
        done = False
        while not done:
            raw_obs_buf.append(venv.get_original_obs()[0].copy())
            action, _ = model.predict(obs, deterministic=True)
            act_buf.append(action[0].copy())
            obs, _r, dones, infos = venv.step(action)
            done = bool(dones[0])
            total_steps += 1
        if ep % 25 == 0:
            print(f"  [{spec['name']}] ep {ep}/{spec['episodes']}, {total_steps} steps so far")
    venv.close()
    print(f"[{spec['name']}] collected {total_steps} (raw_obs, action) pairs")
    return np.stack(raw_obs_buf).astype(np.float32), np.stack(act_buf).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--out", default="runs/multi_expert_demos")
    args = ap.parse_args()

    all_obs, all_act, all_source = [], [], []
    for spec in SPECIALISTS:
        o, a = collect_one(spec, args.frame_stack, TRAIN_SEED_LO + hash(spec["name"]) % 10_000)
        all_obs.append(o); all_act.append(a)
        all_source.append(np.full(len(o), SPECIALISTS.index(spec), dtype=np.int32))

    obs = np.concatenate(all_obs, axis=0)
    act = np.concatenate(all_act, axis=0)
    source = np.concatenate(all_source, axis=0)
    print(f"\n[collect] combined: {obs.shape[0]} transitions, obs_dim={obs.shape[1]}, act_dim={act.shape[1]}")
    for i, spec in enumerate(SPECIALISTS):
        print(f"  {spec['name']}: {int((source == i).sum())} transitions "
              f"({100 * (source == i).mean():.1f}%)")

    # One shared VecNormalize, fit on the COMBINED raw observations (not any
    # one specialist's own scale) -- this is what the BC-trained policy will
    # actually see, and what the subsequent RL fine-tune stage should load
    # via --warm-start-norm so the input scale stays consistent end to end.
    #
    # Fit on a ROBUST (percentile-clipped) copy of the raw obs, not the raw
    # values directly -- found via a real bug: the ft (wrist force/torque)
    # dims spike to huge, essentially unbounded values during collision-
    # impact frames (E33/E51 are both collision-prone specialists), and a
    # handful of such frames dominated the variance estimate for those 3
    # dims (up to 3.5M, ~200x phase3_ppo_best's own max of ~17k) -- which
    # crushed the REAL, everyday force signal to near-zero after
    # normalization for every other step, effectively blinding the policy
    # to contact feedback. Winsorizing to the [0.5, 99.5] percentile range
    # per-dimension before fitting stats is standard practice for exactly
    # this failure mode -- outliers still exist in the actual saved demo
    # actions/behavior (nothing about the recorded actions changes), only
    # the NORMALIZATION SCALE is desensitized to a few extreme spikes.
    lo = np.percentile(obs, 0.5, axis=0)
    hi = np.percentile(obs, 99.5, axis=0)
    obs_clipped_for_stats = np.clip(obs, lo, hi)
    obs_rms = RunningMeanStd(shape=(obs.shape[1],))
    obs_rms.update(obs_clipped_for_stats)
    print(f"[collect] robust-clipped stats fit -- max var before: {obs.var(axis=0).max():.1f}, "
          f"after percentile-clip: {obs_clipped_for_stats.var(axis=0).max():.1f}")
    dummy_base = ManipulaRLEnv(phase=4, split="train", seed=0)
    dummy_venv = DummyVecEnv([lambda: dummy_base])
    if args.frame_stack > 1:
        dummy_venv = VecFrameStack(dummy_venv, n_stack=args.frame_stack)
    shared_norm = VecNormalize(dummy_venv, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, training=False)
    shared_norm.obs_rms = obs_rms
    normalized_obs = shared_norm.normalize_obs(obs).astype(np.float32)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "demos.npz", obs=normalized_obs, act=act, source=source)
    shared_norm.save(str(out / "vecnormalize.pkl"))
    print(f"[collect] saved {out}/demos.npz (normalized obs) and {out}/vecnormalize.pkl")


if __name__ == "__main__":
    main()
