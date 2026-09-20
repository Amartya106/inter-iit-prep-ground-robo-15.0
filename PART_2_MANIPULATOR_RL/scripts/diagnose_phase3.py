#!/usr/bin/env python3
"""Reusable Phase 3 diagnostic: failure-mode breakdown + per-link collision
attribution + carry-vs-free obstacle clearance.

    python diagnose_phase3.py <run_dir> [n_episodes] [algo]   # algo: ppo|sac|tqc, default ppo

Per-link attribution reads `info["collision_links"]`, populated inside
env.step() itself (manipularl/env.py's _obstacle_collision_links()) rather
than re-querying pybullet after venv.step() returns -- for a terminal
collision (collision_is_failure runs), DummyVecEnv auto-resets the sim to
the next episode inside that same step() call, so an external contact-point
query after the fact would see the wrong (already-reset) world.
"""
import sys
sys.path.insert(0, "/storage/Ground-Robo-Prepathon/submission/PART_2_MANIPULATOR_RL")
import numpy as np
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from manipularl.env import ManipulaRLEnv
from manipularl.configs import get_phase, EVAL_SEED_LO

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/phase3_ppo_v4_gradedpenalty"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
ALGO = sys.argv[3] if len(sys.argv) > 3 else "ppo"

ModelCls = {"ppo": PPO, "sac": SAC, "tqc": TQC}[ALGO]
model = ModelCls.load(f"{RUN}/model.zip", device="cpu")
base = ManipulaRLEnv(phase=3, split="eval", seed=EVAL_SEED_LO, grasp_curriculum=0.0)
venv = DummyVecEnv([lambda: base])
venv = VecFrameStack(venv, n_stack=3)
venv = VecNormalize.load(f"{RUN}/vecnormalize.pkl", venv)
venv.training = False; venv.norm_reward = False

place_tol = get_phase(3).place_success_dist
rows = []
link_hits = {}   # link name -> count of collision-steps it appeared in
n_collision_steps = 0

for ep in range(N):
    obs = venv.reset()
    done = False
    grasp_formed = False
    t_first_grasp = -1
    reached_target = False
    released_after_reach = False
    collided_ever = False
    min_obs_dist = np.inf
    obs_dist_free_sum, obs_dist_free_n = 0.0, 0
    obs_dist_carry_sum, obs_dist_carry_n = 0.0, 0
    t = 0
    last_info = {}
    while not done:
        s = base._state_dict()
        peg, goal = s["peg_pos"], s["goal_pos"]
        grasped = bool(s["grasped"])
        d_obs = float(s.get("min_obstacle_dist", np.inf))
        min_obs_dist = min(min_obs_dist, d_obs)
        if np.isfinite(d_obs):
            if grasped:
                obs_dist_carry_sum += d_obs; obs_dist_carry_n += 1
            else:
                obs_dist_free_sum += d_obs; obs_dist_free_n += 1
        if grasped and not grasp_formed:
            grasp_formed, t_first_grasp = True, t
        if grasped and np.linalg.norm(peg - goal) < place_tol:
            reached_target = True
        elif reached_target and not grasped:
            released_after_reach = True

        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, infos = venv.step(action)
        done = bool(dones[0]); last_info = infos[0]; t += 1

        if last_info.get("collision", False):
            n_collision_steps += 1
            collided_ever = True
            # read the links straight from env.step()'s own info dict --
            # populated BEFORE any VecEnv auto-reset, so it reflects the
            # actual collision, not (for a terminal collision) the world
            # state of whatever episode got auto-reset into right after.
            for name in last_info.get("collision_links", []):
                link_hits[name] = link_hits.get(name, 0) + 1

    success = bool(last_info.get("success", False))
    rows.append(dict(success=success, grasp_formed=grasp_formed, t_first_grasp=t_first_grasp,
                      reached_target=reached_target, released_after_reach=released_after_reach,
                      collided=collided_ever, min_obs_dist=min_obs_dist, steps=t,
                      obs_dist_free=(obs_dist_free_sum/obs_dist_free_n if obs_dist_free_n else np.nan),
                      obs_dist_carry=(obs_dist_carry_sum/obs_dist_carry_n if obs_dist_carry_n else np.nan)))

venv.close()

n = len(rows)
succ = sum(r["success"] for r in rows)
print(f"=== {RUN} ===")
print(f"N={n}  success={succ/n:.2f}")
fails = [r for r in rows if not r["success"]]
print(f"failures: {len(fails)}")

def frac(pred, pool):
    return sum(1 for r in pool if pred(r)) / max(1, len(pool))

print(f"  of failures: collided                     {frac(lambda r: r['collided'], fails):.2f}")
print(f"  of failures: never grasped                 {frac(lambda r: not r['grasp_formed'], fails):.2f}")
print(f"  of failures: grasped, never reached target {frac(lambda r: r['grasp_formed'] and not r['reached_target'], fails):.2f}")
print(f"  of failures: reached target, never released{frac(lambda r: r['reached_target'] and not r['released_after_reach'], fails):.2f}")
print(f"  of failures: collided AND never grasped    {frac(lambda r: r['collided'] and not r['grasp_formed'], fails):.2f}")
print(f"  of failures: collided AFTER grasping       {frac(lambda r: r['collided'] and r['grasp_formed'], fails):.2f}")

steps_all = np.array([r["steps"] for r in rows])
steps_fail = np.array([r["steps"] for r in fails]) if fails else np.array([0.])
steps_succ = np.array([r["steps"] for r in rows if r["success"]])
print(f"mean steps: all={steps_all.mean():.1f}  success={steps_succ.mean():.1f}  fail={steps_fail.mean():.1f}")

t_grasp = np.array([r["t_first_grasp"] for r in rows if r["grasp_formed"]])
print(f"grasp formed: {len(t_grasp)}/{n} ({len(t_grasp)/n:.2f}), mean t_first_grasp={t_grasp.mean():.1f}")

free_d = np.array([r["obs_dist_free"] for r in rows if np.isfinite(r["obs_dist_free"])])
carry_d = np.array([r["obs_dist_carry"] for r in rows if np.isfinite(r["obs_dist_carry"])])
print(f"mean obstacle clearance kept -- empty-handed: {free_d.mean()*100:.1f}cm (n={len(free_d)})"
      f"   carrying: {carry_d.mean()*100:.1f}cm (n={len(carry_d)})")

print(f"\ncollision-steps total: {n_collision_steps}")
if n_collision_steps:
    print("which part of the arm/peg was in contact (a step can count toward >1 if multi-contact):")
    for name, cnt in sorted(link_hits.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {cnt:5d}  ({100*cnt/n_collision_steps:.1f}% of collision-steps)")
