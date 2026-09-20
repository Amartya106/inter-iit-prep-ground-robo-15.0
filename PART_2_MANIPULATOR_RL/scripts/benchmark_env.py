"""
benchmark_env.py -- throughput of the rebuilt env, and proof the persistent
scene actually bought something.

    .venv/bin/python scripts/benchmark_env.py

Reports:
  * env.step() rate (single env, random actions)
  * env.reset() rate  -- persistent scene (teleport bodies)
  * env.reset() rate  -- naive baseline (p.resetSimulation + reload URDFs),
    the scaffold's approach, for comparison
  * SubprocVecEnv aggregate step rate at a few worker counts
"""

import time

import numpy as np
import pybullet as p
import pybullet_data

from manipularl.env import ManipulaRLEnv


def time_steps(phase=4, n=2000):
    env = ManipulaRLEnv(phase=phase, split="train")
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(n):
        a = np.clip(a + rng.normal(0, 0.3, size=a.shape), -1, 1)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    env.close()
    return n / dt


def time_reset_persistent(phase=4, n=200):
    env = ManipulaRLEnv(phase=phase, split="train")
    env.reset(seed=0)
    t0 = time.perf_counter()
    for i in range(n):
        env.reset(options={"episode_index": i})
    dt = time.perf_counter() - t0
    env.close()
    return n / dt


def time_reset_naive(n=200):
    """Mimic the scaffold: full resetSimulation + reload plane/table/robot each episode."""
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    t0 = time.perf_counter()
    for _ in range(n):
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -9.81, physicsClientId=c)
        p.loadURDF("plane.urdf", physicsClientId=c)
        p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], physicsClientId=c)
        p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True, physicsClientId=c)
        for _ in range(20):
            p.stepSimulation(physicsClientId=c)
    dt = time.perf_counter() - t0
    p.disconnect(c)
    return n / dt


def time_vec(phase=4, workers=(4, 8), steps=400):
    from manipularl.make_env import make_vec_env

    out = {}
    for w in workers:
        venv = make_vec_env(phase, n_envs=w, split="train", seed=0, training=True, subproc=True)
        venv.reset()
        acts = np.zeros((w,) + venv.action_space.shape, dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(steps):
            venv.step(acts)
        dt = time.perf_counter() - t0
        out[w] = w * steps / dt
        venv.close()
    return out


if __name__ == "__main__":
    print("=== ManipulaRL env throughput ===")
    sr = time_steps()
    print(f"single-env step rate       : {sr:8.1f} steps/s")
    rp = time_reset_persistent()
    rn = time_reset_naive()
    print(f"reset rate (persistent)    : {rp:8.1f} resets/s")
    print(f"reset rate (naive reload)  : {rn:8.1f} resets/s   -> persistent is {rp/rn:.1f}x faster")
    for w, rate in time_vec().items():
        print(f"SubprocVecEnv x{w:<2d} step rate : {rate:8.1f} steps/s")
