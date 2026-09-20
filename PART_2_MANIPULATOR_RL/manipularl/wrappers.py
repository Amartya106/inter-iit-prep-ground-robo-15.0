"""
wrappers.py -- EVALUATION-TIME robustness perturbations (Phase 6).

The PS is explicit: observation noise, action noise, and peg/environment
perturbations are introduced "at evaluation time only (not necessarily seen
during training)". Keeping them in wrappers -- never in the env -- makes that
guarantee structural: training scripts build the bare env, evaluation builds
`make_eval_env(...)` with whatever perturbation magnitudes the sweep needs.

None of these are active unless a non-zero magnitude is passed.
"""

import numpy as np
import gymnasium as gym


class NoisyObservation(gym.ObservationWrapper):
    """Additive Gaussian noise on the observation vector."""

    def __init__(self, env, sigma: float = 0.0):
        super().__init__(env)
        self.sigma = float(sigma)

    def observation(self, obs):
        if self.sigma <= 0.0:
            return obs
        return (obs + self.np_random.normal(0.0, self.sigma, size=obs.shape)).astype(obs.dtype)


class NoisyAction(gym.ActionWrapper):
    """Additive Gaussian noise on the action, then re-clip to [-1, 1]."""

    def __init__(self, env, sigma: float = 0.0):
        super().__init__(env)
        self.sigma = float(sigma)

    def action(self, action):
        if self.sigma <= 0.0:
            return action
        a = np.asarray(action, dtype=np.float32)
        return np.clip(a + self.np_random.normal(0.0, self.sigma, size=a.shape), -1.0, 1.0)


class PegPerturbation(gym.Wrapper):
    """Random small impulses applied to the peg every `period` steps."""

    def __init__(self, env, force: float = 0.0, period: int = 25):
        super().__init__(env)
        self.force = float(force)
        self.period = int(period)
        self._k = 0

    def step(self, action):
        out = self.env.step(action)
        self._k += 1
        if self.force > 0.0 and self._k % self.period == 0:
            base = self.env.unwrapped
            try:
                import pybullet as p

                peg = base.peg_id
                if base.cfg.use_peg:
                    d = self.np_random.normal(0.0, 1.0, size=3)
                    d[2] = abs(d[2]) * 0.2
                    d = d / (np.linalg.norm(d) + 1e-9) * self.force
                    pos = p.getBasePositionAndOrientation(peg, physicsClientId=base.client)[0]
                    p.applyExternalForce(peg, -1, d.tolist(), list(pos), p.WORLD_FRAME,
                                         physicsClientId=base.client)
            except Exception:
                pass
        return out

    def reset(self, **kwargs):
        self._k = 0
        return self.env.reset(**kwargs)


def make_eval_env(phase: int, split: str = "eval", obs_noise: float = 0.0,
                  act_noise: float = 0.0, perturb_force: float = 0.0,
                  render_mode: str = "none", seed=None):
    """Bare env + only the perturbation wrappers whose magnitude is non-zero."""
    from .env import ManipulaRLEnv

    env = ManipulaRLEnv(phase=phase, split=split, render_mode=render_mode, seed=seed)
    if obs_noise > 0.0:
        env = NoisyObservation(env, obs_noise)
    if act_noise > 0.0:
        env = NoisyAction(env, act_noise)
    if perturb_force > 0.0:
        env = PegPerturbation(env, perturb_force)
    return env
