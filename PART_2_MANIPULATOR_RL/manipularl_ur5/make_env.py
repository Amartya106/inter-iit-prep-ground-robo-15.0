"""
make_env.py -- vectorised env construction shared by train.py and evaluate.py.

Wrapper stack (bottom to top):
    ManipulaRLEnv
      -> SubprocVecEnv / DummyVecEnv
      -> VecMonitor          (per-episode return/length + our info fields)
      -> VecFrameStack        (short observation history for noise filtering)
      -> VecNormalize         (obs only; reward left raw for off-policy TQC)

VecNormalize is the OUTERMOST wrapper on purpose: SB3 stores
`get_original_obs()` (pre-normalisation) in the replay buffer, and that must
already be frame-stacked to match the buffer's shape -- so frame-stacking
happens *below* normalisation.

Training builds it with `training=True`; evaluation loads the frozen
VecNormalize statistics and sets `training=False`.
"""

from typing import Optional

from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecNormalize,
)

from .env import ManipulaRLEnv

DEFAULT_FRAME_STACK = 3


def _thunk(phase: int, split: str, seed: int, rank: int, grasp_curriculum, cfg_overrides,
          insert_xy_jitter=None, insert_tilt_jitter_deg=None, linger_pen_w=None,
          settle_bonus_w=None, arm_jitter_pen_w=None, replay_path=None, replay_curriculum=None,
          keypoint_w=None):
    def _init():
        return ManipulaRLEnv(phase=phase, split=split, seed=seed + rank,
                             grasp_curriculum=grasp_curriculum, cfg_overrides=cfg_overrides,
                             insert_xy_jitter=insert_xy_jitter,
                             insert_tilt_jitter_deg=insert_tilt_jitter_deg,
                             linger_pen_w=linger_pen_w,
                             settle_bonus_w=settle_bonus_w,
                             arm_jitter_pen_w=arm_jitter_pen_w,
                             replay_path=replay_path,
                             replay_curriculum=replay_curriculum,
                             keypoint_w=keypoint_w)
    return _init


def make_vec_env(
    phase: int,
    n_envs: int = 4,
    split: str = "train",
    seed: int = 0,
    training: bool = True,
    norm_path: Optional[str] = None,
    n_stack: int = DEFAULT_FRAME_STACK,
    subproc: bool = True,
    grasp_curriculum: Optional[float] = None,
    cfg_overrides: Optional[dict] = None,
    insert_xy_jitter: Optional[float] = None,
    insert_tilt_jitter_deg: Optional[float] = None,
    linger_pen_w: Optional[float] = None,
    settle_bonus_w: Optional[float] = None,
    arm_jitter_pen_w: Optional[float] = None,
    replay_path: Optional[str] = None,
    replay_curriculum: Optional[float] = None,
    keypoint_w: Optional[float] = None,
):
    fns = [_thunk(phase, split, seed, i, grasp_curriculum, cfg_overrides,
                  insert_xy_jitter, insert_tilt_jitter_deg, linger_pen_w,
                  settle_bonus_w, arm_jitter_pen_w, replay_path, replay_curriculum,
                  keypoint_w)
          for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if (subproc and n_envs > 1) else DummyVecEnv(fns)
    venv = VecMonitor(venv)

    if n_stack and n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack)

    if norm_path is not None:
        venv = VecNormalize.load(norm_path, venv)
        venv.training = training
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=training)
    return venv
