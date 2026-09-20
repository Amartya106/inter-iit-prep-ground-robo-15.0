"""
configs.py (UR5 port) -- the only genuinely robot-specific constants (arm DOF count and the
derived action dimension). Everything else -- PhaseConfig, the six phase presets, control/
physics rates, obstacle-observation width, the delta-joint scale, and the train/eval seed
partition -- is TASK-level, not robot-level, and is re-exported unchanged from
`manipularl.configs` (see the plan this was built from: "keep the methods for solving the
problems the same"). This means `manipularl.rewards.RewardComputer` and
`manipularl.randomization.EpisodeSampler` (both imported directly, unmodified, into
`manipularl_ur5/env.py`) see the exact same `PhaseConfig` class this module hands them --
no duplicated logic, no type mismatch.
"""

ARM_DOF = 6                       # UR5: shoulder_pan/lift, elbow, wrist_1/2/3
ACTION_DIM = ARM_DOF + 1          # 6 delta-joint + 1 gripper

from manipularl.configs import (  # noqa: E402 -- re-exported unchanged, see module docstring
    PhaseConfig, PHASES, get_phase,
    CONTROL_HZ, PHYSICS_HZ, SUBSTEPS, MAX_OBSTACLES_IN_OBS, DELTA_Q_SCALE,
    TRAIN_SEED_LO, TRAIN_SEED_HI, EVAL_SEED_LO, EVAL_SEED_HI,
)

__all__ = [
    "ARM_DOF", "ACTION_DIM",
    "PhaseConfig", "PHASES", "get_phase",
    "CONTROL_HZ", "PHYSICS_HZ", "SUBSTEPS", "MAX_OBSTACLES_IN_OBS", "DELTA_Q_SCALE",
    "TRAIN_SEED_LO", "TRAIN_SEED_HI", "EVAL_SEED_LO", "EVAL_SEED_HI",
]
