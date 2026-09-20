"""ManipulaRL-UR5 -- the UR5 (6-DOF) + Robotiq 2F-85 port of manipularl/, built to test
whether the descent-phase xy-drift found in E68 (see EXPERIMENTS.md) is a symptom of KUKA
iiwa's redundant 7th DOF (a 6-DOF pose target leaves one null-space degree of freedom IK can
wander through between re-solves). Same task definitions, reward shaping, and episode
randomization as the KUKA package -- reused directly, unmodified, from `manipularl.rewards`/
`manipularl.randomization` -- only the robot-specific pieces (URDF, DOF count, end-effector
link, rest pose) are new. See the plan this was built from for the full rationale.
"""

from .env import ManipulaRLEnv
from .configs import PHASES, PhaseConfig, get_phase

__all__ = ["ManipulaRLEnv", "PHASES", "PhaseConfig", "get_phase"]
