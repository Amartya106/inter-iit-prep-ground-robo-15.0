"""
randomization.py -- per-episode scene sampling with a hard train/eval split.

`EpisodeConfig` is everything that varies from one episode to the next:
object poses, the arm's initial configuration, obstacle count/layout, and
(Phase 5+) physical parameters. `EpisodeSampler` draws it from a PCG64
stream keyed by an episode index; train uses indices in [0, 1e6), evaluation
uses [1e6, 2e6), so a policy is never evaluated on a layout it could have
seen in training. This is what makes "unseen configurations" a property of
the code rather than a hope.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .configs import PhaseConfig, TRAIN_SEED_LO, TRAIN_SEED_HI, EVAL_SEED_LO, EVAL_SEED_HI

# Workspace bounds shared with the scaffold's sampling ranges.
_PEG_XY_LO = np.array([0.30, -0.25])
_PEG_XY_HI = np.array([0.60, 0.25])
_TARGET_XY_LO = np.array([0.35, -0.20])
_TARGET_XY_HI = np.array([0.55, 0.20])
_REACH_Z = (0.10, 0.35)


@dataclass
class EpisodeConfig:
    seed: int
    split: str                                  # "train" | "eval"

    # arm
    init_joint_offset: np.ndarray = field(default_factory=lambda: np.zeros(7))

    # objects
    peg_xy: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.0]))
    peg_yaw: float = 0.0
    peg_tilt: float = 0.0
    reach_target: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.0, 0.2]))
    place_target: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.0, 0.2]))
    hole_xy: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.0]))

    # obstacles: list of (x, y, z, shape, sx, sy, sz)
    obstacles: List[tuple] = field(default_factory=list)

    # dynamics (Phase 5); None -> use env defaults
    peg_mass: Optional[float] = None
    lateral_friction: Optional[float] = None
    joint_damping: Optional[float] = None
    restitution: Optional[float] = None
    bore_clearance: Optional[float] = None


class EpisodeSampler:
    def __init__(self, cfg: PhaseConfig, split: str = "train", index_seed: Optional[int] = None):
        assert split in ("train", "eval")
        self.cfg = cfg
        self.split = split
        self._lo, self._hi = (
            (TRAIN_SEED_LO, TRAIN_SEED_HI) if split == "train" else (EVAL_SEED_LO, EVAL_SEED_HI)
        )
        # A separate stream chooses episode indices, so callers can just ask
        # for "the next episode" and still land in the right partition.
        # `index_seed` defaults to None, which reproduces every existing
        # logged result's exact episode sequence; pass a value only to draw
        # an independent held-out set (e.g. a repeated-eval variance check).
        self._index_rng = np.random.default_rng(
            index_seed if index_seed is not None else (0 if split == "train" else 1))
        # obstacle-count curriculum: None -> use cfg.obstacle_min/max as-is;
        # set via set_obstacle_range() (live-updated by ObstacleCountCurriculum).
        self._obstacle_range: Optional[Tuple[int, int]] = None

    def set_obstacle_range(self, lo: int, hi: int):
        self._obstacle_range = (int(lo), int(hi))

    def _episode_seed(self, episode_index: Optional[int]) -> int:
        if episode_index is None:
            return int(self._index_rng.integers(self._lo, self._hi))
        return self._lo + int(episode_index) % (self._hi - self._lo)

    def sample(self, episode_index: Optional[int] = None) -> EpisodeConfig:
        seed = self._episode_seed(episode_index)
        rng = np.random.default_rng(seed)
        cfg = self.cfg

        ec = EpisodeConfig(seed=seed, split=self.split)

        ec.init_joint_offset = rng.uniform(-0.4, 0.4, size=7)

        if cfg.randomize_layout:
            ec.peg_xy = rng.uniform(_PEG_XY_LO, _PEG_XY_HI)
            ec.peg_yaw = rng.uniform(-np.pi, np.pi)
            ec.peg_tilt = rng.uniform(0.0, 0.10)
            rt_xy = rng.uniform(_TARGET_XY_LO, _TARGET_XY_HI)
            ec.reach_target = np.array([rt_xy[0], rt_xy[1], rng.uniform(*_REACH_Z)])
            pt_xy = rng.uniform(_TARGET_XY_LO, _TARGET_XY_HI)
            ec.place_target = np.array([pt_xy[0], pt_xy[1], rng.uniform(*_REACH_Z)])
            ec.hole_xy = rng.uniform(_TARGET_XY_LO, _TARGET_XY_HI)

        n = 0
        if cfg.num_obstacles > 0:
            lo, hi = self._obstacle_range or (cfg.obstacle_min, cfg.obstacle_max)
            hi = max(lo, hi)
            n = int(rng.integers(lo, hi + 1))
        ec.obstacles = self._sample_obstacles(rng, n, ec)

        if cfg.randomize_dynamics and cfg.dynamics_ranges:
            dr = cfg.dynamics_ranges
            ec.peg_mass = float(rng.uniform(*dr["peg_mass"]))
            ec.lateral_friction = float(rng.uniform(*dr["lateral_friction"]))
            ec.joint_damping = float(rng.uniform(*dr["joint_damping"]))
            ec.restitution = float(rng.uniform(*dr["restitution"]))
            ec.bore_clearance = float(rng.uniform(*dr["bore_clearance"]))

        return ec

    def _sample_obstacles(self, rng, n: int, ec: EpisodeConfig) -> List[tuple]:
        # Each keepout point has its own exclusion radius. hole_xy's is much
        # larger than the others': the hole-adjacent teleport curricula place
        # the arm there regardless of nearby obstacles, so a small radius let
        # obstacles spawn close enough to make some episodes unwinnable by
        # construction before the policy could act.
        keepout = [
            (np.array([0.0, 0.0]), 0.10),          # robot base
            (np.asarray(ec.peg_xy), 0.10),
            (np.asarray(ec.hole_xy), 0.28),
            (np.asarray(ec.place_target[:2]), 0.10),
            (np.asarray(ec.reach_target[:2]), 0.10),
        ]
        out = []
        for _ in range(n):
            xy = None
            for _try in range(50):
                cand = rng.uniform([0.2, -0.35], [0.65, 0.35])
                if all(np.linalg.norm(cand - k) >= r for k, r in keepout):
                    xy = cand
                    break
            if xy is None:
                # All 50 attempts violated some keepout. Skip this obstacle
                # rather than place it anyway: a slightly emptier scene beats
                # a guaranteed-unwinnable one.
                continue
            shape = "box" if rng.random() < 0.5 else "cylinder"
            h = rng.uniform(0.05, 0.20)
            if shape == "box":
                sx, sy = rng.uniform(0.02, 0.05, size=2)
                sz = h / 2.0
            else:
                sx = sy = rng.uniform(0.02, 0.04)
                sz = h
            z = sz if shape == "box" else h / 2.0
            out.append((float(xy[0]), float(xy[1]), float(z), shape, float(sx), float(sy), float(sz)))
        return out
