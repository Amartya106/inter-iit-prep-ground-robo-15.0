"""
obstacle_generator.py

Random obstacle generator for ManipulaRL (Stages 3 & 4).

Spawns simple box/cylinder obstacles at randomized positions within the
workspace, keeping clear of the robot base, peg, and target/hole. Obstacles
are created procedurally with PyBullet primitives (createCollisionShape /
createVisualShape / createMultiBody), so no URDF files are required and
every field (count, size, position range, mass, friction) can be
randomized per episode -- this is what backs the "hidden obstacle
configurations" used in Stage 3/4 evaluation and the Final Generalization
Challenge.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p


@dataclass
class ObstacleConfig:
    num_obstacles: int = 3
    workspace_xy_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.2, 0.65), (-0.35, 0.35))
    height_range: Tuple[float, float] = (0.05, 0.20)       # obstacle full height
    half_extent_range: Tuple[float, float] = (0.02, 0.05)  # box obstacles (xy half-extents)
    radius_range: Tuple[float, float] = (0.02, 0.04)       # cylinder obstacles
    min_clearance: float = 0.08          # min xy distance from any keepout point
    shapes: Tuple[str, ...] = ("box", "cylinder")
    mass: float = 0.0                    # 0 = static/fixed obstacle
    friction: float = 0.6
    color: Tuple[float, float, float, float] = (0.85, 0.2, 0.2, 1.0)
    seed: Optional[int] = None


class RandomObstacleGenerator:
    """Spawns and manages a randomized set of obstacles in a PyBullet scene."""

    def __init__(self, config: Optional[ObstacleConfig] = None, client_id: int = 0):
        self.cfg = config or ObstacleConfig()
        self.client_id = client_id
        self._rng = np.random.default_rng(self.cfg.seed)
        self.obstacle_ids: List[int] = []

    def reseed(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)

    def clear(self) -> None:
        """Remove all currently-tracked obstacles from the simulation."""
        for oid in self.obstacle_ids:
            try:
                p.removeBody(oid, physicsClientId=self.client_id)
            except p.error:
                pass  # body was already removed (e.g. resetSimulation was called)
        self.obstacle_ids = []

    def spawn(self, keepout_points: Optional[Sequence[np.ndarray]] = None) -> List[int]:
        """
        Spawn a fresh randomized set of obstacles.

        keepout_points: iterable of (x, y[, z]) points -- e.g. the peg
        position, the hole/target position, the robot base -- that
        obstacles must be placed at least `min_clearance` away from (xy
        distance only). Call this once per `env.reset()`.
        """
        self.clear()
        keepout_points = list(keepout_points or [])
        cfg = self.cfg
        client = self.client_id

        for _ in range(cfg.num_obstacles):
            xy = self._sample_valid_xy(keepout_points)
            shape = self._rng.choice(cfg.shapes)

            if shape == "box":
                he = self._rng.uniform(cfg.half_extent_range[0], cfg.half_extent_range[1], size=2)
                half_height = self._rng.uniform(*cfg.height_range) / 2.0
                half_extents = [he[0], he[1], half_height]
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                           rgbaColor=list(cfg.color), physicsClientId=client)
                z = half_height
            else:  # cylinder
                radius = self._rng.uniform(*cfg.radius_range)
                height = self._rng.uniform(*cfg.height_range)
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                           rgbaColor=list(cfg.color), physicsClientId=client)
                z = height / 2.0

            body_id = p.createMultiBody(
                baseMass=cfg.mass,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[xy[0], xy[1], z],
                physicsClientId=client,
            )
            p.changeDynamics(body_id, -1, lateralFriction=cfg.friction, physicsClientId=client)
            self.obstacle_ids.append(body_id)

        return self.obstacle_ids

    def any_contact(self, body_id: int) -> bool:
        """True if `body_id` is currently in contact with any obstacle."""
        for oid in self.obstacle_ids:
            if p.getContactPoints(bodyA=body_id, bodyB=oid, physicsClientId=self.client_id):
                return True
        return False

    def _sample_valid_xy(self, keepout_points: List[np.ndarray], max_tries: int = 50) -> np.ndarray:
        (x_lo, x_hi), (y_lo, y_hi) = self.cfg.workspace_xy_range
        xy = self._rng.uniform([x_lo, y_lo], [x_hi, y_hi])
        for _ in range(max_tries):
            xy = self._rng.uniform([x_lo, y_lo], [x_hi, y_hi])
            if all(
                np.linalg.norm(xy - np.asarray(kp)[:2]) >= self.cfg.min_clearance
                for kp in keepout_points
            ):
                return xy
        return xy  # best-effort fallback if no valid spot found in max_tries