"""
env.py -- ManipulaRLEnv, the single Gymnasium environment for all six phases.

Rebuilt from the assignment scaffold to make the PS actually solvable:

  * PERSISTENT SCENE. plane / table / arm / peg / hole plate / obstacle pool
    are created once in __init__; reset() only teleports bodies and resets
    joint states. The scaffold's per-episode p.resetSimulation() + URDF
    reloads are gone -- reset is ~10-20x cheaper.

  * DELTA-JOINT-POSITION ACTION. action = 7 * (delta q, scaled) + 1 gripper,
    applied with POSITION_CONTROL. Fully joint-space (the safe reading of
    "the decision-making must be learned via RL; conventional control only as
    low-level infrastructure"), and far smoother to learn than the scaffold's
    raw velocity command.

  * 20 Hz CONTROL over 240 Hz physics (12 substeps), vs the scaffold's single
    240 Hz tick with no action repeat.

  * REAL HOLE GEOMETRY. a bore ring on the table, built from cfg.hole_segments
    flat box segments (default 8, an octagon; PyBullet has no boolean/CSG
    cutout, so a true circle isn't possible, only an approximation of one) --
    the scaffold's Stage 4 had no hole at all (target was a point at z=0 and
    success needed the peg 3 cm below it: impossible).

  * OBSTACLES IN THE OBSERVATION. the K nearest obstacles are encoded, so
    Phase 3 avoidance and "generalize to unseen layouts" are learnable. The
    observation is fixed once here and only ever zero-filled per phase, never
    redefined.

  * eval-time noise / perturbation live in wrappers.py, never here.

Robot: kuka_iiwa (7-DOF), bundled with pybullet_data. Grasp is the scaffold's
constraint-based "snap" (kept intentionally -- a stable grasp model for early
RL; not a finger-force simulation).
"""

import dataclasses
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from .configs import (
    PhaseConfig, get_phase, ARM_DOF, ACTION_DIM, SUBSTEPS, PHYSICS_HZ,
    DELTA_Q_SCALE, MAX_OBSTACLES_IN_OBS,
)
from .randomization import EpisodeSampler
from .rewards import RewardComputer

_REST_POSE = np.array([0.0, 0.4, 0.0, -1.6, 0.0, 1.0, 0.0])
_EE_LINK = 6
_PEG_RADIUS = 0.015
_PEG_HEIGHT = 0.08
_TABLE_TOP_Z = 0.0
_HOLE_BOX_HALF = 0.010
_HOLE_HEIGHT = 0.10
# cfg.hole_chamfer: an angled funnel above the straight bore, widening the
# capture opening so an aligned peg slides in instead of landing on the rim.
_CHAMFER_HEIGHT = 0.03
_CHAMFER_TILT_DEG = 35.0
_FINE_CONTROL_RADIUS = 0.05   # cfg.fine_control_frac's taper zone
_GRASP_TRIGGER = 0.09    # forgiving on purpose: dense grasp-shaping and the
                         # grasp gate need to coincide for exploration to find it
_GRASP_THRESH = -0.2     # gripper-signal threshold to *attempt* a grasp
_AIRSPACE_LOW_M = 0.03   # descend_only: random start height above the hole
_AIRSPACE_HIGH_M = 0.15  # is drawn uniformly from [LOW, HIGH]
_GRASP_OFFSET_Z = 0.035  # peg_in_hole only: grasp this far above the peg's
                         # own center instead of at it (offset 0 elsewhere).
                         # A zero-offset grasp puts the flange itself inside
                         # the solid hole ring once the peg reaches insertion
                         # depth, which makes real insertion physically
                         # impossible -- this offset keeps the flange clear.
                         # Real (non-curriculum) grasps also require the
                         # flange to already be within this band of the
                         # peg's own top (PhaseConfig.grasp_admit_axial_lo/hi/
                         # lateral) before welding; otherwise the weld's
                         # preserve-transform behavior (see _form_grasp) would
                         # lock the flange wherever it happened to be, which
                         # can also make insertion depth unreachable. Teleport
                         # curricula place the peg at exactly this offset
                         # first, so they're always admissible.
_MAX_OBSTACLE_POOL = 8   # >= max of any phase's obstacle_max
_OBSTACLE_HALF = np.array([0.03, 0.03, 0.05])   # matches _build_scene's obstacle boxes
_OBSTACLE_BOUND_R = float(np.linalg.norm(_OBSTACLE_HALF))
_EE_VIRTUAL_INFLATE = 0.04   # m, extra bounding radius for _EE_LINK in the
                             # shaping-only obstacle distance (not physical
                             # collision geometry) -- this link is the
                             # dominant collision contributor, so its dense
                             # repulsion penalty gets an earlier warning
_JOINT_LIMIT_MARGIN = 0.05       # fraction of range that counts as "at limit"


def _rot6d(quat) -> np.ndarray:
    """First two columns of the rotation matrix -- a continuous 6-D orientation."""
    m = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    return m[:, :2].reshape(-1)


class ManipulaRLEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        phase: int = 1,
        render_mode: str = "none",
        split: str = "train",
        gamma: float = 0.99,
        grasp_curriculum: Optional[float] = None,
        align_curriculum: Optional[float] = None,
        insert_curriculum: Optional[float] = None,
        seed: Optional[int] = None,
        cfg_overrides: Optional[dict] = None,
        insert_xy_jitter: Optional[float] = None,
        insert_tilt_jitter_deg: Optional[float] = None,
        linger_pen_w: Optional[float] = None,
        settle_bonus_w: Optional[float] = None,
        index_seed: Optional[int] = None,
        arm_jitter_pen_w: Optional[float] = None,
        replay_path: Optional[str] = None,
        replay_curriculum: Optional[float] = None,
        keypoint_w: Optional[float] = None,
        satisficing_proximity: Optional[bool] = None,
        risk_seeking_depth: Optional[bool] = None,
    ):
        super().__init__()
        base_cfg = get_phase(phase)
        self.cfg: PhaseConfig = (
            dataclasses.replace(base_cfg, **cfg_overrides) if cfg_overrides else base_cfg
        )
        self.phase = phase
        self.split = split
        self.render_mode = render_mode
        self.max_steps = self.cfg.max_steps
        # fraction of TRAIN episodes that start with the peg already grasped
        self.grasp_curriculum = (
            self.cfg.grasp_curriculum if grasp_curriculum is None else float(grasp_curriculum)
        )
        # fraction of TRAIN episodes (peg_in_hole only) that start already
        # grasped and above the hole mouth -- one stage past grasp_curriculum
        self.align_curriculum = float(align_curriculum or 0.0)
        # fraction of TRAIN episodes (peg_in_hole only) that start already
        # inserted past the success depth -- one stage past align_curriculum
        self.insert_curriculum = float(insert_curriculum or 0.0)
        # cfg.insert_only: how much starting depth is given for free at
        # reset, 1.0=easiest -> 0.0=hardest (the real task), annealed by
        # InsertStartDepthAnneal. Only for split=="train"; eval always forces
        # 0.0 (see reset()) so eval numbers reflect the real, undiluted skill.
        self.insert_start_depth_frac = 1.0
        # cfg.descend_only: how much of the airspace height range
        # _start_pre_airspace() samples from, 0.0=easiest -> 1.0=hardest
        # (full range), annealed by AirspaceHeightAnneal. Only for
        # split=="train"; eval always forces 1.0 (see reset()).
        self.airspace_height_frac = 1.0
        # cfg.descend_only replay-seeded curriculum (_start_from_replay):
        # fraction of TRAIN episodes that start from a recorded successful
        # rollout instead of a randomized airspace height. eval never uses it.
        self.replay_curriculum = float(replay_curriculum or 0.0)
        self._replay = None
        if replay_path is not None:
            import numpy as _np
            self._replay = _np.load(replay_path)
        # cfg.insert_only robustness: random xy offset and tilt jitter
        # applied to _start_pre_inserted()'s teleport target, TRAIN only --
        # eval stays exactly precise so eval numbers remain comparable.
        self.insert_xy_jitter = float(insert_xy_jitter or 0.0)
        self.insert_tilt_jitter_deg = float(insert_tilt_jitter_deg or 0.0)
        # EE-peg catch radius for a gripper-close to form the grasp
        # constraint (see _handle_grasp); only widened by
        # GraspTriggerCurriculum for split=="train", eval sees the real value
        self._grasp_trigger = _GRASP_TRIGGER

        self.sampler = EpisodeSampler(self.cfg, split=split, index_seed=index_seed)
        self.reward_fn = RewardComputer(self.cfg, gamma=gamma)
        if linger_pen_w is not None:
            self.reward_fn.linger_pen_w = float(linger_pen_w)
        if settle_bonus_w is not None:
            self.reward_fn.settle_bonus_w = float(settle_bonus_w)
        if arm_jitter_pen_w is not None:
            self.reward_fn.arm_jitter_pen_w = float(arm_jitter_pen_w)
        if keypoint_w is not None:
            self.reward_fn.keypoint_w = float(keypoint_w)
        if satisficing_proximity is not None:
            self.reward_fn.satisficing_proximity = bool(satisficing_proximity)
        if risk_seeking_depth is not None:
            self.reward_fn.risk_seeking_depth = bool(risk_seeking_depth)
        self._episode_counter = 0
        self._np_random = np.random.default_rng(seed)

        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        self._build_scene()
        self._q_lower, self._q_upper = self._joint_limits()

        self._step_count = 0
        self._grasp_cid = None
        self._last_gripper = 0.0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._cur = None            # current EpisodeConfig
        self._obstacle_active = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        obs_dim = self._assemble_obs(dry_run=True).shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.obs_dim = obs_dim

    def set_grasp_curriculum(self, frac: float):
        """Live-update the pre-grasped-start fraction (curriculum callback via VecEnv.env_method)."""
        self.grasp_curriculum = float(max(0.0, min(1.0, frac)))

    def set_align_curriculum(self, frac: float):
        """Live-update the pre-aligned-start fraction (peg_in_hole only)."""
        self.align_curriculum = float(max(0.0, min(1.0, frac)))

    def set_insert_curriculum(self, frac: float):
        """Live-update the pre-inserted-start fraction (peg_in_hole only)."""
        self.insert_curriculum = float(max(0.0, min(1.0, frac)))

    def set_insert_start_depth_frac(self, frac: float):
        """Live-update Task 2's free starting depth, 1.0 (easiest) -> 0.0 (real).
        No effect unless cfg.insert_only; never applied for split=='eval'."""
        self.insert_start_depth_frac = float(max(0.0, min(1.0, frac)))

    def set_airspace_height_frac(self, frac: float):
        """Live-update how much of the airspace height range descend_only
        samples from, 0.0 (easiest) -> 1.0 (real, full range). eval always uses 1.0."""
        self.airspace_height_frac = float(max(0.0, min(1.0, frac)))

    def set_grasp_trigger(self, radius: float):
        """Live-update the EE-peg catch radius for a grasp attempt.
        Clamped to [_GRASP_TRIGGER, 0.20]."""
        self._grasp_trigger = float(max(_GRASP_TRIGGER, min(0.20, radius)))

    def set_obstacle_range(self, lo: int, hi: int):
        """Live-update the per-episode obstacle-count range."""
        self.sampler.set_obstacle_range(int(lo), int(hi))

    def set_xy_err_weight(self, w: float):
        """Live-update the dense potential-shaping's xy_err coefficient."""
        self.reward_fn.xy_err_weight = float(w)

    def set_tilt_weight(self, w: float):
        """Live-update the dense potential-shaping's tilt coefficient."""
        self.reward_fn.tilt_weight = float(w)

    def set_keypoint_weight(self, w: float):
        """Live-update the keypoint-docking potential's coefficient (default 0.0 = off)."""
        self.reward_fn.keypoint_w = float(w)

    def set_grasp_drop_pen(self, w: float):
        """Live-update the grasp-drop penalty magnitude."""
        self.reward_fn.grasp_drop_pen = float(w)

    def set_collision_pen_mult(self, w: float):
        """Live-update the collision-penalty multiplier."""
        self.reward_fn.collision_pen_mult = float(w)

    def set_collision_terminal(self, flag: bool):
        """Live-toggle whether a collision also ends the episode immediately
        (cfg.collision_is_failure always precludes success regardless)."""
        self.reward_fn.collision_terminal = bool(flag)

    def set_replay_curriculum(self, frac: float):
        """Live-update the fraction of TRAIN episodes starting from a
        recorded replay snapshot (see _start_from_replay)."""
        self.replay_curriculum = float(max(0.0, min(1.0, frac)))

    # ------------------------------------------------------------------ #
    # scene construction (once)
    # ------------------------------------------------------------------ #
    def _build_scene(self):
        c = self.client
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=c)
        self.table_id = p.loadURDF(
            "table/table.urdf", basePosition=[0.5, 0.0, -0.65], useFixedBase=True,
            physicsClientId=c,
        )
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True,
            physicsClientId=c,
        )
        for j in range(ARM_DOF):
            p.enableJointForceTorqueSensor(self.robot_id, j, True, physicsClientId=c)

        # peg (always present in the scene; parked when a phase doesn't use it)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=_PEG_RADIUS, height=_PEG_HEIGHT,
                                     physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=_PEG_RADIUS, length=_PEG_HEIGHT,
                                  rgbaColor=[0.4, 0.5, 0.95, 1], physicsClientId=c)
        self.peg_id = p.createMultiBody(0.05, col, vis, [0.45, 0.0, _PEG_HEIGHT / 2],
                                        physicsClientId=c)

        # hole socket: a ring of thin static boxes forming the bore wall, a
        # regular polygon with cfg.hole_segments sides (8 = octagon).
        self._hole_segments = self.cfg.hole_segments
        self.hole_box_ids = []
        for _ in range(self._hole_segments):
            hc = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[_HOLE_BOX_HALF, _HOLE_BOX_HALF, _HOLE_HEIGHT / 2],
                physicsClientId=c,
            )
            hv = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[_HOLE_BOX_HALF, _HOLE_BOX_HALF, _HOLE_HEIGHT / 2],
                rgbaColor=[0.55, 0.35, 0.2, 1], physicsClientId=c,
            )
            bid = p.createMultiBody(0.0, hc, hv, [10.0, 10.0, -10.0], physicsClientId=c)
            self.hole_box_ids.append(bid)

        # cfg.hole_chamfer: a second, tilted ring above the straight bore,
        # forming a funnel lead-in (see _place_hole's chamfer branch). Always
        # created so toggling the flag never changes body count/reset cost;
        # parked off-scene when hole_chamfer is False.
        self.hole_chamfer_ids = []
        for _ in range(self._hole_segments):
            cc = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[_HOLE_BOX_HALF, _HOLE_BOX_HALF, _CHAMFER_HEIGHT / 2],
                physicsClientId=c,
            )
            cv = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[_HOLE_BOX_HALF, _HOLE_BOX_HALF, _CHAMFER_HEIGHT / 2],
                rgbaColor=[0.7, 0.5, 0.3, 1], physicsClientId=c,
            )
            bid = p.createMultiBody(0.0, cc, cv, [10.0, 10.0, -10.0], physicsClientId=c)
            self.hole_chamfer_ids.append(bid)

        # obstacle pool (static boxes/cylinders, parked below the floor by default)
        self.obstacle_ids = []
        self.obstacle_shapes = []   # "box"/"cylinder" per pool slot, set on reset
        for _ in range(_MAX_OBSTACLE_POOL):
            oc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.05],
                                        physicsClientId=c)
            ov = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.05],
                                     rgbaColor=[0.85, 0.2, 0.2, 1], physicsClientId=c)
            oid = p.createMultiBody(0.0, oc, ov, [10.0, 10.0, -10.0], physicsClientId=c)
            self.obstacle_ids.append(oid)
            self.obstacle_shapes.append("box")

    def _joint_limits(self):
        lo, up = [], []
        for j in range(ARM_DOF):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client)
            lo.append(info[8]); up.append(info[9])
        return np.array(lo), np.array(up)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        idx = None if options is None else options.get("episode_index")
        ec = self.sampler.sample(idx)
        self._cur = ec
        self._episode_counter += 1
        self._step_count = 0
        self._last_gripper = 0.0
        self._prev_action[:] = 0.0
        self.reward_fn.reset()

        if self._grasp_cid is not None:
            p.removeConstraint(self._grasp_cid, physicsClientId=self.client)
            self._grasp_cid = None

        # arm -> rest pose + per-episode offset, clipped to limits
        q0 = np.clip(_REST_POSE + ec.init_joint_offset, self._q_lower + 0.05, self._q_upper - 0.05)
        for j in range(ARM_DOF):
            p.resetJointState(self.robot_id, j, float(q0[j]), 0.0, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0,
                                    physicsClientId=self.client)

        # dynamics randomization (Phase 5+)
        if ec.joint_damping is not None:
            for j in range(ARM_DOF):
                p.changeDynamics(self.robot_id, j, jointDamping=ec.joint_damping,
                                 physicsClientId=self.client)
        peg_mass = ec.peg_mass if ec.peg_mass is not None else 0.05
        peg_fric = ec.lateral_friction if ec.lateral_friction is not None else 0.6
        peg_rest = ec.restitution if ec.restitution is not None else 0.0
        p.changeDynamics(self.peg_id, -1, mass=peg_mass, lateralFriction=peg_fric,
                         restitution=peg_rest, physicsClientId=self.client)

        # peg placement
        if self.cfg.use_peg:
            yaw = ec.peg_yaw
            tilt = ec.peg_tilt
            quat = p.getQuaternionFromEuler([tilt, 0.0, yaw])
            p.resetBasePositionAndOrientation(
                self.peg_id, [ec.peg_xy[0], ec.peg_xy[1], _PEG_HEIGHT / 2 + 0.001], quat,
                physicsClientId=self.client,
            )
            p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        else:
            p.resetBasePositionAndOrientation(self.peg_id, [10, 10, -10], [0, 0, 0, 1],
                                              physicsClientId=self.client)

        # hole socket placement
        self._place_hole(ec)

        # obstacles
        self._place_obstacles(ec)

        # settle
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client)

        # --- start-state curriculum (training only, split == "eval" always
        # uses the real un-boosted start) ---
        if (self.cfg.task == "peg_in_hole" and self.cfg.descend_only
                and self.split == "train" and self._replay is not None
                and self.replay_curriculum > 0.0
                and self._np_random.random() < self.replay_curriculum):
            # descend_only replay-seeded curriculum: start from a recorded
            # successful rollout instead of a randomized airspace height.
            self._start_from_replay()
        elif self.cfg.task == "peg_in_hole" and self.cfg.descend_only:
            # descend_only: start already grasped + xy-aligned at a random
            # height within the airspace column above the hole.
            self._start_pre_airspace()
        elif self.cfg.task == "peg_in_hole" and self.cfg.insert_only:
            # insert_only: every episode, train and eval, starts already
            # grasped + xy/tilt-aligned in the insertion zone. Train uses the
            # curriculum-controlled starting depth; eval always forces the
            # real depth=0 so eval numbers are never flattered.
            frac = self.insert_start_depth_frac if self.split == "train" else 0.0
            self._start_pre_inserted(depth_frac=frac)
        elif (self.split == "train" and self.cfg.task == "peg_in_hole"
                and self.insert_curriculum > 0.0
                and self._np_random.random() < self.insert_curriculum):
            self._start_pre_inserted()
        elif (self.split == "train" and self.cfg.task == "peg_in_hole"
                and self.align_curriculum > 0.0
                and self._np_random.random() < self.align_curriculum):
            self._start_pre_aligned()
        elif (self.split == "train" and self.cfg.use_peg
                and self.grasp_curriculum > 0.0
                and self._np_random.random() < self.grasp_curriculum):
            self._start_pre_grasped()

        return self._assemble_obs(), {"episode_seed": ec.seed, "split": ec.split}

    def _form_grasp(self, child_off: float):
        """Create the EE<->peg JOINT_FIXED constraint. Every grasp in this
        file (real, via `_handle_grasp`, and every teleport curriculum) goes
        through here. `cfg.grasp_max_force`, if set, caps the constraint
        solver's corrective force so the peg can lag/deflect instead of
        being perfectly rigid; `None` (default) leaves PyBullet's rigid
        default, unchanged from the original behavior.

        `child_off == 0.0` (phases 1-3, align_only): the original weld,
        childFramePosition=[0,0,0], identity frame orientations. Harmless
        here since these tasks' reward has no tilt term.

        `child_off > 0.0` (peg_in_hole, non-align_only): a **preserve-
        transform** weld instead of a hardcoded offset. Bug this fixes:
        `JOINT_FIXED` with identity frame orientations forces
        `child_orn == parent_orn` regardless of `childFramePosition` -- so a
        hardcoded `[0,0,child_off]` weld also force-rotates the peg to the
        flange's arbitrary orientation on every real grasp, since nothing
        controls flange orientation at grasp time. That was the root cause
        of a measured 20-40deg grasp tilt. The fix computes the actual
        relative transform (peg's pose inverted, composed with the flange's
        pose, not the reverse -- composition order matters and was caught
        empirically, not by inspection) and welds exactly that, so grasping
        never re-orients the peg. Curriculum teleports place the peg at
        exactly `peg_pos + [0,0,child_off]` with an identity flange quat
        before calling this, so for them the result is bit-identical to the
        old hardcoded weld; only a real cold grasp changes behavior.
        `_handle_grasp` gates entry into this branch with an admissibility
        check so the preserved transform still keeps the flange near the
        peg's top -- without it, an arbitrary preserved transform could weld
        the flange level with the peg center and make insert_success_depth
        physically unreachable, the original reason `_GRASP_OFFSET_Z` exists.
        """
        if child_off > 0.0:
            fl_pos, fl_orn = p.getLinkState(self.robot_id, _EE_LINK, computeForwardKinematics=True,
                                            physicsClientId=self.client)[:2]
            pg_pos, pg_orn = p.getBasePositionAndOrientation(self.peg_id,
                                                              physicsClientId=self.client)
            # childFrame must satisfy flange_pose == peg_pose ∘ childFrame,
            # so childFrame = peg_pose^-1 ∘ flange_pose (invert the peg's
            # pose, compose with the flange's -- the reversed order is wrong).
            inv_pos, inv_orn = p.invertTransform(pg_pos, pg_orn)
            rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, fl_pos, fl_orn)
            cid = p.createConstraint(
                self.robot_id, _EE_LINK, self.peg_id, -1, p.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], rel_pos, childFrameOrientation=rel_orn,
                physicsClientId=self.client,
            )
        else:
            cid = p.createConstraint(
                self.robot_id, _EE_LINK, self.peg_id, -1, p.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, child_off], physicsClientId=self.client,
            )
        if self.cfg.grasp_max_force is not None:
            p.changeConstraint(cid, maxForce=float(self.cfg.grasp_max_force),
                               physicsClientId=self.client)
        return cid

    def _start_pre_grasped(self):
        """Move the EE onto the peg via IK and form the grasp constraint."""
        peg_pos = np.array(p.getBasePositionAndOrientation(self.peg_id,
                                                           physicsClientId=self.client)[0])
        target = peg_pos + np.array([0.0, 0.0, 0.02])
        for _ in range(40):
            jt = p.calculateInverseKinematics(self.robot_id, _EE_LINK, target.tolist(),
                                              physicsClientId=self.client)
            for j in range(ARM_DOF):
                p.resetJointState(self.robot_id, j, float(jt[j]), 0.0, physicsClientId=self.client)
            p.stepSimulation(physicsClientId=self.client)
        ee = np.array(p.getLinkState(self.robot_id, _EE_LINK, physicsClientId=self.client)[0])
        if np.linalg.norm(ee - peg_pos) < 0.12:
            # peg_in_hole: top-offset grasp (_GRASP_OFFSET_Z); align_only
            # stays zero-offset like phases 1-3, since its success is pure
            # xy/tilt alignment and giving it the offset regressed grasp-hold.
            child_off = (_GRASP_OFFSET_Z if self.cfg.task == "peg_in_hole"
                        and not self.cfg.align_only else 0.0)
            self._grasp_cid = self._form_grasp(child_off)
            self.reward_fn._grasp_bonus_paid = True   # don't pay the gate for a free grasp
            self.reward_fn._picked = True

    def _ik_limited(self, target_pos, target_quat):
        """calculateInverseKinematics, converged, then clipped to joint limits.

        Unbounded IK can return solutions that violate the URDF's joint
        limits; `resetJointState` applies them anyway, and once
        POSITION_CONTROL takes over, the engine's own limit enforcement
        fights the motor command and produces violent oscillation. A
        null-space form (lowerLimits/upperLimits/jointRanges/restPoses) was
        tried and measured worse on both counts: 14-36cm position error with
        joints still outside their limits, versus 0.00cm error from a plain
        200-iteration solve with the limits enforced explicitly afterward
        here. The null-space restPoses bias pulls toward _REST_POSE, which
        is nowhere near the folded configuration insertion needs.
        """
        jt = p.calculateInverseKinematics(
            self.robot_id, _EE_LINK, list(target_pos), list(target_quat),
            maxNumIterations=200, residualThreshold=1e-5,
            physicsClientId=self.client)
        return np.clip(np.asarray(jt[:ARM_DOF], dtype=float),
                       self._q_lower, self._q_upper)

    def _ik_converge(self, target, quat, iters=80, seed_rest=True):
        """One IK-teleport attempt: (optionally) seed from _REST_POSE, then
        iterate `iters` steps of _ik_limited + resetJointState, returning the
        achieved flange position. `iters=80` gives margin above the ~60
        iterations measured to fully converge every sampled hole position to
        0.00cm error. `seed_rest=True` matters: seeding from whatever joint
        state was left over from the previous episode can converge to a
        different local minimum with small XY error but real Z error, on
        this redundant 7-DOF arm under an orientation constraint -- callers
        should check the full 3D position error, not just XY.
        """
        if seed_rest:
            for j in range(ARM_DOF):
                p.resetJointState(self.robot_id, j, float(_REST_POSE[j]), 0.0,
                                  physicsClientId=self.client)
        for _ in range(iters):
            jt = self._ik_limited(target, quat)
            for j in range(ARM_DOF):
                p.resetJointState(self.robot_id, j, float(jt[j]), 0.0, physicsClientId=self.client)
            p.stepSimulation(physicsClientId=self.client)
        return np.array(p.getLinkState(self.robot_id, _EE_LINK, physicsClientId=self.client)[0])

    def _start_pre_aligned(self):
        """peg_in_hole curriculum: teleport the arm+peg to "grasped, held
        above the hole mouth, roughly vertical" -- one stage past
        _start_pre_grasped(). Uses the peg's own body frame as the grasp
        target, so this is safe to call with any peg start pose.
        """
        target_xy = self._hole_xy
        clear_z = self._hole_mouth_z + 0.08
        target = np.array([target_xy[0], target_xy[1], clear_z])
        quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])   # peg axis vertical
        ee = self._ik_converge(target, quat, iters=80, seed_rest=True)
        peg_pos = ee - np.array([0.0, 0.0, _GRASP_OFFSET_Z])   # same top-offset grasp as a real grasp
        p.resetBasePositionAndOrientation(self.peg_id, peg_pos.tolist(), quat,
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._grasp_cid = self._form_grasp(_GRASP_OFFSET_Z)
        self.reward_fn._grasp_bonus_paid = True
        self.reward_fn._picked = True

    def _start_pre_inserted(self, depth_frac: Optional[float] = None):
        """peg_in_hole curriculum: teleport the arm+peg to "grasped, held
        already past the insertion-success depth inside the bore, vertical"
        -- one stage past _start_pre_aligned(). Gives the policy direct
        experience of holding/pushing a peg already through the bore, a
        motion transit+align curricula alone never sample. Same
        IK-teleport-then-constrain mechanics as _start_pre_aligned, deeper.

        `depth_frac` (used by cfg.insert_only): None or 1.0 -> easiest,
        starts a bit past insert_success_depth. 0.0 -> hardest/real, starts
        with depth exactly 0 (peg tip just touching the mouth, aligned,
        grasped) and the whole push-through is left to learn. Intermediate
        values interpolate linearly.
        """
        target_xy = self._hole_xy
        max_depth = float(self.cfg.insert_success_depth) + 0.005   # a bit past
        depth = max_depth if depth_frac is None else max(0.0, min(1.0, depth_frac)) * max_depth
        peg_target_z = self._hole_mouth_z - depth        # where the PEG's center should end up
        # flange target = peg target + _GRASP_OFFSET_Z, keeping the flange
        # clear of the solid hole ring at real insertion depth
        flange_target_z = peg_target_z + _GRASP_OFFSET_Z

        # xy/tilt jitter (cfg.insert_only robustness, TRAIN only -- eval
        # stays exactly precise). The bounded IK solver occasionally
        # converges to a wildly wrong configuration once the target is
        # perturbed off-center, so retry with a fresh jitter draw up to 3
        # times, checked against actual EE convergence, falling back to the
        # exact unjittered placement if every attempt still lands wrong.
        max_ok_err = self.insert_xy_jitter + 0.03
        ee = None
        for attempt in range(3):
            roll, pitch = 0.0, 0.0
            xy = np.array(self._hole_xy, dtype=float)
            if self.split == "train":
                if self.insert_xy_jitter > 0.0:
                    ang = self._np_random.uniform(0.0, 2 * np.pi)
                    r = self._np_random.uniform(0.0, self.insert_xy_jitter)
                    xy = xy + np.array([r * np.cos(ang), r * np.sin(ang)])
                if self.insert_tilt_jitter_deg > 0.0:
                    tilt_mag = np.radians(self._np_random.uniform(0.0, self.insert_tilt_jitter_deg))
                    tilt_dir = self._np_random.uniform(0.0, 2 * np.pi)
                    roll = tilt_mag * np.cos(tilt_dir)
                    pitch = tilt_mag * np.sin(tilt_dir)
            if attempt == 2:
                xy, roll, pitch = np.array(self._hole_xy, dtype=float), 0.0, 0.0  # safe fallback
            target = np.array([xy[0], xy[1], flange_target_z])
            quat = p.getQuaternionFromEuler([roll, pitch, 0.0])
            ee_try = self._ik_converge(target, quat, iters=80, seed_rest=True)
            err = float(np.linalg.norm(ee_try - target))
            if err < max_ok_err or attempt == 2:
                ee = ee_try
                break
        ee = ee if ee is not None else np.array(
            p.getLinkState(self.robot_id, _EE_LINK, physicsClientId=self.client)[0])
        peg_pos = ee - np.array([0.0, 0.0, _GRASP_OFFSET_Z])   # same top-offset grasp as a real grasp
        p.resetBasePositionAndOrientation(self.peg_id, peg_pos.tolist(), quat,
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._grasp_cid = self._form_grasp(_GRASP_OFFSET_Z)
        self.reward_fn._grasp_bonus_paid = True
        self.reward_fn._picked = True
        # pre-set the depth-progress checkpoint to what the teleport already
        # provides, so only further real progress earns the progress bonus
        from .rewards import _DEPTH_PROGRESS_CAP_CM, _DEPTH_PROGRESS_STEP_CM
        self.reward_fn._depth_checkpoint_cm = min(
            _DEPTH_PROGRESS_CAP_CM, int(depth * 100.0) // _DEPTH_PROGRESS_STEP_CM)

    def _start_pre_airspace(self):
        """peg_in_hole curriculum (cfg.descend_only): teleport to "grasped,
        xy/tilt-aligned, at a random height within the airspace column above
        the hole" -- narrower than _start_pre_aligned's fixed 8cm clearance,
        wider than _start_pre_inserted's at/past-the-mouth start. The
        randomized height is the point: the policy has to learn a smooth,
        aligned descent over a variable distance. Reuses the real
        _GRASP_OFFSET_Z grasp and the full, unmodified peg_in_hole
        success/potential (xy+tilt+depth), unlike align_only's
        height-free potential.
        """
        target_xy = np.array(self._hole_xy, dtype=float)
        # eval always samples the full/real range regardless of training's
        # anneal state, so eval numbers reflect undiluted skill
        frac = self.airspace_height_frac if self.split == "train" else 1.0
        high = _AIRSPACE_LOW_M + frac * (_AIRSPACE_HIGH_M - _AIRSPACE_LOW_M)
        height = self._np_random.uniform(_AIRSPACE_LOW_M, high)
        peg_target_z = self._hole_mouth_z + height
        flange_target_z = peg_target_z + _GRASP_OFFSET_Z
        quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])   # peg axis vertical

        target = np.array([target_xy[0], target_xy[1], flange_target_z])
        # a single attempt suffices: seed_rest=True makes it deterministic
        # given the same target, unlike _start_pre_inserted where the
        # target itself varies per retry via jitter
        ee = self._ik_converge(target, quat, iters=80, seed_rest=True)
        peg_pos = ee - np.array([0.0, 0.0, _GRASP_OFFSET_Z])
        p.resetBasePositionAndOrientation(self.peg_id, peg_pos.tolist(), quat,
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._grasp_cid = self._form_grasp(_GRASP_OFFSET_Z)
        self.reward_fn._grasp_bonus_paid = True
        self.reward_fn._picked = True

    def _start_from_replay(self):
        """cfg.descend_only replay-seeded curriculum: teleport to an exact
        recorded state from a proven-successful rollout (see
        scripts/extract_replay_trajectory.py, self._replay), using
        `resetJointState` directly from recorded joint angles -- no IK, so
        no convergence risk. The hole is repositioned to match the
        snapshot's own recorded hole_xy, since a snapshot is only valid
        paired with the hole position it was recorded against.
        """
        idx = int(self._np_random.integers(0, self._replay["q"].shape[0]))
        q = self._replay["q"][idx]
        peg_pos = self._replay["peg_pos"][idx]
        peg_quat = self._replay["peg_quat"][idx]
        hole_xy = self._replay["hole_xy"][idx]

        self._cur.hole_xy = hole_xy.copy()
        self._place_hole(self._cur)

        for j in range(ARM_DOF):
            p.resetJointState(self.robot_id, j, float(q[j]), 0.0, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0,
                                    physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.peg_id, peg_pos.tolist(), peg_quat.tolist(),
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._grasp_cid = self._form_grasp(_GRASP_OFFSET_Z)
        self.reward_fn._grasp_bonus_paid = True
        self.reward_fn._picked = True

    def _regrasp_with_offset(self) -> bool:
        """Hand-off between Task 1 (align, zero-offset grasp) and Task 2
        (insert, _GRASP_OFFSET_Z top-offset grasp): swap the grasp
        constraint from zero-offset to the depth-enabling top-offset.
        Returns False (no-op) if nothing is currently grasped. Only used by
        the composed two-stage eval (scripts/evaluate_composed.py).

        Explicitly slides the peg along its own current axis (not world-z,
        not the flange's) so its top meets the flange, preserving whatever
        alignment Task 1 already achieved instead of snapping to either
        frame's orientation -- necessary under the preserve-transform
        `_form_grasp` (see its docstring), since Task 1's zero-offset grasp
        already has peg pose == flange pose, so the relative transform alone
        would silently drop the offset rather than apply it.
        """
        if self._grasp_cid is None:
            return False
        fl_pos, _fl_orn = p.getLinkState(self.robot_id, _EE_LINK, physicsClientId=self.client)[:2]
        pg_pos, pg_orn = p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.client)
        peg_axis = np.array(p.getMatrixFromQuaternion(pg_orn)).reshape(3, 3)[:, 2]
        new_peg_pos = np.array(fl_pos) - _GRASP_OFFSET_Z * peg_axis
        p.removeConstraint(self._grasp_cid, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.peg_id, new_peg_pos.tolist(), pg_orn,
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._grasp_cid = self._form_grasp(_GRASP_OFFSET_Z)
        return True

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        arm_a, grip_a = action[:ARM_DOF], float(action[ARM_DOF])

        q = np.array([p.getJointState(self.robot_id, j, physicsClientId=self.client)[0]
                      for j in range(ARM_DOF)])
        scale = DELTA_Q_SCALE
        if (self.cfg.fine_control_frac is not None and self.cfg.use_hole
                and self._grasp_cid is not None):
            # narrows the actuator's own per-step motion near the hole (see
            # PhaseConfig.fine_control_frac), applies to whatever produced
            # `action`, RL policy or scripted controller alike
            peg_xy = np.array(
                p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.client)[0][:2])
            if float(np.linalg.norm(peg_xy - self._hole_xy)) < _FINE_CONTROL_RADIUS:
                scale = DELTA_Q_SCALE * self.cfg.fine_control_frac
        q_target = np.clip(q + arm_a * scale, self._q_lower, self._q_upper)
        if self.cfg.joint_max_velocity is not None:
            # setJointMotorControlArray doesn't expose maxVelocity;
            # setJointMotorControl2 (per-joint) does
            for j in range(ARM_DOF):
                p.setJointMotorControl2(
                    self.robot_id, j, p.POSITION_CONTROL,
                    targetPosition=float(q_target[j]), force=200.0,
                    maxVelocity=float(self.cfg.joint_max_velocity),
                    physicsClientId=self.client,
                )
        else:
            p.setJointMotorControlArray(
                self.robot_id, list(range(ARM_DOF)), p.POSITION_CONTROL,
                targetPositions=q_target.tolist(), forces=[200.0] * ARM_DOF,
                physicsClientId=self.client,
            )
        grasp_attempt = self._handle_grasp(grip_a)
        self._last_gripper = grip_a
        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=self.client)

        s = self._state_dict(grasp_attempt=grasp_attempt)
        collision_links = self._obstacle_collision_links()
        collision = bool(collision_links)
        reward, terminated, success, rinfo = self.reward_fn.compute(s, action, collision)
        truncated = self._step_count >= self.max_steps

        self._prev_action = action
        info = {
            "success": success,
            "collision": collision,
            "collision_links": collision_links,
            "episode_collided": rinfo["episode_collided"],
            **self.reward_fn.stats,
        }
        return self._assemble_obs(s), reward, bool(terminated), bool(truncated), info

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    # ------------------------------------------------------------------ #
    # placement helpers
    # ------------------------------------------------------------------ #
    def _place_hole(self, ec):
        if not self.cfg.use_hole:
            for bid in self.hole_box_ids + self.hole_chamfer_ids:
                p.resetBasePositionAndOrientation(bid, [10, 10, -10], [0, 0, 0, 1],
                                                  physicsClientId=self.client)
            self._hole_xy = np.array([0.45, 0.0])
            self._hole_mouth_z = _HOLE_HEIGHT
            self._bore_inradius = _PEG_RADIUS + 0.003
            return
        clr = ec.bore_clearance if ec.bore_clearance is not None else 0.003
        ring_r = _PEG_RADIUS + clr + _HOLE_BOX_HALF
        bore_inradius = _PEG_RADIUS + clr   # the bore's actual inner wall radius
        self._bore_inradius = bore_inradius   # read by rewards.py's keypoint_w via _state_dict()
        cx, cy = ec.hole_xy
        for k, bid in enumerate(self.hole_box_ids):
            ang = 2 * np.pi * k / self._hole_segments
            bx = cx + ring_r * np.cos(ang)
            by = cy + ring_r * np.sin(ang)
            quat = p.getQuaternionFromEuler([0, 0, ang])
            p.resetBasePositionAndOrientation(
                bid, [bx, by, _TABLE_TOP_Z + _HOLE_HEIGHT / 2], quat,
                physicsClientId=self.client,
            )
        self._hole_xy = np.asarray(ec.hole_xy, dtype=float)
        self._hole_mouth_z = _TABLE_TOP_Z + _HOLE_HEIGHT

        if self.cfg.hole_chamfer:
            self._place_chamfer(cx, cy, bore_inradius)
        else:
            for bid in self.hole_chamfer_ids:
                p.resetBasePositionAndOrientation(bid, [10, 10, -10], [0, 0, 0, 1],
                                                  physicsClientId=self.client)

    def _place_chamfer(self, cx, cy, bore_inradius):
        """cfg.hole_chamfer: boxes forming a tilted funnel above the straight
        bore ring, widening the effective capture opening. Each box pitches
        outward about its own tangential axis (tilt in local frame, then yaw
        to its ring position: q_total = q_yaw * q_pitch) so the funnel wall
        leans away from vertical without twisting. Positioned so the box's
        bottom-inner edge lands exactly at (bore_inradius, mouth_z),
        continuous with the straight ring's top edge with no step for the
        peg to catch on.
        """
        tilt = np.radians(_CHAMFER_TILT_DEG)
        half_z = _CHAMFER_HEIGHT / 2.0
        q_pitch = p.getQuaternionFromEuler([0.0, tilt, 0.0])
        for k, bid in enumerate(self.hole_chamfer_ids):
            ang = 2 * np.pi * k / self._hole_segments
            q_yaw = p.getQuaternionFromEuler([0.0, 0.0, ang])
            _, q_total = p.multiplyTransforms([0, 0, 0], q_yaw, [0, 0, 0], q_pitch)
            R = np.array(p.getMatrixFromQuaternion(q_total)).reshape(3, 3)
            local_corner = np.array([-_HOLE_BOX_HALF, 0.0, -half_z])
            world_offset = R @ local_corner
            desired_corner = np.array([
                cx + bore_inradius * np.cos(ang),
                cy + bore_inradius * np.sin(ang),
                self._hole_mouth_z,
            ])
            center = desired_corner - world_offset
            p.resetBasePositionAndOrientation(bid, center.tolist(), q_total,
                                              physicsClientId=self.client)

    def _place_obstacles(self, ec):
        c = self.client
        used = len(ec.obstacles)
        for i, oid in enumerate(self.obstacle_ids):
            if i < used:
                x, y, z, shape, sx, sy, sz = ec.obstacles[i]
                p.resetBasePositionAndOrientation(oid, [x, y, z], [0, 0, 0, 1], physicsClientId=c)
                self.obstacle_shapes[i] = shape
                fric = ec.lateral_friction if ec.lateral_friction is not None else 0.6
                p.changeDynamics(oid, -1, lateralFriction=fric, physicsClientId=c)
                self._obstacle_active = i + 1
            else:
                p.resetBasePositionAndOrientation(oid, [10 + i, 10, -10], [0, 0, 0, 1],
                                                  physicsClientId=c)
        self._obstacle_active = used

    # ------------------------------------------------------------------ #
    # grasp
    # ------------------------------------------------------------------ #
    def _handle_grasp(self, grip_a: float) -> bool:
        if not self.cfg.use_peg:
            return False
        # threshold at -0.2, not 0.5: an untrained policy outputs ~0 for the
        # gripper, and exploration noise around that must be able to trigger
        # a grasp for the behavior to be discovered
        want = grip_a > _GRASP_THRESH
        ee = np.array(p.getLinkState(self.robot_id, _EE_LINK, physicsClientId=self.client)[0])
        peg = np.array(p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.client)[0])
        dist = float(np.linalg.norm(ee - peg))
        attempt = False
        if want and self._grasp_cid is None and dist < self._grasp_trigger:
            attempt = True
            # peg_in_hole grasps near the peg's top (_GRASP_OFFSET_Z), not
            # center, since a zero-offset grasp makes real insert depth
            # physically impossible. align_only is exempted (pure xy/tilt
            # alignment, no depth, so it doesn't need the offset).
            rigid = self.cfg.task == "peg_in_hole" and not self.cfg.align_only
            child_off = _GRASP_OFFSET_Z if rigid else 0.0
            # admissibility gate (rigid only): the preserve-transform weld
            # (see _form_grasp) no longer force-snaps the flange to the
            # peg's top, so only admit a grasp whose flange is already
            # within the configured axial/lateral band of it -- otherwise an
            # ungoverned grasp from the peg's side would weld there
            # permanently. A rejected attempt still counts as a grasp
            # failure, same as the old "too far" case.
            admissible = True
            if rigid:
                pq = np.array(p.getBasePositionAndOrientation(
                    self.peg_id, physicsClientId=self.client)[1])
                peg_axis = np.array(p.getMatrixFromQuaternion(pq)).reshape(3, 3)[:, 2]
                delta = ee - peg
                axial = float(np.dot(delta, peg_axis))
                lateral = float(np.linalg.norm(delta - axial * peg_axis))
                admissible = (self.cfg.grasp_admit_axial_lo <= axial <= self.cfg.grasp_admit_axial_hi
                             and lateral < self.cfg.grasp_admit_lateral)
            if admissible:
                self._grasp_cid = self._form_grasp(child_off)
        elif want and self._grasp_cid is None:
            attempt = True   # tried to close but too far -> counts as a grasp failure
        elif not want and self._grasp_cid is not None:
            p.removeConstraint(self._grasp_cid, physicsClientId=self.client)
            self._grasp_cid = None
        return attempt

    # ------------------------------------------------------------------ #
    # collisions
    # ------------------------------------------------------------------ #
    def _obstacle_collision(self) -> bool:
        return bool(self._obstacle_collision_links())

    def _obstacle_collision_links(self) -> list:
        """Which robot links (+ 'peg') are in contact with an active obstacle
        right now. Returned via env.step()'s info dict, not queried
        afterward, since collision-terminal episodes get auto-reset by the
        VecEnv wrapper the instant they end -- a query after venv.step()
        returns would see the next episode's already-reset world."""
        if self._obstacle_active == 0:
            return []
        hit = set()
        for i in range(self._obstacle_active):
            oid = self.obstacle_ids[i]
            for c in p.getContactPoints(bodyA=self.robot_id, bodyB=oid, physicsClientId=self.client):
                link_idx = c[3]
                hit.add("base" if link_idx < 0 else f"link{link_idx + 1}")
            if self.cfg.use_peg and p.getContactPoints(bodyA=self.peg_id, bodyB=oid,
                                                       physicsClientId=self.client):
                hit.add("peg(held object)")
        return sorted(hit)

    # ------------------------------------------------------------------ #
    # observation / state
    # ------------------------------------------------------------------ #
    def _state_dict(self, grasp_attempt: bool = False) -> dict:
        c = self.client
        js = p.getJointStates(self.robot_id, list(range(ARM_DOF)), physicsClientId=c)
        q = np.array([j[0] for j in js])
        qd = np.array([j[1] for j in js])
        ft = np.array(js[_EE_LINK][2][:6]) if len(js[_EE_LINK][2]) >= 6 else np.zeros(6)

        ls = p.getLinkState(self.robot_id, _EE_LINK, computeLinkVelocity=1, physicsClientId=c)
        ee_pos = np.array(ls[0]); ee_quat = np.array(ls[1])
        ee_lin = np.array(ls[6]); ee_ang = np.array(ls[7])

        if self.cfg.use_peg:
            pp, pq = p.getBasePositionAndOrientation(self.peg_id, physicsClientId=c)
            pv = p.getBaseVelocity(self.peg_id, physicsClientId=c)[0]
            peg_pos = np.array(pp); peg_quat = np.array(pq); peg_vel = np.array(pv)
        else:
            peg_pos = ee_pos.copy(); peg_quat = np.array([0, 0, 0, 1.0]); peg_vel = np.zeros(3)

        grasped = self._grasp_cid is not None

        # goal per task
        if self.cfg.task == "reach":
            goal = self._cur.reach_target
        elif self.cfg.task == "pick_place":
            goal = self._cur.place_target
        else:
            goal = np.array([self._hole_xy[0], self._hole_xy[1], self._hole_mouth_z])

        # peg axis (body +z) in world -> tilt from vertical, depth into bore
        R = np.array(p.getMatrixFromQuaternion(peg_quat)).reshape(3, 3)
        peg_axis = R[:, 2]
        tilt = float(np.arccos(np.clip(abs(peg_axis[2]), 0.0, 1.0)))
        xy_err = float(np.linalg.norm(peg_pos[:2] - self._hole_xy))
        depth = 0.0
        if self.cfg.use_hole and xy_err < (_PEG_RADIUS + 0.02):
            depth = max(0.0, self._hole_mouth_z - float(peg_pos[2]))
        # reward-shaping-only variant tracking the peg TIP, not its center,
        # with a wider xy gate: the center-based `depth` above only crosses
        # the mouth plane after the tip has already penetrated half the peg's
        # height, so the shaping potential paid nothing for the rim-crossing
        # itself. `peg_depth`/success below stays center-based, unchanged.
        depth_shaped = 0.0
        if self.cfg.use_hole and xy_err < (_PEG_RADIUS + 0.05):
            tip_z = float(peg_pos[2]) - _PEG_HEIGHT / 2.0
            depth_shaped = max(0.0, self._hole_mouth_z - tip_z)

        # keypoint "docking" reward geometry (rewards.py's keypoint_w,
        # default 0.0 = off): K points on the peg's bottom rim vs K points
        # on the bore's inner wall at the mouth. None for non-peg_in_hole
        # tasks, so rewards.py's s.get("peg_keypoints") no-ops there.
        peg_keypoints = hole_keypoints = None
        if self.cfg.task == "peg_in_hole":
            K = self.cfg.keypoint_n
            ang_k = 2 * np.pi * np.arange(K) / K
            peg_local = np.stack([_PEG_RADIUS * np.cos(ang_k), _PEG_RADIUS * np.sin(ang_k),
                                  np.full(K, -_PEG_HEIGHT / 2)], axis=1)
            peg_keypoints = peg_pos + peg_local @ R.T
            hole_keypoints = np.stack([
                self._hole_xy[0] + self._bore_inradius * np.cos(ang_k),
                self._hole_xy[1] + self._bore_inradius * np.sin(ang_k),
                np.full(K, self._hole_mouth_z),
            ], axis=1)

        # all-link, not EE-only: the forearm/elbow can clip an obstacle
        # while the EE itself is clear
        link_states = p.getLinkStates(self.robot_id, list(range(ARM_DOF)), physicsClientId=c)
        link_positions = [np.array(ls[0]) for ls in link_states]
        obstacle_check_points = link_positions + ([peg_pos] if self.cfg.use_peg else [])
        # _EE_LINK is the dominant collision contributor by a wide margin,
        # so give it extra virtual clearance in the shaping-only distance
        # signal (physical collision geometry stays untouched)
        extra_radii = [0.0] * len(link_positions) + ([0.0] if self.cfg.use_peg else [])
        extra_radii[_EE_LINK] = _EE_VIRTUAL_INFLATE
        min_obstacle_dist = self._min_obstacle_dist(obstacle_check_points, extra_radii)
        # the forearm (links 4/5), not the flange, is most of the remaining
        # collision rate near the hole -- dense shaping-only proximity
        # signal for just those two links against the hole ring
        min_ring_dist = self._min_ring_dist([link_positions[3], link_positions[4]])

        # peg-vs-ring contact: lets the policy tell "resting on the rim"
        # apart from "entering the bore". Force is clipped to a sane
        # physical scale (50N) before it reaches the obs assembly, since an
        # ungrasped peg driven by a rigid, uncapped motor can produce a
        # stiff-contact force spike that would otherwise dominate this
        # dimension's running-normalization variance.
        ring_contact = 0.0
        ring_contact_force = 0.0
        if self.cfg.task == "peg_in_hole":
            for bid in self.hole_box_ids + self.hole_chamfer_ids:
                pts = p.getContactPoints(bodyA=self.peg_id, bodyB=bid, physicsClientId=c)
                if pts:
                    ring_contact = 1.0
                    ring_contact_force = float(max(pt[9] for pt in pts))
                    break

        # peg-tip (not center) to hole-mouth vector -- xy alignment + depth
        # in one signal, unlike the center-based `goal_pos - peg_pos` above
        peg_tip = peg_pos - peg_axis * (_PEG_HEIGHT / 2.0)
        peg_tip_to_hole = np.array([
            self._hole_xy[0] - peg_tip[0],
            self._hole_xy[1] - peg_tip[1],
            self._hole_mouth_z - peg_tip[2],
        ])

        at_limit = bool(
            np.any(q < self._q_lower + _JOINT_LIMIT_MARGIN * (self._q_upper - self._q_lower))
            or np.any(q > self._q_upper - _JOINT_LIMIT_MARGIN * (self._q_upper - self._q_lower))
        )

        return dict(
            q=q, qd=qd, ft=ft,
            ee_pos=ee_pos, ee_quat=ee_quat, ee_lin=ee_lin, ee_ang=ee_ang,
            peg_pos=peg_pos, peg_quat=peg_quat, peg_vel=peg_vel,
            grasped=grasped, gripper_cmd=self._last_gripper, grasp_attempt=grasp_attempt,
            goal_pos=np.asarray(goal, dtype=float),
            hole_xy=self._hole_xy, hole_mouth_z=self._hole_mouth_z,
            peg_keypoints=peg_keypoints, hole_keypoints=hole_keypoints,
            peg_depth=depth, peg_depth_shaped=depth_shaped,
            peg_tilt_rad=tilt, peg_speed=float(np.linalg.norm(peg_vel)),
            min_obstacle_dist=min_obstacle_dist,
            min_ring_dist=min_ring_dist,
            at_joint_limit=at_limit,
            ring_contact=ring_contact, ring_contact_force=ring_contact_force,
            peg_tip_to_hole=peg_tip_to_hole,
        )

    def _min_obstacle_dist(self, points, extra_radii=None) -> float:
        """Min approx surface distance from any of `points` (world xyz) to any
        currently-active obstacle. `extra_radii` (optional, same length as
        `points`) adds a per-point virtual bounding-radius inflation,
        shaping-only, e.g. to give a particular link (see
        _EE_VIRTUAL_INFLATE) an earlier dense repulsion warning. Used only
        for reward-side proximity shaping (Phase 3+); `inf` if no active
        obstacles."""
        active = getattr(self, "_obstacle_active", 0)
        if active == 0:
            return float("inf")
        best = float("inf")
        for i in range(active):
            pos = np.array(p.getBasePositionAndOrientation(
                self.obstacle_ids[i], physicsClientId=self.client)[0])
            for j, pt in enumerate(points):
                extra = extra_radii[j] if extra_radii is not None else 0.0
                d = float(np.linalg.norm(np.asarray(pt) - pos)) - _OBSTACLE_BOUND_R - extra
                if d < best:
                    best = d
        return max(0.0, best)

    def _min_ring_dist(self, points) -> float:
        """Min approx surface distance from `points` (world xyz) to the hole
        ring's polygon box array (self.hole_box_ids) -- same crude
        euclidean-minus-bounding-radius approximation as
        _min_obstacle_dist, using the box half-diagonal as the bounding
        radius. peg_in_hole only (use_hole); inf otherwise. Feeds a
        targeted forearm-repulsion shaping term (see rewards.py), since the
        forearm (links 4/5), not the flange, causes most collisions here."""
        if not self.cfg.use_hole:
            return float("inf")
        box_r = _HOLE_BOX_HALF * 1.414
        best = float("inf")
        for bid in self.hole_box_ids:
            pos = np.array(p.getBasePositionAndOrientation(bid, physicsClientId=self.client)[0])
            for pt in points:
                d = float(np.linalg.norm(np.asarray(pt) - pos)) - box_r
                if d < best:
                    best = d
        return max(0.0, best)

    def _obstacle_features(self, ee_pos: np.ndarray) -> np.ndarray:
        """K nearest obstacles: rel xyz(3) + halfextent(3) + type onehot(2) + valid(1)."""
        feats = []
        active = getattr(self, "_obstacle_active", 0)
        items = []
        for i in range(active):
            pos = np.array(p.getBasePositionAndOrientation(self.obstacle_ids[i],
                                                           physicsClientId=self.client)[0])
            x, y, z, shape, sx, sy, sz = self._cur.obstacles[i]
            d = np.linalg.norm(pos - ee_pos)
            items.append((d, pos - ee_pos, np.array([sx, sy, sz]),
                          np.array([1.0, 0.0]) if shape == "box" else np.array([0.0, 1.0])))
        items.sort(key=lambda t: t[0])
        for k in range(MAX_OBSTACLES_IN_OBS):
            if k < len(items):
                _, rel, ext, onehot = items[k]
                feats.append(np.concatenate([rel, ext, onehot, [1.0]]))
            else:
                feats.append(np.zeros(9))
        return np.concatenate(feats)

    def _assemble_obs(self, s: Optional[dict] = None, dry_run: bool = False) -> np.ndarray:
        if dry_run:
            self._obstacle_active = 0
            self._cur = self.sampler.sample(0)
            self._hole_xy = np.array([0.45, 0.0]); self._hole_mouth_z = _HOLE_HEIGHT
            self._bore_inradius = _PEG_RADIUS + 0.003
            self._last_gripper = 0.0; self._grasp_cid = None
            s = self._state_dict()
        if s is None:
            s = self._state_dict()

        stage_onehot = np.zeros(4)
        stage_onehot[min(self.cfg.phase, 4) - 1] = 1.0

        # H3's 8 extra dims/frame (peg_tilt_rad, min_ring_dist,
        # peg_depth_shaped, peg_tip_to_hole, ring_contact,
        # ring_contact_force) are no longer appended here -- they broke
        # observation-shape compatibility with every checkpoint trained
        # before H3, including the report's adopted ones. Moved to
        # obs_h3_extras.h3_extra_dims(); not called by default. This
        # restores the original 123-dims/frame observation.
        parts = [
            s["q"], s["qd"],
            s["ee_pos"], _rot6d(s["ee_quat"]), s["ee_lin"], s["ee_ang"],
            [s["gripper_cmd"]], [1.0 if s["grasped"] else 0.0],
            s["peg_pos"], _rot6d(s["peg_quat"]), s["peg_pos"] - s["ee_pos"], s["peg_vel"],
            s["goal_pos"], _rot6d(np.array([0, 0, 0, 1.0])), s["goal_pos"] - s["peg_pos"],
            [s["peg_depth"]],
            s["ft"],
            self._obstacle_features(s["ee_pos"]),
            stage_onehot,
        ]
        return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in parts]).astype(np.float32)
