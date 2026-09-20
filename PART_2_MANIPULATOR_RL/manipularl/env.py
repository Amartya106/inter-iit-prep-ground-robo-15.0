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
# Stage-1 chamfer (cfg.hole_chamfer, see E55/E56 in EXPERIMENTS.md): a funnel
# lead-in above the straight bore. 35deg from vertical, 3cm tall -- widens the
# effective capture opening from the bore's ~3.6cm (peg 3cm dia + 6mm
# clearance) to roughly 3.6cm + 2*3cm*tan(35deg) =~ 7.8cm at the chamfer's own
# top, the standard fix for "aligned peg lands on the rim instead of sliding
# in" (E55's diagnosis: 26.5% of oracle episodes stalled exactly this way).
_CHAMFER_HEIGHT = 0.03
_CHAMFER_TILT_DEG = 35.0
_FINE_CONTROL_RADIUS = 0.05   # cfg.fine_control_frac's taper zone, see its docstring
_GRASP_TRIGGER = 0.09    # forgiving: a policy that hovers near the peg and
                         # closes the gripper should actually grasp, so the
                         # dense grasp-shaping and the grasp gate coincide
_GRASP_THRESH = -0.2     # gripper-signal threshold to *attempt* a grasp
_AIRSPACE_LOW_M = 0.03   # Task 1b/"descend" (descend_only, _start_pre_airspace):
                         # random start height above the hole mouth is drawn
                         # uniformly in [_AIRSPACE_LOW_M, _AIRSPACE_HIGH_M] --
                         # low bound keeps some real descent distance even in
                         # the easy case (not indistinguishable from insert_
                         # only's near-zero start).
_AIRSPACE_HIGH_M = 0.15  # high bound -- generous but still well within the
                         # arm's normal reach, same order of magnitude as
                         # _start_pre_aligned's fixed 8cm reference point and
                         # rewards.py's _HOVER_HEIGHT (10cm).
_GRASP_OFFSET_Z = 0.035  # peg_in_hole ONLY: how far above the peg's own
                         # center the grasp constraint attaches (both the
                         # real _handle_grasp and the curriculum teleports),
                         # i.e. "grasp near the peg's top" instead of "grasp
                         # at its center" (offset 0, used everywhere else --
                         # Phase 1/2/3 keep the original zero-offset grasp
                         # unchanged). Found via direct verification: the
                         # hole ring (manipularl/env.py's polygon box ring)
                         # is SOLID from the table (z=0) up to the mouth
                         # (z=hole_mouth_z) -- far too wide for the flange to
                         # ever fit through. With a zero-offset grasp, the
                         # peg's center = the flange's own position, so for
                         # the peg to reach insert_success_depth (3cm) below
                         # the mouth, the FLANGE itself would ALSO have to be
                         # 3cm below the mouth -- i.e. physically inside
                         # solid ring material (confirmed empirically: 13
                         # simultaneous robot-vs-ring contact points,
                         # reproducible at every hole position tried). This
                         # is very likely the true root cause behind
                         # max_depth landing at EXACTLY 0.00cm across every
                         # independent Phase-4 attempt so far
                         # Grasp-admissibility band (peg_in_hole, non-
                         # align_only only -- see _form_grasp's `rigid`
                         # branch docstring): PhaseConfig.grasp_admit_axial_lo/
                         # hi/lateral (configs.py), not module constants --
                         # made configurable so a widened band can be tested
                         # as an opt-in without touching this file again. A
                         # real cold grasp used to accept ANY flange pose within
                         # _GRASP_TRIGGER (9cm) of the peg center, then
                         # force-snap it to (0,0,_GRASP_OFFSET_Z)+identity --
                         # teleporting the peg and re-orienting it to the
                         # flange's arbitrary orientation regardless of where
                         # that flange actually was (root cause of E56's
                         # 20-40deg tilt and the E31 GRASP<->DROP flicker,
                         # both a symptom of PyBullet's JOINT_FIXED forcing
                         # child_orn == parent_orn for ANY childFramePosition,
                         # not just a nonzero one). Once the weld instead
                         # preserves the REAL relative transform (no
                         # teleport), a grasp attempted from the peg's side or
                         # underside would weld the flange there permanently,
                         # which both looks physically wrong and can make
                         # insert_success_depth unreachable (the original,
                         # legitimate reason _GRASP_OFFSET_Z existed) -- so
                         # real cold grasps additionally require the flange to
                         # already be within this band of the peg's own top
                         # (axial, along the peg's local +z) and centered
                         # over it (lateral). Centered on _GRASP_OFFSET_Z
                         # (0.035) with the peg's radius (0.015) as slack.
                         # Teleport curricula never consult this -- they
                         # pre-place the peg at exactly (0,0,_GRASP_OFFSET_Z)
                         # before welding, always admissible by construction.
                         # (E10/E16/E17/E28/E30/Track-B) -- a scene-geometry
                         # impossibility, not an exploration/curriculum/
                         # reward-shaping problem. Grasping 3.5cm above the
                         # peg's center (peg height 8cm, so up to 4cm is
                         # physically available) instead gives the flange a
                         # ~0.5cm clearance margin above the mouth at the
                         # real success depth.
_MAX_OBSTACLE_POOL = 8            # >= max of any phase's obstacle_max
# obstacle collision bodies are fixed-size boxes (see _build_scene:
# createCollisionShape(GEOM_BOX, halfExtents=[0.03,0.03,0.05])), just
# repositioned per episode -- the (sx,sy,sz) in EpisodeConfig.obstacles is
# observation-only. This is the matching bounding-sphere radius, used for
# reward-side proximity shaping (min_obstacle_dist).
_OBSTACLE_HALF = np.array([0.03, 0.03, 0.05])
_OBSTACLE_BOUND_R = float(np.linalg.norm(_OBSTACLE_HALF))
_EE_VIRTUAL_INFLATE = 0.04   # m, extra VIRTUAL bounding radius added to
                             # _EE_LINK only in the shaping-side obstacle
                             # distance (_min_obstacle_dist) -- not the
                             # physical collision geometry. Motivated by
                             # E24's diagnostic: _EE_LINK ("link7") is the
                             # dominant collision contributor (63% of
                             # collision-steps), so give the dense repulsion
                             # penalty an earlier/stronger warning for it.
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
        # fraction of TRAIN episodes (peg_in_hole only) that start with the
        # peg already grasped AND positioned above the hole mouth -- mirrors
        # grasp_curriculum one stage further down the task chain, see
        # _start_pre_aligned().
        self.align_curriculum = float(align_curriculum or 0.0)
        # fraction of TRAIN episodes (peg_in_hole only) that start with the
        # peg already INSERTED past the success depth -- one stage further
        # than align_curriculum. E16/E17 showed align_curriculum alone
        # doesn't help (alignment was never the bottleneck -- see
        # diagnose_phase4 findings); this targets the actual gap: the policy
        # has never experienced holding a peg that's already through the
        # bore, so it never discovers the push-through motion. See
        # _start_pre_inserted().
        self.insert_curriculum = float(insert_curriculum or 0.0)
        # Task-2/"insert-only" (cfg.insert_only): every episode starts already
        # in the insertion zone (see _start_pre_inserted); this controls HOW
        # MUCH depth is already given for free at reset, 1.0=easiest (starts
        # past insert_success_depth) -> 0.0=hardest (starts at depth exactly
        # 0, the real task). Annealed 1.0->0.0 by InsertStartDepthAnneal.
        # Only consulted for split=="train" -- eval always forces 0.0 (see
        # reset()), so eval numbers always reflect the real, undiluted skill.
        self.insert_start_depth_frac = 1.0
        # Task-1b/"descend" (cfg.descend_only): how much of the airspace
        # column's height range _start_pre_airspace() is allowed to sample
        # from. 0.0=easiest (height always ~_AIRSPACE_LOW_M, a trivial short
        # descent) -> 1.0=hardest (full [_AIRSPACE_LOW_M, _AIRSPACE_HIGH_M]
        # range, the real task). Opposite convention from
        # insert_start_depth_frac (1.0=easy there) because this anneals a
        # DIFFICULTY range up, not a free-credit amount down -- matches
        # ObstacleCountCurriculum/GraspDropPenaltyAnneal's up-ramping
        # pattern instead. Annealed 0.0->1.0 by AirspaceHeightAnneal. Only
        # consulted for split=="train" -- eval always forces 1.0 (see
        # reset()), so eval numbers always reflect the real, full-range
        # skill, never the training crutch.
        self.airspace_height_frac = 1.0
        # Task-1b/"descend" replay-seeded curriculum (see _start_from_replay):
        # fraction of TRAIN episodes that start from an exact recorded state
        # from a real, proven-successful E33 rollout instead of a randomized
        # airspace height. Only loaded (I/O) if replay_path is actually
        # given -- every other config pays nothing for this. eval never uses
        # replay starts (see reset()) -- eval numbers reflect the real skill
        # under the normal randomized-height distribution, not the crutch.
        self.replay_curriculum = float(replay_curriculum or 0.0)
        self._replay = None
        if replay_path is not None:
            import numpy as _np
            self._replay = _np.load(replay_path)
        # Task-2/"insert-only" robustness fix: random xy offset (meters,
        # uniform over a disk of this radius) and tilt (degrees, uniform in
        # [-x, x]) applied to _start_pre_inserted()'s teleport target, for
        # split=="train" ONLY (eval stays exactly precise, unchanged from
        # E33/E37, so eval numbers remain comparable). Motivation: Task 2
        # was only ever trained from near-exact alignment (~0.1-1cm off);
        # the composed-eval experiments (E38/E39) found it cannot tolerate
        # more than ~1.5cm of realistic hand-off misalignment (produces
        # exactly 0.00000 depth beyond that), even though the underlying
        # push-through skill exists (best-case depth 6.24cm from an exact
        # start, E33's diagnostic). A fixed (non-curriculum) perturbation
        # magnitude for this pass, not annealed.
        self.insert_xy_jitter = float(insert_xy_jitter or 0.0)
        self.insert_tilt_jitter_deg = float(insert_tilt_jitter_deg or 0.0)
        # EE-peg catch radius for a gripper-close to actually form the grasp
        # constraint (see _handle_grasp). Default = the real _GRASP_TRIGGER;
        # only ever widened by GraspTriggerCurriculum for split=="train", so
        # eval always sees the real value.
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
        # arXiv:2402.02665 utility-based-RL reformulations (rewards.py's
        # satisficing_proximity/risk_seeking_depth) -- both default off
        # (RewardComputer.__init__ already sets False); only overridden when
        # a config explicitly opts in, same disclosed-exception pattern as
        # every other reward_fn.* override in this block.
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
        """Live-update the pre-grasped-start fraction (used by the anneal
        callback via VecEnv.env_method). No effect for split != 'train'."""
        self.grasp_curriculum = float(max(0.0, min(1.0, frac)))

    def set_align_curriculum(self, frac: float):
        """Live-update the pre-aligned-start fraction (peg_in_hole only, used
        by AlignCurriculumAnneal via VecEnv.env_method)."""
        self.align_curriculum = float(max(0.0, min(1.0, frac)))

    def set_insert_curriculum(self, frac: float):
        """Live-update the pre-inserted-start fraction (peg_in_hole only,
        used by InsertDepthCurriculum via VecEnv.env_method)."""
        self.insert_curriculum = float(max(0.0, min(1.0, frac)))

    def set_insert_start_depth_frac(self, frac: float):
        """Live-update how much starting depth Task 2 (cfg.insert_only) gives
        for free at reset, 1.0 (easiest) -> 0.0 (hardest/real) -- used by
        InsertStartDepthAnneal via VecEnv.env_method. No effect unless
        cfg.insert_only, and never applied to split=='eval' (see reset())."""
        self.insert_start_depth_frac = float(max(0.0, min(1.0, frac)))

    def set_airspace_height_frac(self, frac: float):
        """Live-update how much of the airspace column's height range Task
        1b (cfg.descend_only) samples from at reset, 0.0 (easiest, height
        ~_AIRSPACE_LOW_M) -> 1.0 (hardest/real, full range) -- used by
        AirspaceHeightAnneal via VecEnv.env_method. No effect unless
        cfg.descend_only, and never applied to split=='eval' (see
        _start_pre_airspace / reset())."""
        self.airspace_height_frac = float(max(0.0, min(1.0, frac)))

    def set_grasp_trigger(self, radius: float):
        """Live-update the EE-peg catch radius for a grasp attempt to
        succeed (used by GraspTriggerCurriculum via VecEnv.env_method).
        Clamped to [_GRASP_TRIGGER, 0.20] -- never tighter than the real
        value, never absurdly wide."""
        self._grasp_trigger = float(max(_GRASP_TRIGGER, min(0.20, radius)))

    def set_obstacle_range(self, lo: int, hi: int):
        """Live-update the per-episode obstacle-count range (used by the
        obstacle-count curriculum via VecEnv.env_method). Delegates to the
        sampler; no effect if this phase has num_obstacles == 0."""
        self.sampler.set_obstacle_range(int(lo), int(hi))

    def set_xy_err_weight(self, w: float):
        """Live-update the dense potential-shaping's xy_err coefficient
        (used by XyErrWeightAnneal via VecEnv.env_method) -- softens the
        pull-toward-the-hole gradient from a higher starting value down to
        1.0 as training progresses. Delegates to the reward function
        (RewardComputer.xy_err_weight); see its docstring for the
        motivation (reducing overshoot/jitter right at the target)."""
        self.reward_fn.xy_err_weight = float(w)

    def set_tilt_weight(self, w: float):
        """Live-update the dense potential-shaping's tilt coefficient (usable
        by SuccessGatedAnneal via VecEnv.env_method). See
        RewardComputer.tilt_weight's docstring -- E56's tilt-lock diagnosis."""
        self.reward_fn.tilt_weight = float(w)

    def set_keypoint_weight(self, w: float):
        """Live-update the keypoint-docking potential's coefficient (usable
        by SuccessGatedAnneal via VecEnv.env_method). See
        RewardComputer.keypoint_w's docstring. Delegates to the reward
        function (RewardComputer.keypoint_w); default 0.0 = off."""
        self.reward_fn.keypoint_w = float(w)

    def set_grasp_drop_pen(self, w: float):
        """Live-update the grasp-drop penalty magnitude (used by
        GraspDropPenaltyAnneal via VecEnv.env_method) -- ramps it up from a
        smaller starting value instead of applying full strength from step
        0, on the hypothesis that a harsh, ever-present drop cost during
        early exploration is itself part of what drives grasp-shyness.
        Delegates to the reward function (RewardComputer.grasp_drop_pen)."""
        self.reward_fn.grasp_drop_pen = float(w)

    def set_collision_pen_mult(self, w: float):
        """Live-update the collision-penalty multiplier (used by
        CollisionCurriculum via VecEnv.env_method) -- ramps the per-step
        collision penalty up from a softer starting value. Delegates to
        RewardComputer.collision_pen_mult."""
        self.reward_fn.collision_pen_mult = float(w)

    def set_collision_terminal(self, flag: bool):
        """Live-toggle whether a collision ALSO ends the episode
        immediately, independent of cfg.collision_is_failure (which stays
        fixed -- collision always precludes success). Used by
        CollisionCurriculum to mirror Phase 3's own real history: no early
        termination while the policy is still learning the underlying
        approach/grasp/carry skill, switched on once training has
        progressed far enough to expect that skill already exists.
        Delegates to RewardComputer.collision_terminal."""
        self.reward_fn.collision_terminal = bool(flag)

    def set_replay_curriculum(self, frac: float):
        """Live-update the fraction of TRAIN episodes that start from a
        recorded replay snapshot (see _start_from_replay). No effect unless
        cfg.descend_only and replay_path was given at construction; never
        applied to split=='eval'."""
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

        # hole socket: a ring of thin static boxes forming the bore wall --
        # a regular polygon with cfg.hole_segments sides (8 = octagon,
        # every existing config/checkpoint's shape; more sides approximate
        # a circle more closely, see cfg.hole_segments's docstring).
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

        # Stage-1 chamfer ring (cfg.hole_chamfer): a second, TILTED ring of 8
        # static boxes above the straight bore, forming a funnel lead-in.
        # See _place_hole's chamfer branch for the geometry. Created here
        # (like every other persistent body in this scene) so toggling the
        # config flag never touches PyBullet body count/reset cost -- parked
        # at [10,10,-10] whenever hole_chamfer is False, exactly like the
        # straight ring is parked when use_hole is False.
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

        # --- grasp curriculum (training only) -------------------------------
        # A fraction of training episodes start with the peg already in hand:
        # PPO then experiences the carry -> release sub-task and its rewards,
        # the value function backs that up, and grasping becomes attractive
        # from the un-grasped state too. Never active for split == "eval".
        if (self.cfg.task == "peg_in_hole" and self.cfg.descend_only
                and self.split == "train" and self._replay is not None
                and self.replay_curriculum > 0.0
                and self._np_random.random() < self.replay_curriculum):
            # Task 1b replay-seeded curriculum: start from an exact recorded
            # state from a real, proven-successful E33 rollout instead of a
            # randomized airspace height -- see _start_from_replay. Never
            # for split=="eval" (checked above), so eval numbers reflect the
            # real skill under the normal randomized-height distribution.
            self._start_from_replay()
        elif self.cfg.task == "peg_in_hole" and self.cfg.descend_only:
            # Task 1b ("descend"): the remaining episodes start already
            # grasped + xy-aligned at a RANDOM height within the airspace
            # column above the hole -- see _start_pre_airspace and
            # PhaseConfig.descend_only.
            self._start_pre_airspace()
        elif self.cfg.task == "peg_in_hole" and self.cfg.insert_only:
            # Task 2 ("insert"): EVERY episode, train and eval, starts already
            # grasped + xy/tilt-aligned in the insertion zone -- this is the
            # whole point of the task (see PhaseConfig.insert_only docstring).
            # Train reads the curriculum-controlled starting depth; eval
            # always forces the hardest/real setting (depth=0) regardless of
            # wherever the training curriculum's anneal left it, so eval
            # numbers are never flattered by an easy starting depth.
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
        """Create the EE<->peg JOINT_FIXED constraint, applying cfg.grasp_max_force
        (Stage-1 fix, see PhaseConfig.grasp_max_force's docstring) if set.

        Every grasp in this file -- real (`_handle_grasp`) and every teleport
        curriculum -- goes through here, so the finite-force option applies
        uniformly regardless of how the grasp was formed. `changeConstraint`'s
        `maxForce` caps the constraint solver's corrective force each step;
        below that cap the peg can lag/sag/deflect relative to the flange
        instead of being perfectly rigid, letting it self-centre against the
        chamfer (Stage 1b) rather than fighting it. `None` (default) leaves
        PyBullet's constraint at its default (effectively rigid) max force --
        bit-for-bit the old behaviour.

        `child_off == 0.0` (phases 1-3, align_only): UNCHANGED original weld
        -- childFramePosition=[0,0,0], both frame orientations left at
        PyBullet's identity default. Harmless here specifically because
        these tasks' reward has no tilt term, so the peg being forced to the
        flange's world orientation (see below) is never graded.

        `child_off > 0.0` (peg_in_hole, non-align_only): a **preserve-
        transform** weld instead of the old hardcoded-offset one. Root cause
        this fixes: `JOINT_FIXED` with default (identity) frame orientations
        constrains `child_orn == parent_orn` regardless of
        `childFramePosition` -- so the old `[0,0,child_off]` weld ALSO
        force-rotated the peg to the flange's arbitrary orientation at the
        instant of every real (non-curriculum) grasp, since nothing in
        `_handle_grasp` ever controlled flange orientation. That is the
        mechanism behind E56's measured 20-40deg grasp tilt and the E31
        GRASP<->DROP flicker (289/289 failures) that first appeared when
        this offset was introduced. Curriculum teleports (_start_pre_grasped
        etc.) never hit this: they IK the flange to an identity quat and
        place it at exactly `peg_pos + [0,0,child_off]` BEFORE calling this,
        so for them the transform computed below is bit-identical to the
        old hardcoded one. Only a real cold grasp, where the flange's
        orientation was never controlled, changes behaviour.
        `_handle_grasp` gates entry into this branch with an admissibility
        check (`_GRASP_ADMIT_*`) so the preserved transform still keeps the
        flange near the peg's top -- without that, an arbitrary preserved
        transform could weld the flange level with the peg centre and make
        insert_success_depth physically unreachable again, which is the
        original, legitimate reason `_GRASP_OFFSET_Z` existed.
        """
        if child_off > 0.0:
            fl_pos, fl_orn = p.getLinkState(self.robot_id, _EE_LINK, computeForwardKinematics=True,
                                            physicsClientId=self.client)[:2]
            pg_pos, pg_orn = p.getBasePositionAndOrientation(self.peg_id,
                                                              physicsClientId=self.client)
            # childFrame must satisfy flange_pose == peg_pose ∘ childFrame
            # (parentFrame is identity, so the constraint pins the flange's
            # own pose to the peg's pose composed with childFrame) -- solve
            # by inverting the PEG's pose and composing with the FLANGE's,
            # i.e. childFrame = peg_pose^-1 ∘ flange_pose. Verified against
            # manual matrix composition and an isolated two-body PyBullet
            # scene during development; the reversed form (invert flange,
            # compose with peg) looks equally plausible on paper but is
            # wrong -- caught by test_grasp_weld_preserves_transform's
            # getConstraintInfo check, not by inspection.
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
            # peg_in_hole: same top-offset grasp as _handle_grasp (see
            # _GRASP_OFFSET_Z) -- Phase 1/2/3 keep the original zero offset.
            # align_only (Task 1) ALSO stays zero-offset: its success is
            # pure xy/tilt alignment, never depth, so it never needed the
            # offset -- giving it anyway regressed Phase 3's proven grasp-
            # hold skill into a GRASP<->DROP flicker (E31/E32/E34, 0/200
            # success each). See PhaseConfig.align_only's docstring.
            child_off = (_GRASP_OFFSET_Z if self.cfg.task == "peg_in_hole"
                        and not self.cfg.align_only else 0.0)
            self._grasp_cid = self._form_grasp(child_off)
            self.reward_fn._grasp_bonus_paid = True   # don't pay the gate for a free grasp
            self.reward_fn._picked = True

    def _ik_limited(self, target_pos, target_quat):
        """calculateInverseKinematics, converged, then clipped to joint limits.

        WHY limit-awareness at all: the plain unbounded IK call can and does
        return solutions violating the URDF's joint limits (confirmed:
        _start_pre_aligned's landing config had joint 5 at -2.78 rad against a
        -2.09 rad lower limit). `resetJointState` applies it anyway (no limit
        check), and once POSITION_CONTROL motors take over next step the
        engine's limit enforcement fights the motor command, producing violent
        oscillation (traced: tilt 3->89 degrees, xy drift to 15+cm, within ~5
        steps, even for a trivial "hold position" target).

        WHY NOT PyBullet's null-space form: this used to pass
        lowerLimits/upperLimits/jointRanges/restPoses and nothing else. Measured
        directly (200-iteration sweep over the whole hole_xy sampling range, at
        the flange height insertion actually needs, z=0.105):

            null-space form : 14-36cm position error, AND 1-2 joints still
                              outside their limits
            plain IK, 200 it:  0.00cm error, 0.0deg tilt, every hole position

        It failed at BOTH of its jobs. The restPoses null-space bias pulls the
        solution back toward _REST_POSE, which is nowhere near the low, folded
        configuration insertion requires, and the solver settles there -- more
        iterations do not help (bit-identical at 20 and 200). It also omitted
        maxNumIterations entirely, so it ran PyBullet's 20-iteration default.

        Consequence of the old behaviour, since every peg_in_hole teleport
        curriculum routes through here (_start_pre_aligned, _start_pre_inserted,
        _start_pre_airspace): _start_pre_inserted targets flange z=0.100 and
        landed it at 0.1565 on EVERY reset -- a systematic 5.65cm shortfall
        putting the peg 2.15cm ABOVE the hole mouth when it was supposed to
        start 3.5cm BELOW it. The "start already inserted" task never started
        inserted, in train or eval.

        The correct fix is to converge properly and enforce the limits
        explicitly rather than hoping a null-space bias does it implicitly.
        """
        jt = p.calculateInverseKinematics(
            self.robot_id, _EE_LINK, list(target_pos), list(target_quat),
            maxNumIterations=200, residualThreshold=1e-5,
            physicsClientId=self.client)
        return np.clip(np.asarray(jt[:ARM_DOF], dtype=float),
                       self._q_lower, self._q_upper)

    def _ik_converge(self, target, quat, iters=80, seed_rest=True):
        """One IK-teleport attempt: (optionally) seed from _REST_POSE, then
        iterate `iters` steps of _ik_limited + resetJointState (the mechanics
        every peg_in_hole curriculum teleport uses), returning the achieved
        flange position.

        `iters=80`, not the pre-fix code's 40: measured directly (20 real
        hole_xy from actual resets, insert_only depth_frac=0 target) -- 40
        outer iterations left ~half the samples with 6-9cm residual error,
        while 60 converged all 20 to 0.00cm. Each outer iteration reseeds
        `_ik_limited`'s OWN 200-iteration internal solve from the previous
        iteration's result; this only pays off once the null-space bias
        (which made outer iteration count irrelevant -- "bit-identical at 20
        and 200", see _ik_limited's docstring) is gone. 80 keeps a margin
        above the empirical 60-iteration threshold.

        `seed_rest` matters more than it looks. Found via direct measurement,
        downstream of the _ik_limited fix above: _start_pre_inserted's first
        attempt never reset to _REST_POSE -- it seeded from whatever joint
        state happened to be left over (the previous episode's final pose).
        For this redundant 7-DOF arm under an orientation constraint, a bad
        seed can converge to a genuinely different LOCAL minimum -- one with
        small XY residual but real Z residual (measured: a consistent 2cm Z
        shortfall on insert_only resets, at hole_xy positions where seeding
        from _REST_POSE instead converges to 0.00cm 3D error). The old
        acceptance check only compared `ee_try[:2]` to the target XY, so that
        wrong-Z convergence passed silently -- Z was simply never checked.
        Callers now seed every attempt from _REST_POSE and check the full
        3D position error; see _start_pre_inserted/_start_pre_airspace.
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
        """peg_in_hole curriculum: teleport the arm+peg straight to "grasped,
        held above the hole mouth, roughly vertical" -- one stage further
        down the task chain than _start_pre_grasped(). Bootstraps
        descend->insert->hold before the harder transit+align leg is learned,
        the same way the grasp curriculum bootstrapped carry before reach.
        Uses the peg's own body frame as the grasp target (no dependence on
        wherever the sampler happened to place it) so this is safe to call
        with any peg start position/orientation.
        """
        target_xy = self._hole_xy
        clear_z = self._hole_mouth_z + 0.08
        target = np.array([target_xy[0], target_xy[1], clear_z])
        quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])   # peg axis vertical
        # seed_rest=True (was: whatever joints happened to be left over from
        # before this call) -- see _ik_converge's docstring for why an
        # arbitrary seed can converge to a small-XY/wrong-Z local minimum.
        ee = self._ik_converge(target, quat, iters=80, seed_rest=True)
        peg_pos = ee - np.array([0.0, 0.0, _GRASP_OFFSET_Z])   # same top-offset grasp as a real grasp
        p.resetBasePositionAndOrientation(self.peg_id, peg_pos.tolist(), quat,
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._grasp_cid = self._form_grasp(_GRASP_OFFSET_Z)
        self.reward_fn._grasp_bonus_paid = True
        self.reward_fn._picked = True

    def _start_pre_inserted(self, depth_frac: Optional[float] = None):
        """peg_in_hole curriculum: teleport the arm+peg straight to "grasped,
        held ALREADY PAST the insertion-success depth inside the bore,
        vertical" -- one stage further than _start_pre_aligned(). Diagnostics
        across four independent Phase-4 attempts (E10/E16/E17/E28) all found
        the exact same wall: transit+align already works (grasped episodes
        reliably get the peg within cm of the hole), but max_depth is
        EXACTLY 0.00cm every time -- the policy has never once experienced
        what holding/pushing a peg already through the ~3mm-clearance bore
        feels like, so no reward shaping fixes a motion that's never been
        sampled. This gives it that experience directly, the same way
        _start_pre_grasped/_start_pre_aligned bootstrap the stages before it.
        Same IK-teleport-then-constrain mechanics as _start_pre_aligned,
        just deeper -- safe with any peg start position/orientation.

        `depth_frac` (used by the standalone insert-only Task 2, cfg.insert_only):
        None -> the original Track-A behaviour, always a bit PAST
                insert_success_depth (a guaranteed-successful starting hold).
        1.0  -> same as None (easiest: starts already past success depth).
        0.0  -> starts with depth EXACTLY 0 -- peg tip just touching the hole
                mouth, xy/tilt-aligned, grasped -- "always in the inserting
                zone" but the entire push-through is still left to learn.
        Intermediate values interpolate linearly between the two.
        """
        target_xy = self._hole_xy
        max_depth = float(self.cfg.insert_success_depth) + 0.005   # a bit past
        depth = max_depth if depth_frac is None else max(0.0, min(1.0, depth_frac)) * max_depth
        peg_target_z = self._hole_mouth_z - depth        # where the PEG's center should end up
        # Flange target = peg target + _GRASP_OFFSET_Z (grasp near the peg's
        # top, not its center) -- keeps the flange clear of the solid hole
        # ring even at real insertion depth. See _GRASP_OFFSET_Z's docstring:
        # with the old zero-offset grasp, this line put the FLANGE itself at
        # peg_target_z (inside the ring, confirmed via 13 simultaneous
        # robot-vs-ring contacts) -- a scene-geometry bug that made genuine
        # insertion physically impossible regardless of training.
        flange_target_z = peg_target_z + _GRASP_OFFSET_Z

        # Robustness jitter (Task 2, split=="train" only -- see
        # insert_xy_jitter/insert_tilt_jitter_deg's docstring above). Eval
        # is completely unaffected, so eval numbers stay comparable to
        # E33/E37's precise-start baseline.
        #
        # Found via direct verification: the bounded IK solver occasionally
        # (~20% of resets in a smoke test) converges to a wildly wrong
        # configuration (~17-21cm xy error, far beyond the intended jitter)
        # for certain hole positions once the target is perturbed off-
        # center -- a real, pre-existing IK-convergence fragility this
        # jitter exposes, not something jitter itself causes (unjittered
        # calls to this method don't show it). Retry with a fresh jitter
        # draw up to 3 times, verified against actual EE convergence
        # (not just requested target) before accepting; fall back to the
        # exact unjittered/untilted placement (E33/E37's proven-stable
        # behaviour) if every attempt still lands wrong -- a rare degraded
        # episode is fine, a routinely-broken 20cm-off start is not.
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
            # seed_rest=True on EVERY attempt (including the first) -- see
            # _ik_converge's docstring: a leftover, non-REST_POSE seed can
            # converge to a wrong-Z local minimum that a pre-fix XY-only
            # check accepted silently. Full 3D error checked below now.
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
        # _align_bonus is left to pay naturally on the first compute() call
        # (matches _start_pre_aligned's precedent -- it's a genuine downstream
        # sub-goal the curriculum makes available, not a mechanical artifact
        # of the teleport). The depth-progress-bonus, though, WOULD retroactively
        # pay for the free starting depth as if it were newly-achieved progress
        # -- pre-set the checkpoint to what the teleport itself already
        # provides so only further real progress earns more.
        from .rewards import _DEPTH_PROGRESS_CAP_CM, _DEPTH_PROGRESS_STEP_CM
        self.reward_fn._depth_checkpoint_cm = min(
            _DEPTH_PROGRESS_CAP_CM, int(depth * 100.0) // _DEPTH_PROGRESS_STEP_CM)

    def _start_pre_airspace(self):
        """peg_in_hole curriculum: Task 1b/"descend" (my 2-subpart
        split of Task 1 -- see PhaseConfig.descend_only). Teleport to
        "grasped, xy/tilt-aligned, at a RANDOM height within the 'airspace'
        column above the hole" -- one stage narrower than _start_pre_aligned
        (which always uses a fixed 8cm clearance) and one stage wider than
        _start_pre_inserted (which starts right at/past the mouth). The
        randomized height is the whole point: the policy has to learn a
        smooth, aligned descent over a variable distance, not just hold a
        fixed pose or complete a fixed few cm of push-through.

        Same IK-teleport-then-constrain-then-retry mechanics as
        _start_pre_inserted (including its 3-attempt jitter-convergence
        fallback, even though this method doesn't itself apply xy/tilt
        jitter -- kept for consistency/robustness, costs nothing when jitter
        is 0). Reuses the real _GRASP_OFFSET_Z grasp (this task cares about
        depth, same as insert_only) and the full, unmodified peg_in_hole
        success/potential (xy+tilt+depth all still active) -- descend_only
        does NOT use align_only's height-free potential, that's the other
        subpart's whole point.
        """
        target_xy = np.array(self._hole_xy, dtype=float)
        # Curriculum: eval always samples the FULL/real range regardless of
        # training's current anneal state (matches insert_only's depth_frac
        # precedent -- eval numbers must reflect the real, undiluted skill).
        frac = self.airspace_height_frac if self.split == "train" else 1.0
        high = _AIRSPACE_LOW_M + frac * (_AIRSPACE_HIGH_M - _AIRSPACE_LOW_M)
        height = self._np_random.uniform(_AIRSPACE_LOW_M, high)
        peg_target_z = self._hole_mouth_z + height
        flange_target_z = peg_target_z + _GRASP_OFFSET_Z
        quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])   # peg axis vertical

        target = np.array([target_xy[0], target_xy[1], flange_target_z])
        # Single attempt is now sufficient -- seed_rest=True makes every
        # attempt deterministic given the same target, so a 3x retry (the
        # pre-fix code's structure, copied from _start_pre_inserted where
        # the target genuinely varies per attempt via jitter) would just
        # repeat the identical solve. See _ik_converge's docstring: this
        # replaces an XY-only acceptance check that could silently accept a
        # converged-but-wrong-Z local minimum.
        ee = self._ik_converge(target, quat, iters=80, seed_rest=True)
        peg_pos = ee - np.array([0.0, 0.0, _GRASP_OFFSET_Z])
        p.resetBasePositionAndOrientation(self.peg_id, peg_pos.tolist(), quat,
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._grasp_cid = self._form_grasp(_GRASP_OFFSET_Z)
        self.reward_fn._grasp_bonus_paid = True
        self.reward_fn._picked = True

    def _start_from_replay(self):
        """Task 1b/"descend" (cfg.descend_only) replay-seeded curriculum:
        teleport to an EXACT recorded state from a real, proven-successful
        E33 rollout (see scripts/extract_replay_trajectory.py, self._replay).
        Unlike every other curriculum-teleport method in this file, this
        uses `resetJointState` directly from RECORDED joint angles -- no IK
        at all, so zero convergence risk (it's a replay of an already-valid
        real configuration, not a solve for a new target). The hole is also
        repositioned to match the snapshot's own recorded hole_xy (each
        source episode sampled its own hole position; a snapshot is only
        physically valid paired with ITS hole, not whatever this episode's
        sampler drew) -- overrides self._cur.hole_xy and re-calls
        _place_hole before restoring joints/peg.
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
        """Hand-off between Task 1 ("align", zero-offset grasp) and Task 2
        ("insert", _GRASP_OFFSET_Z top-offset grasp): swap the grasp
        constraint from zero-offset to the depth-enabling top-offset.
        Simulated equivalent of a real robot's brief re-grasp maneuver at a
        stage hand-off. Returns False (no-op) if nothing is currently
        grasped -- nothing to regrasp. Composed two-stage eval (Task 1 ->
        this -> Task 2, scripts/evaluate_composed.py) is the only caller;
        not used by any single-task training config.

        UPDATED for the preserve-transform `_form_grasp` (see its
        docstring): the old version called `_form_grasp(_GRASP_OFFSET_Z)`
        "at the peg's current pose, no teleport" and relied on
        `_form_grasp`'s then-hardcoded childFramePosition to physically pull
        the peg into the new offset via the constraint solver over
        subsequent steps -- a real, if gradual, teleport its own docstring
        undersold. Under the preserve-transform weld that pull no longer
        happens (Task 1's zero-offset grasp already has peg_pos==flange_pos
        and peg_orn==flange_orn, so the measured relative transform would
        just be ~identity again, silently dropping the offset). This now
        explicitly slides the peg along ITS OWN current axis -- not world-z,
        not the flange's -- so its top meets the flange, preserving
        whatever alignment Task 1 already achieved (align_only's success IS
        xy/tilt alignment, so by hand-off time the peg should already be
        near-vertical) instead of force-snapping to either frame's
        orientation.
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
            # Cheap: two extra reads, only when this Stage-1 fix is enabled.
            # See PhaseConfig.fine_control_frac's docstring for the "why" --
            # this narrows the ACTUATOR's own per-step motion near the hole,
            # so it applies uniformly to whatever produced `action` (RL
            # policy or scripts/scripted_expert.py's IK controller alike).
            peg_xy = np.array(
                p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.client)[0][:2])
            if float(np.linalg.norm(peg_xy - self._hole_xy)) < _FINE_CONTROL_RADIUS:
                scale = DELTA_Q_SCALE * self.cfg.fine_control_frac
        q_target = np.clip(q + arm_a * scale, self._q_lower, self._q_upper)
        if self.cfg.joint_max_velocity is not None:
            # setJointMotorControlArray doesn't expose maxVelocity (checked
            # directly -- rejected as an invalid kwarg); setJointMotorControl2
            # (per-joint) does. See PhaseConfig.joint_max_velocity's docstring
            # for why this cap exists at all.
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
        self._bore_inradius = bore_inradius   # keypoint reward geometry (rewards.py's
                                              # keypoint_w) reads this via _state_dict()
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
        """Stage-1 fix (cfg.hole_chamfer): 8 boxes forming a tilted funnel
        directly above the straight bore ring, widening the effective capture
        opening (see _CHAMFER_HEIGHT/_CHAMFER_TILT_DEG's docstring for why).

        Each box is pitched outward about its own TANGENTIAL axis (so it
        leans away from vertical without twisting -- the funnel wall, not a
        rotated post) via quaternion composition: first tilt in the box's own
        local frame (rotation about local Y), then yaw that whole tilted box
        to its position around the ring (rotation about world Z). This is
        q_total = q_yaw * q_pitch (p.multiplyTransforms composes exactly this
        way -- q_pitch is applied to local coordinates first, then q_yaw
        rotates the result), so the pitch axis becomes tangential once yawed,
        matching how the straight ring's own local +x becomes radial.

        Position: solved so the box's own BOTTOM-INNER edge (the corner
        closest to the bore axis, at the box's bottom face) lands exactly at
        (bore_inradius, mouth_z) -- continuous with the straight ring's own
        top edge, no step for the peg to catch on. The TOP-INNER edge then
        sits `chamfer_h * tan(tilt)` further out, forming the funnel mouth.
        """
        tilt = np.radians(_CHAMFER_TILT_DEG)
        half_z = _CHAMFER_HEIGHT / 2.0
        q_pitch = p.getQuaternionFromEuler([0.0, tilt, 0.0])
        for k, bid in enumerate(self.hole_chamfer_ids):
            ang = 2 * np.pi * k / self._hole_segments
            q_yaw = p.getQuaternionFromEuler([0.0, 0.0, ang])
            _, q_total = p.multiplyTransforms([0, 0, 0], q_yaw, [0, 0, 0], q_pitch)
            R = np.array(p.getMatrixFromQuaternion(q_total)).reshape(3, 3)
            # local frame: +x radial-outward, +y tangential, +z up. Bottom-
            # inner corner (mid-edge, y=0) before rotation:
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
        # Threshold at -0.2, not 0.5: an untrained policy outputs ~0 for the
        # gripper, and with exploration noise that must actually trigger a
        # grasp for the behaviour to be discovered. Release still needs a
        # clearly negative signal.
        want = grip_a > _GRASP_THRESH
        ee = np.array(p.getLinkState(self.robot_id, _EE_LINK, physicsClientId=self.client)[0])
        peg = np.array(p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.client)[0])
        dist = float(np.linalg.norm(ee - peg))
        attempt = False
        if want and self._grasp_cid is None and dist < self._grasp_trigger:
            attempt = True
            # peg_in_hole: grasp near the peg's TOP (_GRASP_OFFSET_Z), not
            # its center (offset 0) -- see _GRASP_OFFSET_Z's docstring: a
            # zero-offset grasp makes genuine insert depth physically
            # impossible (the flange would have to sit inside the solid
            # hole ring). Phase 1/2/3 (pick_place/reach) are unaffected --
            # unchanged zero-offset grasp, exactly as validated throughout.
            # align_only (Task 1) is ALSO exempted -- its success is pure
            # xy/tilt alignment, never depth, so it never needed the offset;
            # giving it anyway regressed Phase 3's proven grasp-hold skill
            # into a GRASP<->DROP flicker (E31/E32/E34, 0/200 success each).
            rigid = self.cfg.task == "peg_in_hole" and not self.cfg.align_only
            child_off = _GRASP_OFFSET_Z if rigid else 0.0
            # Grasp-admissibility gate (rigid only, see _form_grasp's
            # docstring): the preserve-transform weld no longer force-snaps
            # the flange to the peg's top, so an ungoverned grasp from the
            # peg's side/underside would now weld it there permanently. Only
            # admit a real grasp whose flange is already within
            # [cfg.grasp_admit_axial_lo, cfg.grasp_admit_axial_hi] of the
            # peg's own top (along the peg's body +z) and within
            # cfg.grasp_admit_lateral of its axis (PhaseConfig fields,
            # configs.py -- default band matches what E79 shipped with; an
            # opt-in config can widen it, e.g. to test whether the band is
            # the binding constraint on cold-start grasp acquisition). A
            # rejected attempt still counts toward grasp_attempts/
            # grasp_failures like the old "too far" case -- this only
            # tightens WHICH close-enough attempts succeed.
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
        right now. Queried and returned via env.step()'s info dict (not
        after the fact) because collision-terminal episodes get auto-reset
        by the VecEnv wrapper the instant they end -- an external
        getContactPoints() query after venv.step() returns would see the
        NEXT episode's already-reset world, not the collision. Only used by
        diagnostics; cheap since collision is a rare event."""
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
        # wrist force/torque from the last arm joint's reaction
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
        # reward-shaping-only variant with a wider xy gate (success/diagnostics
        # keep using the tight `depth` above, unaffected -- this just lets the
        # potential's "+depth" term start contributing gradient before the peg
        # is perfectly centred, instead of a hard cliff at ~3.5cm xy error that
        # made near-the-mouth descent a chicken-and-egg problem for the policy).
        #
        # H2 fix (plan: parsed-plotting-allen.md): `depth`/`peg_depth` above
        # measures the peg's CENTRE against the mouth plane -- with
        # _PEG_HEIGHT=8cm, the centre only crosses the mouth after the TIP
        # has already penetrated 4cm, so `_potential()`'s "+2.0*depth" term
        # (and the discrete _DEPTH_PROGRESS_BONUS staircase, rewards.py,
        # which reads this same field) paid exactly nothing for the entire
        # rim-crossing and first 4cm of real insertion -- the only active
        # gradient there was `-0.8*z_gap` fighting `-1.5*xy_err`/`-1.2*tilt`
        # on any contact-induced disturbance, a ridge computed at ~57:1
        # against attempting descent (see the plan's worked expected-value
        # calculation). `peg_depth`/success stays centre-based, unchanged,
        # for comparability with every prior experiment -- only the SHAPING
        # signal is redefined here, to track the TIP (the part that actually
        # has to cross the rim first).
        depth_shaped = 0.0
        if self.cfg.use_hole and xy_err < (_PEG_RADIUS + 0.05):
            tip_z = float(peg_pos[2]) - _PEG_HEIGHT / 2.0
            depth_shaped = max(0.0, self._hole_mouth_z - tip_z)

        # Keypoint "docking" reward geometry (rewards.py's keypoint_w,
        # default 0.0 = off, a no-op elsewhere): K points on the peg's
        # bottom rim (the leading edge that actually clips the bore rim
        # under tilt -- E56's diagnosis) vs K points on the bore's inner
        # wall at the mouth. Reuses R (peg body->world rotation) already
        # computed above. None for non-peg_in_hole tasks so rewards.py's
        # s.get("peg_keypoints") safely no-ops there.
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

        # all-link, not EE-only: collision isn't limited to the flange -- the
        # forearm/elbow can clip an obstacle while the EE itself is clear.
        # getLinkStates batches all 7 arm links (0..ARM_DOF-1, includes
        # _EE_LINK) in one call.
        link_states = p.getLinkStates(self.robot_id, list(range(ARM_DOF)), physicsClientId=c)
        link_positions = [np.array(ls[0]) for ls in link_states]
        obstacle_check_points = link_positions + ([peg_pos] if self.cfg.use_peg else [])
        # _EE_LINK (the wrist flange, "link7" in scripts/diagnose_phase3.py's
        # 1-indexed naming) is the dominant collision contributor by a wide
        # margin (63% of collision-steps in E24's diagnostic, more than
        # every other link combined) -- give it extra VIRTUAL clearance in
        # the shaping-only distance signal (not the physical collision
        # geometry, which stays untouched) so the dense repulsion penalty
        # warns earlier/harder specifically for this link.
        extra_radii = [0.0] * len(link_positions) + ([0.0] if self.cfg.use_peg else [])
        extra_radii[_EE_LINK] = _EE_VIRTUAL_INFLATE
        min_obstacle_dist = self._min_obstacle_dist(obstacle_check_points, extra_radii)
        # Task 2's diagnostic (E33) found 48.4%+29.0% of collision-steps are
        # the FOREARM (links 4/5, 0-indexed 3/4), not the flange -- a
        # kinematic difficulty reaching low/close to the hole while holding
        # near-vertical orientation. Dense shaping-only proximity signal for
        # just those two links against the hole ring, mirroring
        # _min_obstacle_dist's pattern (see _min_ring_dist).
        min_ring_dist = self._min_ring_dist([link_positions[3], link_positions[4]])

        # H3 (plan: parsed-plotting-allen.md): peg-vs-ring CONTACT -- the
        # signal that would let the policy tell "resting on the rim" apart
        # from "entering the bore" -- was already computed for offline
        # diagnosis (scripts/oracle_ceiling.py's ring-contact tracer, E55)
        # but never exposed to the policy itself. First hit wins (mirrors
        # oracle_ceiling.py's own early-break pattern). Force clipped to a
        # sane physical scale (50N) before this reaches the obs assembly --
        # E56 measured a stiff-contact numerical-explosion artifact peaking
        # 300N+ on an UNGRASPED peg driven by a rigid, uncapped motor; that
        # is a real but rare physics glitch this clip keeps from dominating
        # this one dimension's running-normalization variance, not a signal
        # to hide (ring_contact itself still fires at 1.0 unclipped).
        ring_contact = 0.0
        ring_contact_force = 0.0
        if self.cfg.task == "peg_in_hole":
            for bid in self.hole_box_ids + self.hole_chamfer_ids:
                pts = p.getContactPoints(bodyA=self.peg_id, bodyB=bid, physicsClientId=c)
                if pts:
                    ring_contact = 1.0
                    ring_contact_force = float(max(pt[9] for pt in pts))
                    break

        # peg-TIP (not centre) to hole-mouth vector -- the quantity that's
        # actually geometrically relevant during insertion (xy alignment +
        # depth in one signal, at full precision, unlike `goal_pos -
        # peg_pos` above which uses the centre and is dominated by the
        # peg's own half-height once anywhere near the mouth).
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
        currently-active obstacle. `points` is every arm link position (+ the
        peg, if held) -- collision isn't limited to the end-effector, the
        forearm/elbow can clip an obstacle while the EE itself is clear.
        `extra_radii` (optional, same length as `points`) adds a per-point
        VIRTUAL bounding-radius inflation -- shaping-only, does not touch the
        physical collision geometry -- e.g. to give a particular link (see
        _EE_VIRTUAL_INFLATE) an earlier/stronger dense repulsion warning.
        Used only for reward-side proximity shaping (Phase 3+); `inf` when
        there are no active obstacles."""
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
        targeted forearm-repulsion shaping term (see rewards.py) -- Task
        2's diagnostic (E33) found the forearm (links 4/5), not the flange,
        was 48.4%+29.0% of collision-steps."""
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
