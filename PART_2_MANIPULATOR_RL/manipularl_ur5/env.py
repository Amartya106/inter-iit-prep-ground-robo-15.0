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

  * REAL HOLE GEOMETRY. an 8-box octagonal socket on the table -- the
    scaffold's Stage 4 had no hole at all (target was a point at z=0 and
    success needed the peg 3 cm below it: impossible).

  * OBSTACLES IN THE OBSERVATION. the K nearest obstacles are encoded, so
    Phase 3 avoidance and "generalize to unseen layouts" are learnable. The
    observation is fixed once here and only ever zero-filled per phase, never
    redefined.

  * eval-time noise / perturbation live in wrappers.py, never here.

Robot: UR5 (6-DOF) + Robotiq 2F-85 gripper, `assets/ur5_robotiq/urdf/ur5_robotiq_85.urdf`
(MIT-licensed, the PS's own alternative robot config -- see NOTICE.md alongside the asset).
UR5 PORT of manipularl/env.py, built to test whether the descent-phase xy-drift diagnosed in
E68 (EXPERIMENTS.md) is a symptom of KUKA iiwa's redundant 7th DOF -- a 6-DOF arm commanding a
6-DOF pose has no null space to wander through between successive IK re-solves. Everything
below is an unmodified copy of manipularl/env.py except the robot-specific pieces (URDF load,
_EE_LINK, _REST_POSE, and the ARM_DOF/ACTION_DIM this package's own configs.py supplies) --
"keep the methods for solving the problems the same, just change accordingly for the one less
DOF." Grasp is still the scaffold's constraint-based "snap" (kept intentionally, matching KUKA's
own approach exactly -- a stable grasp model for early RL, not finger-force simulation; the
Robotiq gripper's own finger joints are present in the URDF but never actuated).
"""

import dataclasses
import os
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
from manipularl.randomization import EpisodeSampler   # reused unmodified, see configs.py
from manipularl.rewards import RewardComputer          # reused unmodified, see configs.py

# UR5 6-DOF neutral pose (shoulder_pan, shoulder_lift, elbow, wrist_1/2/3). The plan's own
# initial guess (elbow-up, hand-picked) left the EE 28-42cm from the workspace center on a
# smoke test -- grasp_ok collapsed to 0.10 (vs KUKA's usual ~0.66-0.71). SOLVED instead via
# direct IK (200-iteration converge, matching env._ik_limited's own precedent) targeting the
# workspace center [0.45, 0, 0.25] (randomization.py's own, reused-unmodified sampling
# center) with the gripper pointing straight down -- converged to 2.6e-11m residual.
_REST_POSE = np.array([-0.245, -1.383, 1.809, -0.426, -3.387, 0.0])
_EE_LINK = 5   # wrist_3_link, the last of the 6 arm joints' own link -- mirrors KUKA's
              # exact convention (_EE_LINK = ARM_DOF - 1): IK/grasp use the arm's own
              # terminal link directly, not a separate fixed ee_link/gripper-base link.
_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_UR5_URDF = os.path.join(_ASSET_DIR, "ur5_robotiq", "urdf", "ur5_robotiq_85.urdf")
_PEG_RADIUS = 0.015
_PEG_HEIGHT = 0.08
_TABLE_TOP_Z = 0.0
_HOLE_BOXES = 8
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
                         # hole ring (manipularl/env.py's octagonal box ring)
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
            _UR5_URDF, basePosition=[0, 0, 0], useFixedBase=True,
            physicsClientId=c,
        )
        # UR5 PORT FIX (Phase 3 diagnostic, see EXPERIMENTS.md): the
        # obstacle-repulsion SHAPING signal (_state_dict()'s min_obstacle_dist,
        # below) used to only check link_states over range(ARM_DOF) -- correct
        # for KUKA, where _EE_LINK is genuinely the chain's last link, but
        # wrong here: UR5's real gripper (ee_link, robotiq_arg2f_base_link,
        # fingers -- links 6+) sits offset SIDEWAYS from wrist_3_link by
        # 8-17cm and was entirely invisible to that check. Traced directly:
        # 21/21 collisions in a 40-episode Phase-3-checkpoint rollout involved
        # a gripper link, and one was caught with the shaping signal reporting
        # "10cm clear" the instant a finger was already touching the
        # obstacle -- the dense avoidance reward had a total blind spot
        # exactly where the real geometry was. Cache the TRUE link count here
        # (not ARM_DOF) so the shaping check below covers the whole robot,
        # not just the actuated arm joints. The actual collision/termination
        # detection (_obstacle_collision_links(), via getContactPoints on the
        # whole body) was never affected by this -- only the dense,
        # BEFORE-contact shaping signal was blind.
        self._n_robot_links = p.getNumJoints(self.robot_id, physicsClientId=c)
        for j in range(ARM_DOF):
            p.enableJointForceTorqueSensor(self.robot_id, j, True, physicsClientId=c)

        # UR5 PORT FIX (Phase 3, checkpoint-sweep diagnostic): the Robotiq
        # gripper's 12 links (ee_link, robotiq_arg2f_base_link, and the
        # finger/knuckle/pad links, indices ARM_DOF..self._n_robot_links-1)
        # carry full-fidelity MESH collision geometry (~17cm of structure,
        # offset 8.2cm sideways from wrist_3_link via ee_fixed_joint) but are
        # NEVER actuated -- only range(ARM_DOF) ever receives a motor
        # command (see step()), and grasping is a snap p.createConstraint at
        # _EE_LINK, not a physical grip. KUKA's kuka_iiwa/model.urdf has no
        # gripper at all, so this geometry is a pure confound versus KUKA,
        # not a genuine part of the robot being compared. A 100-episode
        # trace on the standing-best checkpoint found 35% of collisions were
        # gripper-link-only and the obstacle curriculum's easy stage (0-1
        # obstacles) already saw 55-60% collision, vs KUKA's 17.5-20% at the
        # full 2-5. Disabling collisions on these links only (group=mask=0)
        # leaves the constraint-based grasp, the arm's own collision
        # detection, and the visual mesh all unaffected.
        for _link in range(ARM_DOF, self._n_robot_links):
            p.setCollisionFilterGroupMask(self.robot_id, _link, 0, 0, physicsClientId=c)

        # peg (always present in the scene; parked when a phase doesn't use it)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=_PEG_RADIUS, height=_PEG_HEIGHT,
                                     physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=_PEG_RADIUS, length=_PEG_HEIGHT,
                                  rgbaColor=[0.4, 0.5, 0.95, 1], physicsClientId=c)
        self.peg_id = p.createMultiBody(0.05, col, vis, [0.45, 0.0, _PEG_HEIGHT / 2],
                                        physicsClientId=c)

        # hole socket: a ring of thin static boxes forming an octagonal bore.
        self.hole_box_ids = []
        for _ in range(_HOLE_BOXES):
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
        for _ in range(_HOLE_BOXES):
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
        # ec.init_joint_offset is sized 7 (manipularl/randomization.py:34,94,
        # hardcoded, reused unmodified -- see configs.py's docstring); slice
        # to this arm's own ARM_DOF=6 -- a per-episode random joint-space
        # perturbation with no meaning tied to a specific joint identity, so
        # dropping the 7th value is harmless.
        q0 = np.clip(_REST_POSE + ec.init_joint_offset[:ARM_DOF],
                     self._q_lower + 0.05, self._q_upper - 0.05)
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
        """
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
        from manipularl.rewards import _DEPTH_PROGRESS_CAP_CM, _DEPTH_PROGRESS_STEP_CM
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
        constraint from zero-offset to the depth-enabling top-offset AT THE
        PEG'S CURRENT POSE -- no teleport, the peg doesn't move. Mirrors
        exactly the constraint _start_pre_inserted() already forms, just
        without also repositioning anything. Simulated equivalent of a real
        robot's brief re-grasp maneuver at a stage hand-off. Returns False
        (no-op) if nothing is currently grasped -- nothing to regrasp.
        Composed two-stage eval (Task 1 -> this -> Task 2) is the intended
        caller; not used by any single-task training config.
        """
        if self._grasp_cid is None:
            return False
        p.removeConstraint(self._grasp_cid, physicsClientId=self.client)
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
        # manipularl.rewards.RewardComputer (reused unmodified, see configs.py's
        # docstring) hardcodes _WRIST_VEL_JOINTS=(4,5,6) -- KUKA's 7-DOF chain --
        # to index s["qd"] for one minor shaping term (wrist_vel_pen); index 6
        # is out of range for this arm's real 6-length qd. Pad a SEPARATE copy
        # of `s` (not the one returned via _assemble_obs below, which correctly
        # wants the real length-6 arrays) with a trailing zero -- "the
        # nonexistent 7th joint has zero velocity" is a safe, semantically
        # reasonable adaptation for a shaping-only term; harmless for the OTHER
        # qd-reading term (arm_jitter_pen) too, since it sums max(0, qd-thresh)
        # over all joints and a zero-velocity extra contributes exactly 0.
        s_for_reward = dict(s)
        if len(s["qd"]) < 7:
            s_for_reward["qd"] = np.concatenate([np.asarray(s["qd"], dtype=float),
                                                 np.zeros(7 - len(s["qd"]))])
        # Same coupling, `action` side: RewardComputer.reset() hardcodes
        # `self._prev_action = np.zeros(8)` (KUKA's ACTION_DIM) and compute()
        # reads `action[:7]`/`self._prev_action[:7]` for two action-magnitude/
        # jerk shaping terms (rewards.py:585-586) -- both ARM-only (excludes
        # the gripper slot deliberately). Build an 8-length view with this
        # arm's real ARM_DOF values, a fake-zero pad for the missing slot(s)
        # BEFORE the gripper (preserving the "arm..., then gripper" layout
        # those two terms and _prev_action's own internal tracking rely on),
        # then the real gripper value last -- a constant zero pad contributes
        # nothing to either term, every step, so this is exactly as harmless
        # as the qd padding above.
        action_for_reward = action
        if len(action) < 8:
            action_arr = np.asarray(action, dtype=np.float32)
            action_for_reward = np.concatenate([
                action_arr[:ARM_DOF], np.zeros(7 - ARM_DOF, dtype=np.float32),
                action_arr[ARM_DOF:],
            ])
        reward, terminated, success, rinfo = self.reward_fn.compute(
            s_for_reward, action_for_reward, collision)
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
            ang = 2 * np.pi * k / _HOLE_BOXES
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
            ang = 2 * np.pi * k / _HOLE_BOXES
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
                self._clear_obstacle_from_rest_pose(oid)
            else:
                p.resetBasePositionAndOrientation(oid, [10 + i, 10, -10], [0, 0, 0, 1],
                                                  physicsClientId=c)
        self._obstacle_active = used

    def _clear_obstacle_from_rest_pose(self, oid, min_clear: float = 0.05, max_tries: int = 8):
        """UR5 PORT FIX (Phase 3, checkpoint-sweep diagnostic): manipularl/randomization.py's
        obstacle keepout (shared with KUKA, not edited here) only excludes a 10cm disc
        around the robot BASE's xy origin -- it has no model of the arm's own swept
        geometry, so an obstacle can spawn already overlapping the rest pose. A
        100-episode trace on the standing-best checkpoint found 9/100 episodes collided
        within the first 2 steps (before any real approach was possible). Since
        collision is instantly terminal for this phase (cfg.collision_is_failure), those
        episodes contribute nothing but noise. Push any obstacle that spawns within
        min_clear of the arm (links 0..ARM_DOF-1, matching what can actually collide
        now that the gripper's own collision geometry is disabled) radially outward
        from its nearest contact point, re-checking until clear or max_tries exhausted."""
        c = self.client
        for _ in range(max_tries):
            closest = p.getClosestPoints(bodyA=self.robot_id, bodyB=oid, distance=min_clear,
                                         physicsClientId=c)
            arm_hits = [pt for pt in closest if pt[3] < ARM_DOF]
            if not arm_hits:
                return
            nearest = min(arm_hits, key=lambda pt: pt[8])
            robot_pt = np.array(nearest[5])
            obs_pos, obs_orn = p.getBasePositionAndOrientation(oid, physicsClientId=c)
            obs_pos = np.array(obs_pos)
            away = obs_pos[:2] - robot_pt[:2]
            norm = np.linalg.norm(away)
            away = (away / norm) if norm > 1e-6 else np.array([1.0, 0.0])
            new_xy = obs_pos[:2] + away * (min_clear + 0.03)
            new_xy = np.clip(new_xy, [0.20, -0.35], [0.65, 0.35])
            p.resetBasePositionAndOrientation(oid, [new_xy[0], new_xy[1], obs_pos[2]], obs_orn,
                                              physicsClientId=c)

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
            child_off = (_GRASP_OFFSET_Z if self.cfg.task == "peg_in_hole"
                        and not self.cfg.align_only else 0.0)
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
        # H2 fix, ported from the KUKA side (plan: parsed-plotting-allen.md):
        # same bug, same file layout (this env.py started as a copy of
        # manipularl/env.py) -- `depth`/`peg_depth` measures the peg's
        # CENTRE, so with _PEG_HEIGHT=8cm the centre only crosses the mouth
        # after the TIP has already penetrated 4cm, leaving the shaping
        # potential's "+2.0*depth" term (and rewards.py's
        # _DEPTH_PROGRESS_BONUS staircase, which now reads this same field)
        # dead for the entire rim-crossing. `peg_depth`/success stays
        # centre-based, unchanged, for comparability with every prior UR5
        # run -- only the SHAPING signal is redefined here, to track the tip.
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
        # UR5 PORT: reverted back to range(ARM_DOF) (links 0..5, the arm
        # joints only), matching KUKA -- a prior fix widened this to
        # range(self._n_robot_links) (all 18 links incl. the Robotiq
        # gripper) to cover the gripper's own collision geometry, but that
        # geometry is now DISABLED entirely (see _build_scene's
        # setCollisionFilterGroupMask comment: the gripper is never actuated
        # and can no longer physically collide), so including those links
        # here only inflates min_obstacle_dist's "in_danger" zone for
        # geometry that can't trigger a real collision. A checkpoint-sweep
        # diagnostic on the widened version found the dense goal-approach
        # shaping (rewards.py's `in_danger` gate) was silenced on 39.5% of
        # all steps -- this reverts that regression.
        link_states = p.getLinkStates(self.robot_id, list(range(ARM_DOF)),
                                      physicsClientId=c)
        link_positions = [np.array(ls[0]) for ls in link_states]
        obstacle_check_points = link_positions + ([peg_pos] if self.cfg.use_peg else [])
        # _EE_LINK (the wrist flange, "link7" in scripts/diagnose_phase3.py's
        # 1-indexed naming) is the dominant collision contributor on KUKA by a
        # wide margin (63% of collision-steps in E24's diagnostic) -- give it
        # extra VIRTUAL clearance in the shaping-only distance signal (not the
        # physical collision geometry, which stays untouched) so the dense
        # repulsion penalty warns earlier/harder specifically for this link.
        # UR5 PORT: mirrors KUKA exactly now that the gripper's own links are
        # out of both the physical-collision and shaping-distance pictures.
        extra_radii = [0.0] * len(link_positions) + ([0.0] if self.cfg.use_peg else [])
        extra_radii[_EE_LINK] = _EE_VIRTUAL_INFLATE
        min_obstacle_dist = self._min_obstacle_dist(obstacle_check_points, extra_radii)
        # Task 2's diagnostic (E33) found 48.4%+29.0% of collision-steps are
        # the FOREARM (links 4/5, 0-indexed 3/4 on KUKA's 7-DOF chain), not
        # the flange -- a kinematic difficulty reaching low/close to the hole
        # while holding near-vertical orientation. UR5 PORT NOTE: indices 3/4
        # here are wrist_1/wrist_2, not a verified forearm-equivalent for
        # this arm's own collision pattern -- carried over unverified since
        # this is a shaping-only signal, not exercised by the oracle-only
        # port this file was built for; re-diagnose (E33-style) before
        # trusting it in any UR5 RL training. Dense shaping-only proximity
        # signal for just these two links against the hole ring, mirroring
        # _min_obstacle_dist's pattern (see _min_ring_dist).
        min_ring_dist = self._min_ring_dist([link_positions[3], link_positions[4]])

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
        ring's octagonal box array (self.hole_box_ids) -- same crude
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
