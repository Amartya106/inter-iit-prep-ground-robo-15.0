"""
configs.py -- phase presets for ManipulaRL.

One environment, one observation/action interface, six phases that *extend*
(never redefine) it:

  Phase 1  Reaching                  reach a random target, no peg
  Phase 2  Pick & Place              reach -> grasp -> carry -> release at target
  Phase 3  Obstacle-Aware Manip.     Phase 2 + randomized static obstacles
  Phase 4  Peg-in-Hole Insertion     reach -> grasp -> transit -> align -> insert
  Phase 5  Standard/Dynamics Gen.    Phase 4 task, domain randomization ON,
                                     evaluated on unseen layouts + dynamics
  Phase 6  Robustness / Stress       Phase 4 task, eval-time observation/action
                                     noise + perturbations (via wrappers)

`task` collapses the six phases onto three scene templates; the remaining
per-phase differences are captured by the flags below so the scene builder
and reward function stay single implementations.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---- global interface constants (fixed across all phases) ----
ARM_DOF = 7                       # kuka_iiwa
ACTION_DIM = ARM_DOF + 1          # 7 delta-joint + 1 gripper
CONTROL_HZ = 20.0
PHYSICS_HZ = 240.0
SUBSTEPS = int(PHYSICS_HZ / CONTROL_HZ)   # 12
MAX_OBSTACLES_IN_OBS = 6         # K nearest obstacles encoded in the observation
DELTA_Q_SCALE = 0.05             # rad per unit action per control step

# train / eval episode-seed partition -- "previously unseen layouts" is
# enforced structurally, not by hope.
TRAIN_SEED_LO, TRAIN_SEED_HI = 0, 1_000_000
EVAL_SEED_LO, EVAL_SEED_HI = 1_000_000, 2_000_000


@dataclass
class PhaseConfig:
    phase: int
    name: str
    task: str                       # "reach" | "pick_place" | "peg_in_hole"
    max_steps: int = 300

    # scene content
    use_peg: bool = True
    use_hole: bool = False
    num_obstacles: int = 0
    obstacle_min: int = 0
    obstacle_max: int = 0

    # success / reward shaping
    require_release: bool = False    # Phase 2/3: peg must be released at target
    reach_success_dist: float = 0.03
    place_success_dist: float = 0.03
    insert_success_depth: float = 0.03
    insert_xy_tol: float = 0.010
    insert_tilt_tol_deg: float = 10.0
    insert_hold_steps: int = 10     # tolerance must hold this many consecutive steps
    collision_is_failure: bool = False   # Phase 3+: a colliding run cannot score full success

    # Stage-1 environment fixes (E55's oracle-ceiling diagnosis: the frozen
    # env's own privileged scripted expert scores 2% on Phase 4, so the RL
    # policies plateauing at 1.5-2% were never the bottleneck -- 90% of the
    # collapse is one step, an aligned peg failing to CROSS the rim into the
    # bore: 26.5% of oracle episodes register peg-vs-ring contact, at a mean
    # xy_err of 4.6cm, i.e. landing on the sharp rim, not the 1cm-tolerance
    # opening). Both default to current (frozen) behaviour -- every existing
    # config/run in this file is bit-for-bit unaffected; a NEW opt-in config
    # sets these explicitly. See EXPERIMENTS.md's Stage-1 entries for the
    # oracle numbers measured with each on.
    hole_chamfer: bool = False   # add an angled lead-in ring above the bore
                                 # mouth (see env._place_hole) -- converts
                                 # "land within the 3mm-clearance opening" into
                                 # "land within the chamfer's wider opening and
                                 # slide in", the standard fix for this exact
                                 # failure mode in real insertion fixtures.
    grasp_max_force: Optional[float] = None   # finite max force on the grasp
                                 # constraint (env._grasp_cid) instead of the
                                 # default infinite/rigid JOINT_FIXED -- lets
                                 # the peg deflect and self-centre against the
                                 # chamfer (the sim analogue of an RCC wrist),
                                 # None = unchanged rigid grasp.
    hole_segments: int = 8   # number of flat box segments forming the bore
                                 # ring (env._build_scene/_place_hole) -- the
                                 # wall is a regular polygon, never a true
                                 # circle (PyBullet has no boolean/CSG cutout),
                                 # so this is how round the opening looks and
                                 # feels. 8 = the octagon every existing
                                 # config/run/checkpoint in this file was
                                 # built and trained against, unaffected by
                                 # this field existing. A higher count (e.g.
                                 # 24-32) approximates a circle much more
                                 # closely -- smaller facets, smaller gaps
                                 # between them -- but is a real physics
                                 # change (different contact geometry at the
                                 # rim), so it needs its own retrain/re-eval
                                 # before any number measured under it is
                                 # comparable to the octagon's. Not yet tried.
    # E79's grasp-admissibility band (peg_in_hole, non-align_only real
    # grasps only -- see env._handle_grasp/_form_grasp). Defaults match the
    # values the fix shipped with; every existing config is bit-for-bit
    # unaffected. Widening axial_hi/lateral only makes MORE approach poses
    # admissible (never re-opens the "flange too low, insert depth
    # unreachable" failure _GRASP_OFFSET_Z exists to prevent, which only
    # axial_lo protects against -- left alone here on purpose).
    grasp_admit_axial_lo: float = 0.015
    grasp_admit_axial_hi: float = 0.055
    grasp_admit_lateral: float = 0.035
    joint_max_velocity: Optional[float] = None   # cap each arm joint's
                                 # POSITION_CONTROL maxVelocity (rad/s) --
                                 # found via direct trace while diagnosing
                                 # E55's stalled-descent episodes: the rigid
                                 # motor (force=200 N·m, no velocity cap
                                 # anywhere in this codebase) driving INTO an
                                 # ungrasped peg builds up 6mm of penetration
                                 # and 300N+ of contact force before the
                                 # solver's correction impulse launches the
                                 # 0.05kg peg away at ~0.8 m/s -- a classic
                                 # stiff-contact numerical explosion, not a
                                 # skill failure. None = unchanged (no cap,
                                 # PyBullet's own ~100rad/s default).
    fine_control_frac: Optional[float] = None   # scale DELTA_Q_SCALE by this
                                 # factor whenever grasped + within
                                 # _FINE_CONTROL_RADIUS of hole_xy (env.step)
                                 # -- the actuator's per-step quantum (5.3cm
                                 # EE motion at DELTA_Q_SCALE=0.05, E55's plan
                                 # notes) is ~5x the 1cm success tolerance;
                                 # this narrows it near the hole ("a robot
                                 # slows down for a precision task") without
                                 # touching normal-phase control resolution.
                                 # Applies to EVERY controller (scripted
                                 # expert included, via the environment's own
                                 # actuator response -- not a change to any
                                 # controller's decision logic). None =
                                 # unchanged, constant DELTA_Q_SCALE always.

    # Keypoint "docking" reward geometry (structural, not a live-tunable
    # weight -- the weight itself, `keypoint_w`, lives on RewardComputer, see
    # rewards.py, so it gets SuccessGatedAnneal curriculum compatibility for
    # free). K evenly-spaced points on the peg's bottom rim are matched
    # (best circular-fit rotation, yaw-invariant -- the peg is a rotationally
    # symmetric cylinder, env.py:381, so a FIXED label match would penalize
    # perfectly valid insertions at the "wrong" yaw) against K points on the
    # bore's inner wall at the mouth, in _state_dict() (env.py). Default 4 =
    # unchanged from every existing config's implicit behavior when
    # keypoint_w stays 0.0 (the term is a no-op either way at w=0, this only
    # controls the geometry IF the term is ever turned on).
    keypoint_n: int = 4

    # training aid: fraction of TRAIN episodes that start with the peg already
    # grasped (grasp-discovery curriculum). Never used for eval.
    grasp_curriculum: float = 0.0

    # Task-decomposition variants (SeqPolicy-style: pick / align / insert as
    # separate sub-policies rather than one monolithic policy -- see
    # https://www.sciopen.com/article/10.26599/AIR.2024.9150043). Both are
    # applied via ManipulaRLEnv(cfg_overrides=...), never set directly in
    # PHASES below -- phase=4's own PhaseConfig stays the real, official task.
    align_only: bool = False   # peg_in_hole: succeed once xy/tilt-aligned
                                # above the hole mouth (grasped, held), no
                                # depth requirement -- Task 1/"reach+align".
    insert_only: bool = False  # peg_in_hole: every episode (train AND eval)
                                # starts already grasped + xy/tilt-aligned
                                # directly in the insertion zone (see
                                # env._start_pre_inserted); only the final
                                # push-through is left to learn -- Task 2/
                                # "insert". Train-time start depth is
                                # curriculum-controlled (see
                                # set_insert_start_depth_frac); eval always
                                # starts at the hardest setting (depth=0,
                                # peg just touching the mouth).
    descend_only: bool = False  # peg_in_hole: I further split Task 1 into
                                # two sub-stages -- "1a" (align_
                                # only, above) reaches ANY height in the
                                # "airspace" column above the hole, xy/tilt-
                                # aligned; "1b"/descend_only (this flag)
                                # starts already grasped + xy-aligned at a
                                # RANDOM height within that airspace (see
                                # env._start_pre_airspace) and must move
                                # smoothly straight down into the hole while
                                # staying aligned -- reuses the full (non-
                                # align_only) peg_in_hole success/potential
                                # unchanged, since success here genuinely IS
                                # depth+xy+tilt, just approached from a
                                # randomized starting height instead of
                                # always right at the mouth (insert_only).

    # randomization
    randomize_layout: bool = True
    randomize_dynamics: bool = False       # Phase 5
    dynamics_ranges: dict = field(default_factory=dict)

    # eval-time robustness (consumed by wrappers, never during training)
    eval_obs_noise: float = 0.0
    eval_act_noise: float = 0.0
    eval_perturb_force: float = 0.0


_DYN_RANGES = dict(
    peg_mass=(0.03, 0.12),
    lateral_friction=(0.4, 1.2),
    joint_damping=(0.5, 3.0),
    restitution=(0.0, 0.2),
    bore_clearance=(0.002, 0.006),
)


PHASES = {
    1: PhaseConfig(
        phase=1, name="reaching", task="reach",
        use_peg=False, use_hole=False, max_steps=200,
    ),
    2: PhaseConfig(
        phase=2, name="pick_place", task="pick_place",
        use_peg=True, use_hole=False, require_release=True, max_steps=300,
    ),
    3: PhaseConfig(
        phase=3, name="obstacle_aware", task="pick_place",
        use_peg=True, use_hole=False, require_release=True,
        num_obstacles=3, obstacle_min=2, obstacle_max=5,
        collision_is_failure=True, max_steps=350,
    ),
    4: PhaseConfig(
        phase=4, name="peg_in_hole", task="peg_in_hole",
        use_peg=True, use_hole=True,
        num_obstacles=3, obstacle_min=2, obstacle_max=5,
        collision_is_failure=True, max_steps=400,
    ),
    5: PhaseConfig(
        phase=5, name="dynamics_generalization", task="peg_in_hole",
        use_peg=True, use_hole=True,
        num_obstacles=3, obstacle_min=2, obstacle_max=6,
        collision_is_failure=True, max_steps=400,
        randomize_dynamics=True, dynamics_ranges=_DYN_RANGES,
    ),
    6: PhaseConfig(
        phase=6, name="robustness_stress", task="peg_in_hole",
        use_peg=True, use_hole=True,
        num_obstacles=3, obstacle_min=2, obstacle_max=6,
        collision_is_failure=True, max_steps=400,
        randomize_dynamics=True, dynamics_ranges=_DYN_RANGES,
        eval_obs_noise=0.01, eval_act_noise=0.01, eval_perturb_force=2.0,
    ),
}


def get_phase(phase: int) -> PhaseConfig:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}, got {phase}")
    return PHASES[phase]
