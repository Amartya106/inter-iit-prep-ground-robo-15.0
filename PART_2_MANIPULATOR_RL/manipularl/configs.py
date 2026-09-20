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

    # Both default to current (frozen) behaviour; every existing config/run
    # is unaffected. See EXPERIMENTS.md's Stage-1 entries for measurements.
    hole_chamfer: bool = False   # angled lead-in ring above the bore mouth
                                 # (env._place_hole), widens the effective
                                 # capture opening for insertion.
    grasp_max_force: Optional[float] = None   # finite max force on the grasp
                                 # constraint instead of a rigid JOINT_FIXED,
                                 # lets the peg deflect and self-centre.
                                 # None = unchanged rigid grasp.
    hole_segments: int = 8   # number of flat box segments forming the bore
                                 # ring (env._build_scene/_place_hole) -- a
                                 # regular polygon, never a true circle
                                 # (PyBullet has no boolean/CSG cutout). 8 =
                                 # the octagon every existing config/run/
                                 # checkpoint was trained against. A higher
                                 # count approximates a circle more closely
                                 # but is a real physics change, needs its
                                 # own retrain before comparing results.
    # Grasp-admissibility band (peg_in_hole, non-align_only real grasps
    # only, see env._handle_grasp/_form_grasp). Widening axial_hi/lateral
    # only makes more approach poses admissible.
    grasp_admit_axial_lo: float = 0.015
    grasp_admit_axial_hi: float = 0.055
    grasp_admit_lateral: float = 0.035
    joint_max_velocity: Optional[float] = None   # cap each arm joint's
                                 # POSITION_CONTROL maxVelocity (rad/s) --
                                 # without a cap, the rigid motor driving
                                 # into an ungrasped peg can build up enough
                                 # contact force to launch it away. None =
                                 # unchanged (no cap).
    fine_control_frac: Optional[float] = None   # scale DELTA_Q_SCALE by this
                                 # factor whenever grasped and within
                                 # _FINE_CONTROL_RADIUS of hole_xy (env.step),
                                 # narrowing per-step motion near the hole for
                                 # a precision approach. Applies to every
                                 # controller, not just the RL policy. None =
                                 # unchanged, constant DELTA_Q_SCALE always.

    # Keypoint "docking" reward geometry (the weight, keypoint_w, lives on
    # RewardComputer in rewards.py). K evenly-spaced points on the peg's
    # bottom rim are matched, by best circular-fit rotation, against K
    # points on the bore's inner wall at the mouth. Default 4 is a no-op
    # while keypoint_w stays 0.0.
    keypoint_n: int = 4

    # fraction of TRAIN episodes that start with the peg already grasped
    # (grasp-discovery curriculum). Never used for eval.
    grasp_curriculum: float = 0.0

    # Task-decomposition variants (pick / align / insert as separate
    # sub-policies rather than one monolithic policy). Applied via
    # ManipulaRLEnv(cfg_overrides=...), never set directly in PHASES below --
    # phase=4's own PhaseConfig stays the official task.
    align_only: bool = False   # peg_in_hole: succeed once xy/tilt-aligned
                                # above the hole mouth (grasped, held), no
                                # depth requirement -- Task 1/"reach+align".
    insert_only: bool = False  # peg_in_hole: every episode (train AND eval)
                                # starts already grasped + xy/tilt-aligned
                                # directly in the insertion zone (see
                                # env._start_pre_inserted); only the final
                                # push-through is left to learn -- Task 2/
                                # "insert". Train-time start depth is
                                # curriculum-controlled; eval always starts
                                # at the hardest setting (depth=0).
    descend_only: bool = False  # peg_in_hole: Task 1b -- starts already
                                # grasped and xy-aligned at a random height
                                # above the hole (env._start_pre_airspace)
                                # and must descend smoothly while staying
                                # aligned. Reuses the full peg_in_hole
                                # success/potential unchanged.

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
