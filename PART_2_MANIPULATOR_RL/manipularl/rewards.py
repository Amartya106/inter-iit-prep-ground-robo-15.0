"""
rewards.py -- staged, potential-based reward + success logic for all phases.

Design (this is the "core RL-design component" the PS asks to be justified):

* Shaping is potential-based:  F = k * (gamma * phi(s') - phi(s)).
  With any potential phi this leaves the optimal policy unchanged
  (Ng et al. 1999), so the dense signal cannot be farmed for reward the way
  a raw -distance term can -- the agent only profits by actually making
  progress toward the current sub-goal.

* The task is a chain of sub-goals, gated by discrete one-time bonuses:
      reach peg  --(+1 on grasp)-->  carry  --(+1 on align)-->  insert.
  phi switches target as each gate is passed, so the gradient always points
  at the next thing to do.

* Regularizers: small quadratic penalties on action magnitude and on action
  change (jerk), plus a per-step time penalty. These keep motions smooth
  and discourage dithering without dominating task reward.

* Collisions (Phase 3+): a per-step penalty AND an episode-level flag. A run
  that reaches the goal but collided along the way is reported as
  success=False -- "collision-free behaviour is part of the objective, not a
  side constraint" (PS).

* Peg-in-hole success requires depth + xy-tolerance + tilt-tolerance to all
  hold for `insert_hold_steps` consecutive control steps, so a lucky
  single-frame alignment does not terminate the episode as a success.
"""

import numpy as np

from .configs import PhaseConfig

_K_SHAPE = 5.0           # potential-shaping gain
_HOVER_HEIGHT = 0.10     # m, pre-grasp approach target height above the peg
                         # (Phase 3+ only, cfg.num_obstacles>0) -- matches
                         # scripts/scripted_expert.py's own APPROACH target
                         # (peg + [0,0,0.10]), which reaches 55-87% success
                         # specifically because it never sweeps laterally
                         # through clutter near the peg. E24/E25's
                         # diagnostics both show link7/EE collisions
                         # concentrated in the pre-grasp reach -- the
                         # straight-line `-d_ee_peg` potential can route
                         # diagonally through that clutter; this stages the
                         # approach into "align above, then descend" instead.
_HOVER_XY_TOL = 0.03     # m, xy tolerance to call the EE "aligned above the
                         # peg" and switch from the hover target to the peg
                         # itself -- tight enough to force a real vertical
                         # descent, loose enough to be reachable under noise.
_GRASP_BONUS = 12.0      # grasping commits the policy to the harder carry
                         # sub-goal; the bonus has to outweigh that "cost".
                         # 8 -> 12: diagnose_phase2 showed carry works from a
                         # pre-grasped start (~17%), so make the value fn back
                         # the pick-up decision up harder.
_HOLD_BONUS = 0.03       # per-step reward for keeping the peg grasped -- paid
                         # ONLY before reaching the target (see compute()):
                         # after that it becomes a hold-forever trap because
                         # `require_release` means holding never scores.
_HOLD_AT_TARGET_PEN = 0.05   # per-step pressure to release once at the target
_CARRY_CHECKPOINT_BONUS = 2.0  # one-time, first time the held peg gets within
                               # 2x the place tolerance -- shortens the
                               # credit-assignment horizon for the carry leg
                               # (mirrors _ALIGN_BONUS for peg-in-hole)
_ALIGN_BONUS = 3.0
_DEPTH_PROGRESS_BONUS = 1.0   # one-time, per cm of NEW max real insertion depth
                              # reached (peg_in_hole). Diagnostic on a trained
                              # Track-A policy found episodes getting the peg
                              # aligned to within 1mm of the hole XY but
                              # max_depth staying EXACTLY 0 -- the continuous
                              # potential's "+2.0*depth" term is too weak
                              # against the risk of a tilt/xy-disrupting rim
                              # contact (weighted 0.5/1.5) during a real
                              # insertion attempt, so the optimum under pure
                              # shaping is "hover aligned, never push in". This
                              # discrete, monotonic (never un-paid) bonus makes
                              # committing to depth pay off regardless of that
                              # per-step noise -- same pattern as
                              # _CARRY_CHECKPOINT_BONUS for the pick_place leg.
_DEPTH_PROGRESS_STEP_CM = 1
_DEPTH_PROGRESS_CAP_CM = 5    # cap a little past insert_success_depth (~3cm)
_ACT_REG = 0.005
_JERK_REG = 0.002
_TIME_PEN = 0.01         # also cancels the small +ve that discounted shaping
                         #   gives to "hover near the goal without arriving"
# Collision + repulsion are now GRASPED-DEPENDENT: a Phase-3 diagnostic on
# the grasp-miss-penalty policy (E19, 0.71/0.22) found 66% of ALL failures
# collide specifically while CARRYING the peg (only 17% collide before ever
# grasping) -- juggling "reach the goal" + "avoid the obstacle" while also
# holding cargo is the harder, still-unprotected case. Raise both terms, and
# raise them more while grasped, so carrying near an obstacle is punished
# harder than merely transiting near one empty-handed.
_COLLISION_PEN_FREE = 1.2    # not grasped (was flat 1.0)
_COLLISION_PEN_CARRY = 1.75  # grasped
_REPULSE_MARGIN = 0.12   # m -- start pushing back on any arm link (+ the peg,
                         # if held) inside this distance to the nearest
                         # obstacle. The collision penalty above only fires on
                         # contact (sparse, too late); this is a dense "veer
                         # away" gradient from env._min_obstacle_dist (now
                         # all-link, not EE-only -- see env.py).
_REPULSE_W_FREE = 0.5    # penalty at d=0, ramping to 0 at _REPULSE_MARGIN,
                         # not grasped. 0.2(orig)->0.75(E21)->0.5(E22 corrected,
                         # was mistakenly run at 0.1): E21's diagnostic showed
                         # the 0.75/1.5 pair DID cut post-grasp collisions
                         # (66%->53% of failures) via genuinely more carry-time
                         # clearance (17.2cm vs 13.5cm empty-handed), but was
                         # over-tuned -- the extra caution cost carry-
                         # completion time (mean steps 104->120, "grasped
                         # never reached target" failures 29%->35%), netting
                         # out to ~break-even success. E22 (0.1/1.0) landed in
                         # the same band as E21 -- repulsion weight wasn't the
                         # sensitive parameter across that range. 0.5 sits
                         # between the two, still above the pre-split baseline
                         # (0.2) for the empty-handed leg.
_REPULSE_W_CARRY = 1.0   # grasped -- still above the original flat 0.2 (real
                         # carry-time avoidance signal) but below E21's 1.5,
                         # to keep some of the collision-reduction win without
                         # as much of the carry-completion cost.
# Utility-based RL (arXiv:2402.02665) -- opt-in, disclosed reformulations of
# two reward terms as explicit utility functions instead of an implicit
# linear scalarization. Both default OFF (RewardComputer.satisficing_
# proximity / risk_seeking_depth, see __init__); every existing checkpoint's
# eval semantics are bit-for-bit unaffected unless a run explicitly opts in.
_SATISFICING_SUPPRESS_FRAC = 0.5   # satisficing_proximity: the paper's Sec.
                         # 4.4 ("U(r)=1 if r>=threshold else 0") applied to
                         # goal-shaping suppression specifically. Today's
                         # `in_danger` gate (compute(), below) zeroes ALL
                         # goal-approach shaping anywhere inside the FULL
                         # _REPULSE_MARGIN -- a UR5 diagnostic earlier this
                         # session found this silences shaping on 39.5% of
                         # ALL training steps, far more than the region that
                         # is actually dangerous. This fraction narrows the
                         # SUPPRESSION radius to a tighter sub-threshold
                         # (here, half of _REPULSE_MARGIN) while leaving the
                         # repulsion PENALTY itself (which already only
                         # activates inside the full margin, see repulse_pen
                         # below) unchanged -- "safe enough" costs nothing
                         # AND suppresses nothing; only genuine proximity
                         # does either.
_RISK_SEEKING_ENTRY_XY = 0.05   # risk_seeking_depth: the peg_in_hole xy_err
                         # radius (matches env.py's _FINE_CONTROL_RADIUS)
                         # inside which the policy is genuinely attempting
                         # entry, not just approaching -- the same region
                         # the plan's own hover-vs-attempt ridge calculation
                         # (H2 diagnosis) was computed over.
_RISK_SEEKING_PENALTY_CAP = 0.10   # risk_seeking_depth: Sec. 4.2's
                         # risk-aware utility (U=mu-lambda*sigma) generalised
                         # to the risk-SEEKING case (reward upside instead of
                         # penalising variance), translated into a bounded-
                         # loss/unbounded-gain structure since a literal
                         # sigma isn't available inside a per-step reward.
                         # Caps ONLY the (xy_err_weight*xy_err +
                         # tilt_weight*tilt) penalty portion of _potential()
                         # peg_in_hole's return, while inside
                         # _RISK_SEEKING_ENTRY_XY -- the depth *bonus* term
                         # stays uncapped. Verified by recomputing the plan's
                         # own ridge table with this cap: narrows the
                         # measured ~59x hover-over-attempt expected-value
                         # gap to ~17x (does not fully flip the sign at
                         # E56's ~10% entry-success rate, but substantially
                         # reduces the penalty asymmetry driving it --
                         # verified directly against the real reward function).
_RING_REPULSE_MARGIN = 0.06   # m -- dense "veer away" gradient for the
                              # FOREARM (links 4/5) against the hole ring
                              # specifically (env._min_ring_dist). Task 2's
                              # diagnostic (E33) found 48.4%+29.0% of
                              # collision-steps were the forearm, not the
                              # flange -- a kinematic difficulty reaching
                              # low/close to the hole while holding near-
                              # vertical orientation. Tighter margin than
                              # _REPULSE_MARGIN (0.12) since the ring's own
                              # footprint is small (~4cm outer radius) and a
                              # wide margin would suppress the approach
                              # itself, not just the forearm's proximity to it.
_RING_REPULSE_W = 1.5        # peg_in_hole only, applies regardless of
                              # grasped state (forearm proximity to the ring
                              # is a risk either way, unlike obstacle
                              # repulsion which is carry/free split)
_INVALID_PEN = 0.05
_SUCCESS_BONUS = 50.0    # dominates the shaped trajectory reward -> reach the
                         #   threshold, don't camp near it
_GRASP_SHAPE_W = 0.08    # per-step nudge on "gripper closing while near peg".
                         # 0.04 -> 0.08 (Phase 2). Was doubled to 0.16 for
                         # E42/Task-1-v7 (targeting E41's 35%-never-grasped
                         # finding) but REVERTED: E42's diagnostic showed
                         # this made things WORSE (never-grasped 35%->41%,
                         # severe flicker 12%->59%) -- likely because a
                         # bigger reward for "hovering near the peg,
                         # gripper half-closing" made that a MORE
                         # profitable safe-harbor than actually grasping
                         # (which risks _GRASP_DROP_PEN) -- directly
                         # undermining the commitment it was meant to
                         # encourage. Back to the wider `near` kernel's
                         # original weight (cold start the EE was stalling
                         # ~27 cm from the peg, diagnose_phase2).
_IN_RANGE_BONUS = 0.05   # small per-step bonus for having the EE inside the
                         # grasp trigger radius (distinct from grasping) --
                         # rewards nailing the approach, not just closing.
                         # Also reverted from E42's 0.10 -- see
                         # _GRASP_SHAPE_W's docstring, same finding/reasoning.
_GRASP_TRIGGER_M = 0.09  # mirrors env._GRASP_TRIGGER (EE-peg dist that lets a
                         # gripper-close actually form the grasp constraint)
_GRASP_MISS_PEN = 0.02   # small per-step penalty for a FAILED grasp attempt
                         # (gripper closed, but too far for the constraint to
                         # form -- env._handle_grasp's grasp_attempt=True with
                         # grasped still False). Today _IN_RANGE_BONUS/
                         # grasp_shape reward closing NEAR the peg but closing
                         # FAR costs nothing beyond _TIME_PEN -- a one-sided
                         # incentive, not a timing signal. This is the mirror
                         # penalty. Deliberately tiny (~1/2.5x _IN_RANGE_BONUS)
                         # since _GRASP_THRESH is set low on purpose (an
                         # untrained policy's near-zero default output already
                         # reads as "wants to grasp"), so this fires often
                         # early in training by design -- "learn to suppress
                         # gripper-close until near" -- and must not become a
                         # training-destabilizing drag before the policy can
                         # even approach. Diagnosed via Phase-3's E13
                         # grasp_failure_rate=0.72 (EXPERIMENTS.md).

# Isolated wrist angular-velocity penalty, PRE-GRASP only, joints 5/6/7 only
# (0-indexed 4,5,6) -- deliberately separate from _ACT_REG/_JERK_REG (which
# regularize the ACTION vector, i.e. commanded delta-joint targets, not
# measured joint velocity `qd`). Motivated by E24's diagnostic: _EE_LINK
# ("link7") and its two nearest neighbours dominate collisions, and 65% of
# E24's failures now happen pre-grasp (one contact during reach is instantly
# fatal there -- see the collision-terminal change). Threshold set from a
# real rollout of the E24 checkpoint: pre-grasp |qd| for these three joints
# cruises at ~0.38-0.43 rad/s typically (p90 <= 0.40 for all three); 2.0
# rad/s sits well above normal reaching speed and well below the ~100 rad/s
# spikes seen in the same data (almost certainly collision-impact numerical
# artifacts, not real commanded motion) -- so this targets genuinely
# fast/aggressive wrist motion specifically, not ordinary reaching.
_WRIST_VEL_JOINTS = (4, 5, 6)   # 0-indexed = joints 5,6,7
_WRIST_VEL_THRESH = 2.0         # rad/s
_WRIST_VEL_PEN_W = 0.02         # per rad/s of excess, per joint, summed

_GRASP_DROP_PEN_DEFAULT = 2.0    # peg_in_hole only: one-time penalty fired on the
                         # exact grasped True->False transition. Found via
                         # direct tracing (E31/E32) that Task 1 ("align")
                         # rapidly cycles GRASP<->DROP (up to 218 toggles in
                         # a single 400-step episode) instead of holding --
                         # confirmed NOT a physics-stability issue (a
                         # zero-action hold test showed the grasp constraint
                         # itself stays bounded), the POLICY is choosing to
                         # release. The reward function had no direct cost
                         # for dropping a formed grasp: _GRASP_BONUS (12.0)
                         # is one-time-only (never re-paid on a re-grasp)
                         # and _HOLD_BONUS (0.03/step) only stops accruing
                         # on a drop, it doesn't actively penalize one.
                         # peg_in_hole has no require_release mechanic
                         # (unlike pick_place), so releasing before success
                         # is never legitimate -- safe to penalize outright.
                         # 2.0 sits well above the tiny 0.02-0.05 per-step
                         # terms (a single drop should sting) but well below
                         # _GRASP_BONUS (12.0) so one honest grasp is never
                         # net-negative -- the point is that REPEATED
                         # flickering (which can recur dozens of times per
                         # episode) compounds into a severe cost, not that
                         # grasping itself becomes risky. This is the
                         # DEFAULT for RewardComputer.grasp_drop_pen (a
                         # live-settable attribute, mirroring xy_err_weight)
                         # -- E43's GraspDropPenaltyAnneal callback can ramp
                         # it up from a smaller value instead of applying
                         # full strength from step 0, on the hypothesis
                         # that a harsh, ever-present drop cost during early
                         # exploration (when accidental bad grasps are
                         # common) is itself what drives grasp-shyness.

_ARM_JITTER_THRESH = 1.5   # rad/s -- lower than _WRIST_VEL_THRESH (2.0)
                           # since this gates on the POST-GRASP fine-hold
                           # phase specifically (Task 1's v5 diagnostic:
                           # once grasped, best-case xy_err was 0.18cm but
                           # mean was 5.7cm with only 13% under the 1cm
                           # success tolerance -- the arm isn't settling
                           # into a steady hold near the target). Applies
                           # to ALL arm joints (unlike _WRIST_VEL_PEN_W's
                           # joints-5/6/7-only, pre-grasp-only scope) --
                           # this is about steady FINE HOLD near the hole,
                           # not aggressive reaching motion.
_ARM_JITTER_PEN_W = 0.03  # per rad/s of excess, per joint, summed --
                           # slightly above _WRIST_VEL_PEN_W (0.02), same
                           # order of magnitude.

_LINGER_STEPS_CAP = 20     # cap on the linger penalty's growth (see
                           # `linger_pen` in compute()) -- with the
                           # E43-variant's linger_pen_w=0.05, this caps the
                           # per-step penalty at 1.0, comparable in scale to
                           # _GRASP_DROP_PEN_DEFAULT, rather than growing
                           # unboundedly over a 100+-step lingering stretch.

_SETTLE_BONUS_PER_STEP = 0.15  # peg_in_hole only: dense reward scaled by
                           # self._hold (the consecutive-steps-aligned
                           # counter _success() already computes -- see
                           # _success()'s `aligned`/`task1_aligned`
                           # predicates and cfg.insert_hold_steps). Before
                           # this, the ONLY reward tied to the actual
                           # success predicate was the sparse, all-or-
                           # nothing _SUCCESS_BONUS at the full hold --
                           # nothing rewarded PARTIAL progress toward
                           # sustaining it. E40/E41's diagnostics showed the
                           # policy CAN reach the required precision
                           # momentarily (best-case xy_err 0.16-0.18cm,
                           # consistently) but doesn't settle there (mean
                           # ~6cm) -- this gives a felt, compounding
                           # gradient toward sustaining the hold once
                           # reached, distinct from the cost-side
                           # (_GRASP_DROP_PEN, jitter penalty) and
                           # approach-gradient-side (xy_err_weight) levers
                           # already tried. 0.15/step * up to
                           # cfg.insert_hold_steps(10) ~= 1.5, same order of
                           # magnitude as one _ALIGN_BONUS (3.0) by the time
                           # a hold is nearly complete -- a real, felt
                           # incentive, not a token nudge. Gated at the
                           # peg_in_hole task level (not align_only-only)
                           # since Task 2's own success branch uses the
                           # identical self._hold mechanic.


class RewardComputer:
    """Per-episode reward state machine. One instance per env, reset each episode."""

    def __init__(self, cfg: PhaseConfig, gamma: float = 0.99):
        self.cfg = cfg
        self.gamma = gamma
        # Dense potential-shaping's xy_err coefficient (see _potential,
        # peg_in_hole branch) -- a TRAINING-PROGRESS value, not per-episode
        # state, so set here (persists across reset()) not in reset().
        # Default 1.5 matches the value used unconditionally throughout;
        # live-updatable via env.set_xy_err_weight (XyErrWeightAnneal
        # callback) to soften the pull-toward-the-hole gradient as training
        # progresses -- a strong attraction term can itself drive
        # overshoot/oscillation right at the target (classic high-gain-near-
        # setpoint intuition), which is consistent with Task 1 v5's
        # diagnostic showing imprecise, unsettled hovering concentrated near
        # the hole rather than during transit.
        self.xy_err_weight = 1.5
        # Live-settable (mirrors xy_err_weight), default raised 0.5 -> 1.2
        # (E56's tilt-lock diagnosis): tracing "well-aligned but never
        # enters" episodes found tilt, not xy, as the actual entry blocker --
        # a peg held 20-40deg off-axis has a footprint far wider than the
        # bore's 3mm clearance permits, regardless of how good xy is. That
        # diagnosis came from `scripts/scripted_expert.py`'s solve-once-then-
        # freeze IK (fixing IT directly destabilized position tracking, see
        # EXPERIMENTS.md's Stage-1 entry) -- but a REWARD-trained policy can
        # pursue position and orientation continuously rather than as a
        # one-shot hard constraint, so strengthening this term is the right
        # place to act on the finding, not the scripted diagnostic tool.
        self.tilt_weight = 1.2
        # Live-settable, mirrors xy_err_weight -- see _GRASP_DROP_PEN_DEFAULT's
        # docstring. GraspDropPenaltyAnneal can ramp this up from a lower
        # starting value; left at the default (full strength from step 0,
        # E40/E41/E42's behavior) unless a config attaches that callback.
        self.grasp_drop_pen = _GRASP_DROP_PEN_DEFAULT
        # Fixed (non-curriculum) magnitude for the linger penalty -- see the
        # `linger_pen` block in compute() for the full rationale. 0.0 =
        # disabled (default, matches every run before E43's linger-penalty
        # variant); set via ManipulaRLEnv(linger_pen_w=...) for the variant
        # that wants it.
        self.linger_pen_w = 0.0
        # Live-settable magnitude for the settle bonus (see
        # _SETTLE_BONUS_PER_STEP's docstring for the mechanism). Defaults to
        # the module constant so every run before/including the v8 batch is
        # unaffected; set to 0.0 via ManipulaRLEnv(settle_bonus_w=0.0) to
        # reproduce E41's pre-settle-bonus incentive landscape exactly --
        # needed because E43 found the settle bonus ALONE (isolated from
        # v7's other two changes) reproduces most of v7's toggle/collision
        # regression, so a clean "vary only xy_err_weight on top of E41"
        # experiment must turn it off, not just skip re-adding v7's other
        # two changes.
        self.settle_bonus_w = _SETTLE_BONUS_PER_STEP
        # Live-settable collision levers, decoupled from cfg.collision_is_
        # failure (which stays fixed -- collision ALWAYS precludes success,
        # unconditionally, for every phase-3+ config; see _success()). These
        # two instead control the OTHER two things collision does:
        #   collision_pen_mult: multiplies the per-step _COLLISION_PEN_*
        #     penalty (default 1.0 = full strength, matches every run so
        #     far). CollisionCurriculum can ramp this up from a lower
        #     starting value.
        #   collision_terminal: whether a collision ALSO ends the episode
        #     immediately (default True, matches E24's fix -- the behavior
        #     every Task 1/2 run so far has trained under). Mirrors
        #     Phase 3's own real history: E9-E23 had collision preclude
        #     success but NOT end the episode early; E24 later added
        #     immediate termination once the underlying approach/grasp/carry
        #     skill was already decent (0.68-0.73 success). Set False via
        #     CollisionCurriculum to let a still-learning policy keep
        #     experiencing the rest of an episode after a collision, instead
        #     of being cut off before it discovers grasp/align at all.
        self.collision_pen_mult = 1.0
        self.collision_terminal = True
        # Live-settable arm-jitter penalty magnitude (see the `arm_jitter_pen`
        # block in compute() and _ARM_JITTER_THRESH/_ARM_JITTER_PEN_W's
        # docstrings for the mechanism). Defaults to the module constant so
        # every run before this is unaffected; set via
        # ManipulaRLEnv(arm_jitter_pen_w=...) to test a different strength
        # (e.g. the Task-1-"airspace"/"descend" split's 1.5x/2x/2.5x sweep --
        # E41's own introduction of this penalty was the single best change
        # in the whole Task 1 series, so testing whether MORE of it helps
        # the smooth-descent sub-task specifically is a direct, motivated
        # follow-up, not a shot in the dark).
        self.arm_jitter_pen_w = _ARM_JITTER_PEN_W
        # Live-settable keypoint-docking potential coefficient (see
        # _keypoint_dist's docstring and _potential's peg_in_hole branch).
        # 0.0 = off (default) -- every existing run/config/checkpoint is
        # bit-for-bit unaffected until something explicitly opts in via
        # ManipulaRLEnv(keypoint_w=...) or env.set_keypoint_weight.
        self.keypoint_w = 0.0
        # Utility-based RL (arXiv:2402.02665), both opt-in, both False by
        # default -- see _SATISFICING_SUPPRESS_FRAC/_RISK_SEEKING_* above
        # for the full mechanism and rationale of each. False reproduces
        # every existing run's exact reward semantics; set via
        # ManipulaRLEnv(satisficing_proximity=..., risk_seeking_depth=...).
        self.satisficing_proximity = False
        self.risk_seeking_depth = False
        self.reset()

    def reset(self):
        self._phi = None
        self._prev_action = np.zeros(8, dtype=np.float32)
        self._grasp_bonus_paid = False
        self._align_bonus_paid = False
        self._carry_checkpoint_paid = False
        self._depth_checkpoint_cm = 0
        self._was_grasped = False
        self._episode_collided = False
        self._hold = 0
        self._linger_steps = 0
        self._picked = False
        self._reached_target = False
        self._release_window = 0
        self.stats = dict(
            collisions=0, invalid_states=0, grasp_attempts=0, grasp_failures=0,
            max_depth=0.0, min_xy_err=np.inf, min_tilt_deg=np.inf,
            min_keypoint_dist=np.inf,
        )

    # ------------------------------------------------------------------ #
    def _keypoint_dist(self, s: dict) -> float:
        """Best circular-fit mean distance between the K peg-bottom-rim
        keypoints and the K bore-inner-wall keypoints (env._state_dict()'s
        `peg_keypoints`/`hole_keypoints`, None for non-peg_in_hole tasks).

        "Best circular-fit": searches all K rotational alignments (a
        circular SHIFT of the matching, not a full permutation) and takes
        the lowest-distance one, rather than pairing peg keypoint i with
        hole keypoint i by a FIXED absolute-angle label. The peg is a
        rotationally symmetric cylinder (env.py's p.GEOM_CYLINDER) -- any
        relative yaw fits the bore equally well, so a fixed label match
        would penalize perfectly valid insertions at the "wrong" yaw, one
        specific rotation out of K equally-good ones. Circularly searching
        for the best alignment each step makes the term yaw-invariant
        (matches whichever rotation the peg's current yaw already implies)
        while still penalizing exactly what actually blocks entry: a
        tilted peg's footprint spreading away from the bore wall on one
        side. K<=8 here, so the O(K^2) search is negligible per step."""
        pk, hk = s.get("peg_keypoints"), s.get("hole_keypoints")
        if pk is None or hk is None:
            return 0.0
        K = len(pk)
        return min(
            float(np.mean([np.linalg.norm(pk[i] - hk[(i + r) % K]) for i in range(K)]))
            for r in range(K)
        )

    def _potential(self, s: dict) -> float:
        """Negative distance to the *current* sub-goal (task-dependent)."""
        task = self.cfg.task
        ee = s["ee_pos"]

        if task == "reach":
            return -np.linalg.norm(ee - s["goal_pos"])

        peg = s["peg_pos"]
        d_ee_peg = float(np.linalg.norm(ee - peg))
        # Target the peg while it is not held. `_picked` no longer latches
        # permanently: if the policy drops the peg and the hand drifts away,
        # go back to targeting the peg so a drop is recoverable, not a trap.
        if not s["grasped"] and (not self._picked or d_ee_peg > 0.15):
            if self.cfg.num_obstacles > 0:
                # Staged Z-aligned approach (Phase 3+ only -- gated off for
                # Phase 1/2, which have no clutter to avoid and only
                # regression risk on an already-solved result). Align above
                # the peg (xy only, ignoring peg's actual lower z) before
                # rewarding any descent, so there's no gradient toward
                # cutting diagonally down-and-across through obstacles near
                # the peg -- mirrors scripted_expert.py's own
                # APPROACH(10cm above)->DESCEND state machine. Pure function
                # of current state (no sticky flag), so it stays within the
                # existing potential-shaping pattern.
                xy_err = float(np.linalg.norm(ee[:2] - peg[:2]))
                if xy_err > _HOVER_XY_TOL:
                    target = np.array([peg[0], peg[1], peg[2] + _HOVER_HEIGHT])
                else:
                    target = peg
                return -float(np.linalg.norm(ee - target))
            return -d_ee_peg

        if task == "pick_place":
            # Constant offset so the potential does NOT jump sharply negative
            # the instant the peg is grasped (which was training the policy
            # that grasping is bad); + a lift term rewarding getting the peg
            # off the table.
            lift = min(0.15, max(0.0, float(peg[2]) - 0.05))
            return 0.6 - np.linalg.norm(peg - s["goal_pos"]) + 0.5 * lift

        # peg_in_hole, holding the peg
        hole_xy = s["hole_xy"]
        mouth_z = s["hole_mouth_z"]
        xy_err = np.linalg.norm(peg[:2] - hole_xy)
        z_gap = max(0.0, peg[2] - mouth_z)          # still above the plate
        tilt = s["peg_tilt_rad"]
        # shaping-only wider-gated depth (see env._state_dict) -- success
        # still checks the tight `peg_depth` independently in compute() below.
        depth = s.get("peg_depth_shaped", s["peg_depth"])
        # keypoint-docking term (see _keypoint_dist's docstring) -- an
        # additional, opt-in (keypoint_w defaults 0.0) potential term, added
        # alongside xy_err/tilt/z_gap/depth below, NOT replacing them.
        kp_pen = self.keypoint_w * self._keypoint_dist(s)
        # once roughly aligned above the mouth, reward downward progress.
        # xy_err's coefficient is self.xy_err_weight (default 1.5, live-
        # annealed toward 1.0 by XyErrWeightAnneal -- see __init__).
        if self.cfg.align_only:
            # Task 1a/"airspace align" (my 2-subpart split of Task 1): the
            # peg only has to reach the "airspace" column above the hole --
            # ANY height, xy/tilt-aligned -- not one specific height. The
            # z_gap/depth terms below exist to pull the peg DOWN toward the
            # mouth, which is exactly the wrong incentive here (they were
            # silently active for align_only before this change, fighting
            # the "any height is fine" framing this task is meant to have).
            # xy_err + tilt alone, no height/depth pull at all.
            return -(self.xy_err_weight * xy_err) - (self.tilt_weight * tilt) - kp_pen
        penalty = (self.xy_err_weight * xy_err) + (self.tilt_weight * tilt)
        # risk_seeking_depth (arXiv:2402.02665 Sec. 4.2, opt-in): a
        # risk-averse linear utility punishes the VARIANCE of attempting
        # descent -- a bad outcome (rim-clip disturbance: xy_err/tilt both
        # spike) costs far more than a good outcome gains, so hovering wins
        # in expectation even though depth progress is the actual goal (the
        # plan's own ridge calculation: ~59x in favour of hovering at
        # E56's ~10% clean-entry rate). Bounded-loss/unbounded-gain
        # translation of the risk-SEEKING case (reward upside instead of
        # penalising variance) since no literal sigma is available inside a
        # per-step reward: cap the PENALTY portion only, only while
        # genuinely attempting entry (xy_err within _RISK_SEEKING_ENTRY_XY
        # of the bore axis -- the same region the ridge calc was computed
        # over), leaving the depth BONUS term below fully uncapped. Verified
        # (see _RISK_SEEKING_PENALTY_CAP's docstring): narrows the ~59x gap
        # to ~17x at this cap value -- a substantial, disclosed reduction,
        # not asserted to fully flip the sign on its own.
        if self.risk_seeking_depth and xy_err < _RISK_SEEKING_ENTRY_XY:
            penalty = min(penalty, _RISK_SEEKING_PENALTY_CAP)
        return -penalty - (0.8 * z_gap) + (2.0 * depth) - kp_pen

    # ------------------------------------------------------------------ #
    def compute(self, s: dict, action: np.ndarray, collision: bool):
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float32)
        info = {}

        # --- shaping term ---
        phi = self._potential(s)
        # Suppress the dense pull-to-goal while any part of the arm (or the
        # held peg) is inside the obstacle danger radius (Phase 3+): the
        # repulsion penalty below can be outweighed by this shaping's
        # goal-approach reward on some paths, making "cut through the danger
        # zone" net-positive -- exactly the shortcut we don't want. Zeroing
        # it removes any incentive to travel through that region at all
        # (only the repulsion penalty applies there now); `self._phi` still
        # updates every step, so shaping resumes cleanly from wherever the
        # trajectory exits the zone, crediting no progress made while inside
        # it. This trades the strict Ng et al. policy-invariance guarantee
        # (which only covers the pure potential-diff term) for a harder
        # behavioural constraint -- a disclosed choice, same spirit as the
        # other non-potential terms (collision/repulsion/time penalties)
        # already in this reward.
        # satisficing_proximity (arXiv:2402.02665 Sec. 4.4, opt-in): a
        # threshold utility on suppression specifically -- "safe enough"
        # (outside a TIGHTER sub-threshold) suppresses nothing, only
        # genuine proximity does. The repulsion PENALTY below is untouched
        # either way (already only activates inside the full margin).
        danger_radius = (_REPULSE_MARGIN * _SATISFICING_SUPPRESS_FRAC
                         if self.satisficing_proximity else _REPULSE_MARGIN)
        in_danger = cfg.num_obstacles > 0 and s.get("min_obstacle_dist", float("inf")) < danger_radius
        if self._phi is None or in_danger:
            shaped = 0.0
        else:
            shaped = _K_SHAPE * (self.gamma * phi - self._phi)
        self._phi = phi

        # --- dense grasp shaping (before the peg is grasped) ---
        # Grasp is a gated event: gripper signal > 0.5 AND ee within trigger
        # distance, simultaneously, by exploration. PPO's Gaussian noise almost
        # never stumbles onto it. Reward "gripper closing WHILE near the peg"
        # so the gate is reached by gradient, not luck. Zero once grasped, so
        # it does not distort the post-grasp reward. Applies to every peg phase
        # (2-6) -- they all have to approach and pick up the peg; phase 1
        # (reach, no peg) is unaffected.
        grasp_shape = 0.0
        if cfg.use_peg and not s["grasped"] and not self._picked:
            d_ee_peg = float(np.linalg.norm(s["ee_pos"] - s["peg_pos"]))
            # wide kernel: diagnose_phase2 showed the EE stalling ~27 cm out on
            # cold start, so the pull has to reach well beyond the 12 cm the
            # narrow kernel covered.
            near = np.exp(-d_ee_peg / 0.08) if d_ee_peg < 0.35 else 0.0
            # ramps 0 -> 1 as the gripper signal rises past the grasp threshold
            closing = float(np.clip((s.get("gripper_cmd", 0.0) + 0.2) / 1.2, 0.0, 1.0))
            grasp_shape = _GRASP_SHAPE_W * near * closing
            # being *in grasp range* is worth a nudge on its own (nailing the
            # approach), independent of whether the gripper is closing yet
            if d_ee_peg < _GRASP_TRIGGER_M:
                grasp_shape += _IN_RANGE_BONUS
            # failed attempt this step -- ground truth from env._handle_grasp,
            # not a re-derived distance/threshold check, so it can't drift out
            # of sync with env.py's own (possibly curriculum-annealed) trigger
            if s.get("grasp_attempt", False) and not s["grasped"]:
                grasp_shape -= _GRASP_MISS_PEN

        # --- isolated wrist angular-velocity penalty (pre-grasp, joints 5/6/7 only) ---
        # Separate from _ACT_REG/_JERK_REG (action-space regularizers) --
        # this reads actual measured joint velocity `qd`, not the commanded
        # action, and only engages above _WRIST_VEL_THRESH (see constants).
        wrist_vel_pen = 0.0
        if cfg.num_obstacles > 0 and not s["grasped"] and not self._picked:
            qd_wrist = np.abs(np.asarray(s["qd"], dtype=float)[list(_WRIST_VEL_JOINTS)])
            excess = np.maximum(0.0, qd_wrist - _WRIST_VEL_THRESH)
            wrist_vel_pen = -_WRIST_VEL_PEN_W * float(np.sum(excess))

        # --- discrete gates ---
        gate = 0.0
        if s["grasped"] and not self._grasp_bonus_paid:
            gate += _GRASP_BONUS
            self._grasp_bonus_paid = True
            self._picked = True
        if s["grasped"] and not self._reached_target:
            gate += _HOLD_BONUS      # holding pays -- but only until the target
                                     # is reached; after that it is a trap
                                     # (require_release => holding never scores)
        if cfg.task == "pick_place" and s["grasped"] \
                and not self._carry_checkpoint_paid:
            d_pg = float(np.linalg.norm(s["peg_pos"] - s["goal_pos"]))
            if d_pg < 2.0 * cfg.place_success_dist:
                gate += _CARRY_CHECKPOINT_BONUS
                self._carry_checkpoint_paid = True
        if cfg.task == "peg_in_hole" and s["grasped"] and not self._align_bonus_paid:
            xy_err = float(np.linalg.norm(s["peg_pos"][:2] - s["hole_xy"]))
            # 2x insert_xy_tol was tight enough that it may rarely fire cold
            # (the scripted expert itself needed a <=3.5mm entry gate to reach
            # this reliably); loosen to 3x -- this is a one-time SHAPING bonus,
            # not the success predicate, which still requires 1x insert_xy_tol.
            if xy_err < 3.0 * cfg.insert_xy_tol and s["peg_tilt_rad"] < np.radians(cfg.insert_tilt_tol_deg):
                gate += _ALIGN_BONUS
                self._align_bonus_paid = True
        if cfg.task == "peg_in_hole" and s["grasped"]:
            # H2 fix (plan: parsed-plotting-allen.md): this staircase used to
            # read the centre-based `peg_depth`, so like `_potential()`'s
            # continuous term it paid nothing for the first 4cm of real
            # insertion (_PEG_HEIGHT=8cm). Reads the same TIP-based
            # `peg_depth_shaped` `_potential()` now uses (env.py's
            # _state_dict), so the discrete +1/cm checkpoints and the
            # continuous shaping gradient agree on what "progress" means.
            # `.get(..., s["peg_depth"])` falls back for any caller whose
            # state dict predates this field (defensive only -- both env.py
            # implementations in this repo always provide it).
            depth_for_progress = float(s.get("peg_depth_shaped", s["peg_depth"]))
            cm_now = min(_DEPTH_PROGRESS_CAP_CM,
                         int(depth_for_progress * 100.0) // _DEPTH_PROGRESS_STEP_CM)
            if cm_now > self._depth_checkpoint_cm:
                gate += _DEPTH_PROGRESS_BONUS * (cm_now - self._depth_checkpoint_cm)
                self._depth_checkpoint_cm = cm_now

        # --- regularizers ---
        reg = (
            -_ACT_REG * float(np.sum(action[:7] ** 2))
            - _JERK_REG * float(np.sum((action - self._prev_action)[:7] ** 2))
            - _TIME_PEN
        )
        self._prev_action = action

        # --- collision --- (grasped-dependent, see constants above)
        col_pen = 0.0
        if collision:
            self.stats["collisions"] += 1
            self._episode_collided = True
            col_pen = -self.collision_pen_mult * (
                _COLLISION_PEN_CARRY if s["grasped"] else _COLLISION_PEN_FREE)

        # --- obstacle repulsion (dense, Phase 3+, grasped-dependent) ---
        repulse_pen = 0.0
        if cfg.num_obstacles > 0:
            d_obs = s.get("min_obstacle_dist", float("inf"))
            if d_obs < _REPULSE_MARGIN:
                w = _REPULSE_W_CARRY if s["grasped"] else _REPULSE_W_FREE
                repulse_pen = -w * (1.0 - d_obs / _REPULSE_MARGIN) ** 2

        # --- arm-jitter penalty (dense, any task, post-grasp/while carrying) ---
        # All arm joints, not just wrist -- see _ARM_JITTER_THRESH/
        # _ARM_JITTER_PEN_W: originally targeted the imprecise, unsettled
        # fine-hold behavior near the hole diagnosed at Task 1 v5 (mean
        # xy_err 5.7cm despite a 0.18cm best case -- the arm isn't
        # settling), peg_in_hole-only at the time (E41). Generalized to any
        # task while grasped, to try it for a Phase 3 jitter-penalty
        # variant warm-started from E24 -- the same "arm isn't settling
        # into steady, precise motion once holding something" rationale
        # applies equally to carrying a peg toward a place target, not just
        # holding it still above a hole. No config toggle, same as every
        # other permanent reward.py term.
        arm_jitter_pen = 0.0
        if s["grasped"]:
            qd_all = np.abs(np.asarray(s["qd"], dtype=float))
            excess = np.maximum(0.0, qd_all - _ARM_JITTER_THRESH)
            arm_jitter_pen = -self.arm_jitter_pen_w * float(np.sum(excess))

        # --- forearm-vs-hole-ring repulsion (dense, peg_in_hole only) ---
        ring_repulse_pen = 0.0
        if cfg.task == "peg_in_hole":
            d_ring = s.get("min_ring_dist", float("inf"))
            if d_ring < _RING_REPULSE_MARGIN:
                ring_repulse_pen = -_RING_REPULSE_W * (1.0 - d_ring / _RING_REPULSE_MARGIN) ** 2

        # --- grasp-hold stability: penalize DROPPING a formed grasp
        # (peg_in_hole only -- see _GRASP_DROP_PEN_DEFAULT) ---
        grasp_drop_pen = 0.0
        if cfg.task == "peg_in_hole":
            if self._was_grasped and not s["grasped"]:
                grasp_drop_pen = -self.grasp_drop_pen
            self._was_grasped = bool(s["grasped"])

        # --- linger penalty: penalize spending consecutive steps within
        # grasp range WITHOUT grasping (peg_in_hole only, disabled unless
        # self.linger_pen_w > 0 -- see RewardComputer.__init__'s docstring).
        # E42's diagnostic found the reverse of what was wanted from
        # doubling the pre-grasp approach bonuses: hovering near the peg
        # got MORE profitable, not less attractive, so grasp-shyness got
        # WORSE. This targets the same problem from the opposite
        # direction -- a growing COST for inaction instead of a bigger
        # reward for proximity, closing the "hover forever for free
        # reward" loophole directly. Grows linearly with consecutive
        # in-range-not-grasped steps (capped at _LINGER_STEPS_CAP so a very
        # long lingering stretch doesn't blow up to an absurd magnitude --
        # never-grasped episodes averaged 100-150 attempts in E40-E42's
        # diagnostics), resets the moment it grasps (or leaves range) --
        # cheap to escape by just committing.
        linger_pen = 0.0
        if cfg.task == "peg_in_hole" and self.linger_pen_w > 0.0:
            d_ee_peg = float(np.linalg.norm(s["ee_pos"] - s["peg_pos"]))
            in_range_not_grasped = (not s["grasped"]) and d_ee_peg < _GRASP_TRIGGER_M
            self._linger_steps = self._linger_steps + 1 if in_range_not_grasped else 0
            linger_pen = -self.linger_pen_w * min(self._linger_steps, _LINGER_STEPS_CAP)

        # --- invalid arm state ---
        inv_pen = 0.0
        if s.get("at_joint_limit", False):
            self.stats["invalid_states"] += 1
            inv_pen = -_INVALID_PEN

        # --- grasp bookkeeping ---
        if s.get("grasp_attempt", False):
            self.stats["grasp_attempts"] += 1
            if not s["grasped"]:
                self.stats["grasp_failures"] += 1

        # --- release pressure ---
        # Once the held peg has been brought to the target, still holding is a
        # trap (require_release => holding never scores, and the hold bonus is
        # cut above). A small per-step penalty pushes the policy to let go.
        release_pen = 0.0
        if cfg.task == "pick_place" and self._reached_target and s["grasped"]:
            release_pen = -_HOLD_AT_TARGET_PEN

        reward = (shaped + grasp_shape + gate + reg + col_pen + repulse_pen + inv_pen
                  + release_pen + wrist_vel_pen + ring_repulse_pen + grasp_drop_pen
                  + arm_jitter_pen + linger_pen)

        # --- success / termination ---
        success, terminated = self._success(s)
        if success:
            reward += _SUCCESS_BONUS

        # --- dense settling-progress bonus (peg_in_hole only) ---
        # self._hold is updated INSIDE _success() above (the consecutive-
        # steps-aligned counter) -- read it back now that it reflects this
        # step. See _SETTLE_BONUS_PER_STEP's docstring.
        if cfg.task == "peg_in_hole":
            reward += self.settle_bonus_w * self._hold

        info["episode_collided"] = self._episode_collided
        return float(reward), bool(terminated), bool(success), info

    # ------------------------------------------------------------------ #
    def _success(self, s: dict):
        cfg = self.cfg
        ee, peg = s["ee_pos"], s["peg_pos"]

        if cfg.task == "reach":
            ok = np.linalg.norm(ee - s["goal_pos"]) < cfg.reach_success_dist
            return ok, ok

        if cfg.task == "pick_place":
            # The target is a position to bring the peg to and let go of -- it
            # is in mid-air, so requiring the released peg to then stay settled
            # there is unsatisfiable. Success = the peg reached the tolerance
            # while held, and was then released within a short window.
            at_target = np.linalg.norm(peg - s["goal_pos"]) < cfg.place_success_dist
            if at_target and s["grasped"]:
                self._reached_target = True
            if not cfg.require_release:
                ok = at_target and s["grasped"]
            else:
                self._release_window = getattr(self, "_release_window", 0)
                if self._reached_target and s["grasped"]:
                    self._release_window = 15         # steps left to let go
                elif self._release_window > 0:
                    self._release_window -= 1
                ok = (self._reached_target and not s["grasped"]
                      and self._release_window > 0
                      and np.linalg.norm(peg - s["goal_pos"]) < 2.0 * cfg.place_success_dist)
            if cfg.collision_is_failure and self._episode_collided:
                # Collision -> IMMEDIATE terminal failure, not just "not a
                # success held open until `ok` is met". Previously the
                # episode kept running after a collision (only `success`
                # was forced False), so there was nothing stopping the
                # policy from colliding to cut a corner and then still
                # finishing the task for however much reward remained.
                # Ending the episode right here removes that: colliding
                # forfeits all further reward (including _SUCCESS_BONUS),
                # so cutting through an obstacle is never a viable shortcut.
                # (collision_terminal, live-settable, defaults True = exactly
                # this behavior; CollisionCurriculum can hold it False early.)
                return False, self.collision_terminal
            return ok, ok

        # peg_in_hole
        xy_err = float(np.linalg.norm(peg[:2] - s["hole_xy"]))
        tilt_deg = float(np.degrees(s["peg_tilt_rad"]))
        depth = float(s["peg_depth"])
        self.stats["max_depth"] = max(self.stats["max_depth"], depth)
        self.stats["min_xy_err"] = min(self.stats["min_xy_err"], xy_err)
        self.stats["min_tilt_deg"] = min(self.stats["min_tilt_deg"], tilt_deg)
        self.stats["min_keypoint_dist"] = min(self.stats["min_keypoint_dist"], self._keypoint_dist(s))

        if cfg.align_only:
            # Task 1 ("reach+align", SeqPolicy-style decomposition -- see
            # PhaseConfig.align_only): success is reaching + HOLDING the real
            # xy/tilt tolerance (not the loosened 3x used by the one-time
            # _ALIGN_BONUS shaping gate above) while grasped -- no depth
            # requirement at all. This hands off a genuinely tight,
            # insertion-ready pose to Task 2 (cfg.insert_only) rather than
            # the loose "roughly above the hole" state the monolithic
            # _ALIGN_BONUS gate settles for.
            task1_aligned = (
                xy_err < cfg.insert_xy_tol
                and tilt_deg < cfg.insert_tilt_tol_deg
                and bool(s["grasped"])
            )
            self._hold = self._hold + 1 if task1_aligned else 0
            held = self._hold >= cfg.insert_hold_steps
            if cfg.collision_is_failure and self._episode_collided:
                return False, self.collision_terminal
            return held, held

        aligned = (
            depth >= cfg.insert_success_depth
            and xy_err < cfg.insert_xy_tol
            and tilt_deg < cfg.insert_tilt_tol_deg
        )
        self._hold = self._hold + 1 if aligned else 0
        held = self._hold >= cfg.insert_hold_steps
        if cfg.collision_is_failure and self._episode_collided:
            # Same immediate-terminal fix as pick_place, above -- was gated
            # behind `held`, so a mid-transit collision (Phase 4 does have
            # obstacles, num_obstacles=3) didn't actually end the episode
            # until/unless the insert-hold condition was separately met.
            return False, self.collision_terminal
        return held, held
