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
_HOVER_HEIGHT = 0.10     # m, pre-grasp hover target above the peg (Phase 3+
                         # only) -- aligns above before descending, so the
                         # approach can't cut diagonally through clutter.
_HOVER_XY_TOL = 0.03     # m, xy tolerance to call the EE "aligned above the
                         # peg" and switch the target from hover to the peg.
_GRASP_BONUS = 12.0      # one-time grasp bonus; has to outweigh the "cost"
                         # of committing to the harder carry sub-goal.
_HOLD_BONUS = 0.03       # per-step reward for holding the peg, paid only
                         # before the target is reached (see compute()) --
                         # after that, holding forever would be a free trap.
_HOLD_AT_TARGET_PEN = 0.05   # per-step pressure to release once at the target
_CARRY_CHECKPOINT_BONUS = 2.0  # one-time, first time the held peg gets within
                               # 2x the place tolerance (shortens the
                               # credit-assignment horizon for carrying)
_ALIGN_BONUS = 3.0
_DEPTH_PROGRESS_BONUS = 1.0   # one-time, per cm of new real insertion depth
                              # reached. The continuous depth term alone is
                              # too weak against the risk of a tilt/xy-
                              # disrupting rim contact, so without this a
                              # policy can learn to hover aligned and never
                              # push in; this discrete bonus is never un-paid.
_DEPTH_PROGRESS_STEP_CM = 1
_DEPTH_PROGRESS_CAP_CM = 5    # cap a little past insert_success_depth (~3cm)
_ACT_REG = 0.005
_JERK_REG = 0.002
_TIME_PEN = 0.01         # also cancels the small reward shaping would
                         # otherwise give for hovering near the goal
# Collision and repulsion penalties are higher while carrying the peg than
# empty-handed, since carrying near an obstacle is the harder, riskier case.
_COLLISION_PEN_FREE = 1.2    # not grasped
_COLLISION_PEN_CARRY = 1.75  # grasped
_REPULSE_MARGIN = 0.12   # m -- start pushing back on any arm link (+ the peg,
                         # if held) inside this distance to the nearest
                         # obstacle. Dense "veer away" gradient, separate
                         # from the sparse on-contact collision penalty.
_REPULSE_W_FREE = 0.5    # penalty at d=0, ramping to 0 at _REPULSE_MARGIN,
                         # not grasped. Tuned to cut carry-time collisions
                         # without costing too much carry-completion time.
_REPULSE_W_CARRY = 1.0   # grasped -- above the empty-handed weight, since
                         # carrying near an obstacle is riskier.
# Utility-based RL (arXiv:2402.02665) reformulations of two reward terms.
# Both default OFF; every existing checkpoint's eval semantics are
# unaffected unless a run explicitly opts in.
_SATISFICING_SUPPRESS_FRAC = 0.5   # satisficing_proximity: narrows the
                         # goal-shaping suppression radius (see `in_danger`
                         # in compute()) to a tighter sub-threshold instead
                         # of the full repulsion margin, so shaping is only
                         # suppressed when actually close to danger.
_RISK_SEEKING_ENTRY_XY = 0.05   # risk_seeking_depth: the peg_in_hole xy_err
                         # radius inside which the policy is genuinely
                         # attempting entry, not just approaching.
_RISK_SEEKING_PENALTY_CAP = 0.10   # risk_seeking_depth: caps the xy/tilt
                         # penalty portion of the potential while genuinely
                         # attempting entry, so a bad outcome doesn't
                         # outweigh the incentive to try at all. The depth
                         # bonus term stays uncapped.
_RING_REPULSE_MARGIN = 0.06   # m -- dense "veer away" gradient for the
                              # forearm against the hole ring specifically.
                              # Tighter than _REPULSE_MARGIN since the
                              # ring's own footprint is small.
_RING_REPULSE_W = 1.5        # peg_in_hole only, applies regardless of
                              # grasped state.
_INVALID_PEN = 0.05
_SUCCESS_BONUS = 50.0    # dominates the shaped trajectory reward -> reach the
                         #   threshold, don't camp near it
_GRASP_SHAPE_W = 0.08    # per-step nudge on "gripper closing while near peg".
_IN_RANGE_BONUS = 0.05   # small per-step bonus for having the EE inside the
                         # grasp trigger radius, distinct from grasping --
                         # rewards nailing the approach, not just closing.
_GRASP_TRIGGER_M = 0.09  # mirrors env._GRASP_TRIGGER (EE-peg dist that lets a
                         # gripper-close actually form the grasp constraint)
_GRASP_MISS_PEN = 0.02   # small per-step penalty for a failed grasp attempt
                         # (gripper closed, but too far for the constraint to
                         # form). Deliberately tiny, since the grasp
                         # threshold is set low on purpose and this fires
                         # often early in training by design.

# Isolated wrist angular-velocity penalty, pre-grasp only, joints 5/6/7 only.
# Separate from _ACT_REG/_JERK_REG, which regularize the commanded action,
# not measured joint velocity. Threshold set above normal reaching speed
# and well below collision-impact spikes, so it targets genuinely
# fast/aggressive wrist motion, not ordinary reaching.
_WRIST_VEL_JOINTS = (4, 5, 6)   # 0-indexed = joints 5,6,7
_WRIST_VEL_THRESH = 2.0         # rad/s
_WRIST_VEL_PEN_W = 0.02         # per rad/s of excess, per joint, summed

_GRASP_DROP_PEN_DEFAULT = 2.0    # peg_in_hole only: one-time penalty on the
                         # exact grasped True->False transition. Without
                         # this the reward had no direct cost for dropping a
                         # formed grasp, so the policy would cycle grasp and
                         # drop instead of holding. peg_in_hole has no
                         # require_release mechanic, so releasing early is
                         # never legitimate. Sized above the small per-step
                         # terms but below _GRASP_BONUS, so one honest grasp
                         # is never net-negative but repeated flickering
                         # compounds into a real cost. Default for
                         # RewardComputer.grasp_drop_pen (live-settable).

_ARM_JITTER_THRESH = 1.5   # rad/s -- lower than _WRIST_VEL_THRESH since this
                           # gates on the post-grasp fine-hold phase (the arm
                           # not settling into a steady hold near the
                           # target), across all arm joints, not just wrist.
_ARM_JITTER_PEN_W = 0.03  # per rad/s of excess, per joint, summed

_LINGER_STEPS_CAP = 20     # cap on the linger penalty's growth (see
                           # `linger_pen` in compute()), so it doesn't grow
                           # unboundedly over a long lingering stretch.

_SETTLE_BONUS_PER_STEP = 0.15  # peg_in_hole only: dense reward scaled by
                           # self._hold (the consecutive-steps-aligned
                           # counter _success() computes). Without this, the
                           # only reward tied to the success predicate was
                           # the sparse all-or-nothing _SUCCESS_BONUS at the
                           # full hold, so nothing rewarded sustaining a
                           # near-precise alignment once reached.


class RewardComputer:
    """Per-episode reward state machine. One instance per env, reset each episode."""

    def __init__(self, cfg: PhaseConfig, gamma: float = 0.99):
        self.cfg = cfg
        self.gamma = gamma
        # Dense potential-shaping's xy_err coefficient (see _potential,
        # peg_in_hole branch) -- a training-progress value, not per-episode
        # state, so set here (persists across reset()) not in reset().
        # Live-updatable via env.set_xy_err_weight to soften the
        # pull-toward-the-hole gradient as training progresses.
        self.xy_err_weight = 1.5
        # Live-settable (mirrors xy_err_weight). A peg held well off-axis has
        # a footprint far wider than the bore's clearance permits regardless
        # of how good xy is, so this weights orientation heavily too.
        self.tilt_weight = 1.2
        # Live-settable, mirrors xy_err_weight -- see _GRASP_DROP_PEN_DEFAULT.
        # A curriculum callback can ramp this up from a lower starting value.
        self.grasp_drop_pen = _GRASP_DROP_PEN_DEFAULT
        # Fixed (non-curriculum) magnitude for the linger penalty -- see the
        # `linger_pen` block in compute(). 0.0 = disabled (default).
        self.linger_pen_w = 0.0
        # Live-settable magnitude for the settle bonus (see
        # _SETTLE_BONUS_PER_STEP). Defaults to the module constant; set to
        # 0.0 to disable it for an isolated ablation.
        self.settle_bonus_w = _SETTLE_BONUS_PER_STEP
        # Live-settable collision levers, decoupled from cfg.collision_is_
        # failure (which stays fixed -- collision always precludes success,
        # unconditionally, for every phase-3+ config; see _success()).
        #   collision_pen_mult: multiplies the per-step collision penalty
        #     (default 1.0 = full strength). A curriculum can ramp this up.
        #   collision_terminal: whether a collision also ends the episode
        #     immediately (default True). Set False to let a still-learning
        #     policy keep experiencing the rest of an episode after a
        #     collision, instead of being cut off before it can learn
        #     grasp/align at all.
        self.collision_pen_mult = 1.0
        self.collision_terminal = True
        # Live-settable arm-jitter penalty magnitude (see the `arm_jitter_pen`
        # block in compute() and the constants above). Defaults to the
        # module constant.
        self.arm_jitter_pen_w = _ARM_JITTER_PEN_W
        # Live-settable keypoint-docking potential coefficient (see
        # _keypoint_dist). 0.0 = off (default) -- every existing
        # run/config/checkpoint is unaffected until this is explicitly set.
        self.keypoint_w = 0.0
        # Utility-based RL (arXiv:2402.02665), both opt-in, both False by
        # default -- see the constants above for the mechanism of each.
        # False reproduces every existing run's exact reward semantics.
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
        circular shift of the matching, not a full permutation) and takes
        the lowest-distance one, rather than pairing peg keypoint i with
        hole keypoint i by a fixed absolute-angle label. The peg is a
        rotationally symmetric cylinder, so any relative yaw fits the bore
        equally well -- a fixed label match would penalize perfectly valid
        insertions at the "wrong" yaw. Circularly searching for the best
        alignment each step makes the term yaw-invariant while still
        penalizing what actually blocks entry: a tilted peg's footprint
        spreading away from the bore wall on one side. K<=8 here, so the
        O(K^2) search is negligible per step."""
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
                # Staged Z-aligned approach (Phase 3+ only): align above the
                # peg (xy only) before rewarding any descent, so there's no
                # gradient toward cutting diagonally through obstacles near
                # the peg.
                xy_err = float(np.linalg.norm(ee[:2] - peg[:2]))
                if xy_err > _HOVER_XY_TOL:
                    target = np.array([peg[0], peg[1], peg[2] + _HOVER_HEIGHT])
                else:
                    target = peg
                return -float(np.linalg.norm(ee - target))
            return -d_ee_peg

        if task == "pick_place":
            # Constant offset so the potential does not jump sharply negative
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
        # shaping-only wider-gated depth; success still checks the tight
        # `peg_depth` independently in compute() below.
        depth = s.get("peg_depth_shaped", s["peg_depth"])
        # keypoint-docking term: an additional, opt-in potential term, added
        # alongside xy_err/tilt/z_gap/depth below, not replacing them.
        kp_pen = self.keypoint_w * self._keypoint_dist(s)
        # once roughly aligned above the mouth, reward downward progress.
        if self.cfg.align_only:
            # Task 1a/"airspace align": the peg only has to reach the
            # airspace column above the hole, any height, xy/tilt-aligned.
            # The z_gap/depth terms below would pull it down toward the
            # mouth, the wrong incentive here, so this branch skips them.
            return -(self.xy_err_weight * xy_err) - (self.tilt_weight * tilt) - kp_pen
        penalty = (self.xy_err_weight * xy_err) + (self.tilt_weight * tilt)
        # risk_seeking_depth (arXiv:2402.02665 Sec. 4.2, opt-in): caps the
        # xy/tilt penalty while genuinely attempting entry, so a bad outcome
        # (rim-clip disturbance) doesn't outweigh the incentive to attempt
        # descent at all. The depth bonus term below stays uncapped.
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
        # held peg) is inside the obstacle danger radius (Phase 3+): without
        # this, goal-approach shaping can outweigh the repulsion penalty on
        # some paths, making "cut through the danger zone" net-positive.
        # `self._phi` still updates every step, so shaping resumes cleanly
        # once the trajectory exits the zone.
        # satisficing_proximity (opt-in): narrows the suppression radius to
        # a tighter sub-threshold, so "safe enough" suppresses nothing, only
        # genuine proximity does. The repulsion penalty below is unchanged.
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
        # (2-6); phase 1 (reach, no peg) is unaffected.
        grasp_shape = 0.0
        if cfg.use_peg and not s["grasped"] and not self._picked:
            d_ee_peg = float(np.linalg.norm(s["ee_pos"] - s["peg_pos"]))
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
            # loosened to 3x insert_xy_tol -- this is a one-time SHAPING
            # bonus, not the success predicate, which still requires 1x.
            if xy_err < 3.0 * cfg.insert_xy_tol and s["peg_tilt_rad"] < np.radians(cfg.insert_tilt_tol_deg):
                gate += _ALIGN_BONUS
                self._align_bonus_paid = True
        if cfg.task == "peg_in_hole" and s["grasped"]:
            # Reads the same tip-based `peg_depth_shaped` `_potential()`
            # uses, so the discrete +1/cm checkpoints and the continuous
            # shaping gradient agree on what "progress" means.
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
        # All arm joints, not just wrist -- targets imprecise, unsettled
        # fine-hold behavior once something is grasped, whether holding
        # still above a hole or carrying toward a place target.
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
        # grasp range without grasping (peg_in_hole only, disabled unless
        # self.linger_pen_w > 0). Grows linearly with consecutive
        # in-range-not-grasped steps (capped at _LINGER_STEPS_CAP), resets
        # the moment it grasps or leaves range -- cheap to escape by just
        # committing, closing the "hover forever for free reward" loophole.
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
                # Collision -> immediate terminal failure, so colliding to
                # cut a corner and finishing the task anyway is never a
                # viable shortcut.
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
            # Task 1 ("reach+align"): success is reaching + HOLDING the real
            # xy/tilt tolerance (not the loosened 3x used by the one-time
            # _ALIGN_BONUS shaping gate above) while grasped, no depth
            # requirement. Hands off a tight, insertion-ready pose to Task 2
            # rather than the loose "roughly above the hole" state the
            # monolithic _ALIGN_BONUS gate settles for.
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
            # Same immediate-terminal rule as pick_place above: a mid-transit
            # collision ends the episode instead of waiting on insert-hold.
            return False, self.collision_terminal
        return held, held
