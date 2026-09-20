#!/usr/bin/env python3
"""
scripted_expert_ur5.py -- UR5 (6-DOF) PORT of scripts/scripted_expert.py, built to test
whether E68's descent-phase xy-drift finding (EXPERIMENTS.md) is a symptom of KUKA iiwa's
redundant 7th DOF. Identical control logic to the original (same state machine, same
peg-frame FK control from E65/E66, same commit-gate/descend-ramp/xy-feedback parameters from
E68) -- only the imports are repointed to manipularl_ur5. See the plan this was built from.

scripted_expert.py -- a hand-coded IK controller that solves pick-and-place
(Phase 2) and, when the phase calls for it, obstacle avoidance (Phase 3+)
and peg-in-hole insertion (Phase 4+). Used to generate demonstrations for
behaviour cloning ("Track B" -- see the Phase 3-6 plan / EXPERIMENTS.md).

It is NOT a submission policy -- it reads privileged sim state (exact peg,
goal, hole and obstacle positions) and drives the arm with inverse
kinematics. Its only job is to produce successful (obs, action)
trajectories so a policy can be BC-warm-started past a hard-exploration
wall, the way it did for Phase 2 (unused there in the end -- from-scratch
RL solved Phase 2 first -- but the pipeline is proven: 80% expert success,
BC val-MSE 0.038).

State machine, all targets in the arm base frame:
    APPROACH  -> EE 10 cm above the peg, gripper open
    DESCEND   -> EE 2 cm above the peg, gripper open
    GRASP     -> gripper closed, hold a few steps until the constraint forms
    LIFT      -> EE 15 cm above the peg's pick spot, gripper closed
  pick_place (Phase 2/3):
    CARRY     -> EE at the place target, gripper closed
    RELEASE   -> gripper open once the peg is within the place tolerance
  peg_in_hole (Phase 4+):
    ALIGN     -> peg centred above the hole mouth, clearance height
    INSERT    -> peg descends into the bore and holds (no release --
                 peg-in-hole's success predicate doesn't require it)

Obstacle avoidance (Phase 3+, cfg.num_obstacles > 0): a classical repulsive
offset from the nearest active obstacle is added to the IK target during
transit states (APPROACH/DESCEND-transit/LIFT/CARRY/ALIGN) -- not during the
precision states (GRASP/INSERT), where the goal point may legitimately sit
close to an obstacle (10 cm keepout on centres only) and repulsion would
fight the approach itself.
"""

import numpy as np
import pybullet as p

from manipularl_ur5.env import ManipulaRLEnv, _EE_LINK
from manipularl_ur5.configs import ARM_DOF, DELTA_Q_SCALE, get_phase

_AVOID_MARGIN = 0.15   # m, start steering away inside this distance
_AVOID_GAIN = 1.2      # offset magnitude at contact, 0 at the margin
_NON_AVOID_STATES = ("GRASP", "INSERT")

# Compliant-insertion constants (asking myself "is there a better way to get
# expert demos than IK" -- see EXPERIMENTS.md's Stage-1 tilt-lock diagnosis
# for the motivating finding these two levers act on).
_REORIENT_RATE = 1.0 / 50   # slerp step per env step -> full vertical
                            # correction over ~50 steps (~2.5s @ 20Hz), NOT a
                            # hard snap -- see _slerp's docstring for why a
                            # gradual correction was needed instead of the
                            # instant re-orientation I already
                            # tried and reverted (it broke ALIGN's position
                            # convergence, the peg swinging through the arc).
                            # Measured 3 rates directly (N=100 each,
                            # apples-to-apples): 1/20 (fast) wrecked position
                            # tracking hardest (align_ok conditional 0.056 --
                            # a bigger per-step swing destabilises IK more);
                            # 1/80 (slow) recovered align_ok best (0.643) but
                            # left more residual tilt error whenever it DID
                            # transition, costing enter_ok (conditional
                            # 0.067, worse than baseline's 0.089); 1/50 (this
                            # value) is the middle of that tradeoff, not a
                            # clear winner over either -- an inherent tension
                            # given the peg is RIGIDLY attached (swings as
                            # the wrist reorients, whatever the rate), not a
                            # tuning oversight.
_STUCK_STEPS = 15           # INSERT: steps of no depth progress before the
                            # spiral search kicks in
_SEARCH_RADIUS_RATE = 0.0003   # m added to spiral radius per stuck-step
_SEARCH_MAX_RADIUS = 0.008     # m -- a bit past the ~1.8cm bore inradius's
                                # own clearance budget, bounded so the search
                                # never wanders outside a plausible catch zone
_SEARCH_ANGULAR_RATE = 0.6     # rad added to spiral angle per stuck-step

# Peg-frame FK control (reorient_mode="pegframe"): the slerp scheme above
# aims the flange at `desired_peg_point + (ee - peg)`, an offset measured at
# the CURRENT orientation, while separately slerping a DIFFERENT target
# orientation for that same step -- the two disagree by up to
# `_GRASP_OFFSET_Z * sin(tilt)` (env.py's peg-in-hole grasp offset is 3.5cm,
# so 1.2-2.25cm of peg drift at the diagnosed 20-40deg tilts), fighting
# ALIGN's own 3.5mm xy gate. This mode instead solves ONE FK problem per
# step: given the peg's desired pose, and the peg's pose measured relative
# to the flange (re-estimated every step so grasp compliance/sag is
# absorbed, not just captured once), back out the flange pose IK must hit --
# position and orientation can no longer disagree because both come from
# the same desired peg pose.
_TILT_OK_DEG = 5.0       # measured peg tilt considered "vertical enough" to
                         # commit to INSERT -- inside the env's own 10deg
                         # success tolerance, not just the slerp timer's
                         # nominal endpoint
_TILT_HOLD_STEPS = 5     # consecutive steps _TILT_OK_DEG must hold before
                         # the ALIGN->INSERT gate fires under "pegframe"

# "pegframe"-only ALIGN->INSERT position/height gate (diagnosed via direct
# per-step logging, see EXPERIMENTS.md's follow-up entry): the ORIGINAL
# 3.5mm xy / 1.5cm height gate below was tuned for position-only IK with NO
# reorientation happening at all, and requires ALL of xy/height/tilt under
# their (tight) bars at the SAME instant. Measured directly, that combined
# instant never once occurred in 100 pegframe episodes -- not because any
# ONE bar is unreachable (each is individually hit; in one traced episode
# min_xy=0.13cm, min_height=0.16cm, min_tilt=3.4deg, all comfortably under
# their bars) but because active reorientation keeps the peg in gentle
# continuous motion, so the three minima land at DIFFERENT timesteps rather
# than overlapping. Loosening to the tolerances the environment's own
# success predicate actually requires (not an extra pre-reorientation-era
# safety margin) gives simultaneity a real chance without giving up genuine
# precision -- and INSERT's own target keeps re-aiming at hole_xy exactly
# every step regardless of the exact commit height (see INSERT's own
# comment), so a few extra cm of starting height costs little.
_PEGFRAME_ALIGN_XY_TOL = 0.010       # == cfg.insert_xy_tol
_PEGFRAME_ALIGN_HEIGHT_TOL = 0.03    # 2x the original 1.5cm

# ALIGN->INSERT descend ramp (found via direct per-step tracing after the
# gate-loosening fix above finally let episodes commit at all): INSERT's own
# z-target used to jump INSTANTLY, the very first INSERT step, from ALIGN's
# clearance height (mouth+8cm) to the full insertion depth target (mouth-4cm)
# -- a 12cm discontinuous commanded jump. Traced directly (3 episodes,
# "pegframe"): tilt spiked from ~3deg at commit to 15-22deg within 2-3 steps
# of entering INSERT, and xy_err simultaneously ballooned from <1cm to 4-5cm
# -- a sudden IK-solve instability at the discontinuity, not a slow drift,
# and the exact reason every one of the newly-committing episodes still
# landed on the rim (0/5 made any depth progress). Ramping the z-target
# smoothly over `_DESCEND_RATE`'s schedule (same pattern as `_REORIENT_RATE`
# already uses for orientation) removes the discontinuity instead of asking
# IK to jump a well-converged pose 12cm in one step.
_DESCEND_RATE = 1.0 / 40   # ramp step per env step -> full descend-target
                            # transition over ~40 steps (~2s @ 20Hz)


def _slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions (any consistent
    component order -- treated as a plain 4-vector, so this works for
    PyBullet's [x,y,z,w] convention without needing to know it explicitly).

    WHY this exists: a first attempt at fixing the tilt-lock bug forced the
    orientation TARGET to true-vertical instantly at the GRASP->LIFT
    transition. Measured directly, that broke ALIGN's own position
    convergence (align_ok conditional 0.848->0.028) -- the peg is RIGIDLY
    attached to the flange, so an instant multi-cm orientation jump swings
    the peg's actual xy position through the reorientation arc, and IK
    re-solving position+the-new-orientation simultaneously each step fights
    its own previous solution rather than settling. Slerping the TARGET
    itself toward vertical over many steps (instead of jumping the target)
    means the arm is always solving for a target that's only slightly
    different from where it already is -- smooth, continuous correction
    instead of a discontinuous constraint change.
    """
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:          # take the shorter arc
        q1, dot = -q1, -dot
    if dot > 0.9995:        # nearly parallel -- linear blend is numerically safer
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


class ScriptedManipulator:
    def __init__(self, env: ManipulaRLEnv, phase: int = 2, reorient_mode: str = "slerp",
                 ik_iters: int = 80, ik_threshold: float = 1e-4,
                 descend_rate: float = _DESCEND_RATE, xy_feedback_gain: float = 0.0):
        """`reorient_mode` (peg_in_hole only; ignored for reach/pick_place):
            "slerp"    -- default, bit-for-bit the original E58 scheme (open-
                          loop flange-orientation timer, gates ALIGN->INSERT
                          on the timer's own progress, not a measurement).
            "none"     -- pre-E58 baseline, no reorientation at all (gate
                          reduces to the original xy/height-only check).
            "pegframe" -- FK-based control of the PEG's pose directly, closed
                          on measured tilt. See the module docstring above
                          _TILT_OK_DEG.

        `ik_iters`/`ik_threshold` -- passed to every `_ik()` call (default:
        the original, never-revisited values). `descend_rate` -- overrides
        `_DESCEND_RATE` per-instance. `xy_feedback_gain` -- "pegframe" mode's
        INSERT state only; 0.0 (default) = today's open-loop behavior (aim
        at hole_xy, trust IK to hold it). See EXPERIMENTS.md's descent-drift
        follow-up entry for why these three exist -- candidate fixes for a
        slow xy drift found during INSERT even with tilt/z-jump already
        fixed, none of which change behavior at their default values.
        """
        if reorient_mode not in ("slerp", "none", "pegframe"):
            raise ValueError(f"reorient_mode must be slerp/none/pegframe, got {reorient_mode!r}")
        self.env = env
        self.phase_num = phase
        self.cfg = get_phase(phase)
        self.avoid = self.cfg.num_obstacles > 0
        self.insertion = self.cfg.task == "peg_in_hole"
        self.reorient_mode = reorient_mode
        self.ik_iters = int(ik_iters)
        self.ik_threshold = float(ik_threshold)
        self.descend_rate = float(descend_rate)
        self.xy_feedback_gain = float(xy_feedback_gain)
        self.reset()

    def reset(self):
        self.state = "APPROACH"
        self._grasp_hold = 0
        # BUGFIX (found while diagnosing the tilt-lock fix above): this never
        # cleared `_grip_quat`, so once episode 1 set it, `act()`'s condition
        # (`self.state != "APPROACH" and hasattr(self, "_grip_quat")`) made
        # EVERY subsequent episode's DESCEND phase also target that stale
        # orientation -- reproducing the exact "orientation fights the reach,
        # grasp_ok collapses" regression regardless of which state the fix
        # above set _grip_quat in, from episode 2 onward. `hasattr` needs a
        # true absence, not a reset value, to correctly gate DESCEND back to
        # position-only IK each episode.
        for attr in ("_grip_quat", "_grip_quat_target", "_grip_quat_start",
                     "_peg_in_flange", "_peg_quat_start", "_peg_quat_target",
                     "_peg_target_pt", "_align_clear_z"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._reorient_t = 0.0
        self._peg_reorient_t = 0.0
        self._tilt_hold = 0
        self._descend_ramp_t = 0.0
        self._best_depth = 0.0
        self._stuck_steps = 0

    def reorient_progress(self) -> float:
        """Linear progress (0..1) of the gradual GRASP->LIFT reorientation
        toward vertical -- see _slerp's docstring and the GRASP->LIFT
        transition. 0.0 before any grasp has happened yet."""
        return float(getattr(self, "_reorient_t", 0.0))

    def _ik(self, target_pos, target_quat=None):
        kw = dict(maxNumIterations=self.ik_iters, residualThreshold=self.ik_threshold,
                  physicsClientId=self.env.client)
        if target_quat is not None:
            jt = p.calculateInverseKinematics(
                self.env.robot_id, _EE_LINK, list(target_pos), list(target_quat), **kw)
        else:
            jt = p.calculateInverseKinematics(
                self.env.robot_id, _EE_LINK, list(target_pos), **kw)
        jt = np.asarray(jt[:ARM_DOF], dtype=np.float64)
        # Clip to the arm's own joint limits, matching env.py's `_ik_limited`
        # (whose docstring documents this exact gap: an unclipped solution
        # can violate the URDF's limits, and the caller then acting on it
        # produces "tilt 3->89 degrees, xy drift to 15+cm, within ~5 steps").
        # `env.step()` clips the FINAL joint target after the delta-action is
        # applied (env.py:933), so this isn't the only line of defense, but
        # an out-of-range `jt` still skews the raw (jt - q) delta direction
        # this method hands back before that later clip ever sees it.
        q_lower = getattr(self.env, "_q_lower", None)
        q_upper = getattr(self.env, "_q_upper", None)
        if q_lower is not None and q_upper is not None:
            jt = np.clip(jt, q_lower, q_upper)
        return jt

    def _avoid_offset(self, points) -> np.ndarray:
        """Repulsive offset away from the nearest active obstacle to any of
        `points` (world xyz), zero beyond _AVOID_MARGIN. Not a planner --
        just a nudge on top of the waypoint state machine."""
        if not self.avoid or self.state in _NON_AVOID_STATES:
            return np.zeros(3)
        active = getattr(self.env, "_obstacle_active", 0)
        if active == 0:
            return np.zeros(3)
        total = np.zeros(3)
        for i in range(active):
            opos = np.array(p.getBasePositionAndOrientation(
                self.env.obstacle_ids[i], physicsClientId=self.env.client)[0])
            for pt in points:
                delta = np.asarray(pt) - opos
                d = float(np.linalg.norm(delta))
                if d < _AVOID_MARGIN:
                    direction = delta / d if d > 1e-6 else np.array([0.0, 0.0, 1.0])
                    total += direction * _AVOID_GAIN * (_AVOID_MARGIN - d)
        return total

    def act(self, s: dict) -> np.ndarray:
        """One 8-D action in [-1,1] for the current privileged state dict."""
        ee = np.asarray(s["ee_pos"], float)
        peg = np.asarray(s["peg_pos"], float)
        goal = np.asarray(s["goal_pos"], float)
        grasped = bool(s["grasped"])
        q = np.asarray(s["q"], float)

        if self.insertion and self.reorient_mode == "pegframe" and grasped \
                and self.state in ("LIFT", "ALIGN", "INSERT"):
            # Re-estimate the peg<->flange transform EVERY step (not just
            # once at grasp time) -- a no-op under a rigid grasp (the
            # transform is genuinely constant), but tracks real deflection
            # under a compliant grasp (cfg.grasp_max_force) instead of
            # solving FK against a stale snapshot.
            inv_pos, inv_quat = p.invertTransform(list(s["ee_pos"]), list(s["ee_quat"]))
            self._peg_in_flange = p.multiplyTransforms(
                inv_pos, inv_quat, list(s["peg_pos"]), list(s["peg_quat"]))
        if self.insertion:
            tilt_deg = float(np.degrees(s.get("peg_tilt_rad", 0.0)))
            self._tilt_hold = self._tilt_hold + 1 if tilt_deg <= _TILT_OK_DEG else 0

        grip = -1.0
        if self.state == "APPROACH":
            tgt = peg + [0, 0, 0.10]
            if np.linalg.norm(ee - tgt) < 0.03:
                self.state = "DESCEND"
        elif self.state == "DESCEND":
            # Deliberately still position-only IK (no target_quat) -- DESCEND/
            # GRASP need full orientation freedom to actually reach and grip
            # the peg; forcing an orientation constraint here was tried and
            # broke grasping outright (grasp_ok 0.66->0.115). Orientation
            # correction starts at GRASP->LIFT below, once the peg is safely
            # gripped and this constraint no longer fights the reach itself.
            tgt = peg + [0, 0, 0.02]
            if np.linalg.norm(ee - peg) < 0.06:
                self.state = "GRASP"
        elif self.state == "GRASP":
            tgt = peg + [0, 0, 0.02]
            grip = 1.0
            self._grasp_hold += 1
            if grasped and self._grasp_hold > 3:
                self._lift_from = peg.copy()
                self.state = "LIFT"
                # COMPLIANT-INSERTION FIX 1/2 (asking myself "is there a
                # better way to get expert demos than IK"): E56 found the
                # peg is held wherever the redundant DESCEND solution
                # happened to leave it (75% of episodes >10deg tilt, median
                # 18.4deg, n=20) and NOTHING ever corrects it -- a peg tilted
                # 20-30deg has a footprint far wider than the bore's 3mm
                # clearance and cannot physically enter regardless of xy
                # precision. Forcing an instant vertical target broke ALIGN's
                # position convergence instead (see _slerp's docstring). Fix:
                # GRADUALLY slerp the TARGET from wherever it was captured
                # toward true vertical over _REORIENT_RATE's schedule,
                # starting now, so LIFT/ALIGN/INSERT converge position AND
                # orientation together instead of fighting a discontinuous
                # jump.
                if self.reorient_mode == "slerp":
                    self._grip_quat_target = np.asarray(
                        p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
                    self._grip_quat_start = np.asarray(s["ee_quat"], float)
                    self._grip_quat = self._grip_quat_start
                    self._reorient_t = 0.0
                # "pegframe": start the PEG's own (not the flange's) target
                # orientation schedule from its ACTUAL orientation at grasp
                # time. `_peg_in_flange` (the rigid transform IK will solve
                # against) is intentionally NOT captured once here -- it is
                # re-estimated every step below so a compliant grasp's real
                # sag/slip is tracked, not frozen at this instant.
                elif self.reorient_mode == "pegframe":
                    self._peg_quat_start = np.asarray(s["peg_quat"], float)
                    self._peg_quat_target = np.asarray(
                        p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
                    self._peg_reorient_t = 0.0
                    self._tilt_hold = 0
        elif self.state == "LIFT":
            tgt = self._lift_from + [0, 0, 0.15]
            grip = 1.0
            if ee[2] > self._lift_from[2] + 0.10 or not grasped:
                self.state = "ALIGN" if self.insertion else "CARRY"
        elif self.state == "ALIGN":
            # peg-in-hole: get the held peg centred above the mouth at a
            # clearance height before committing to the descent. The bore is
            # tight (peg radius _PEG_RADIUS=0.015, clearance ~3 mm), so this
            # needs mm-level centring, not the ~1 cm tolerance CARRY uses --
            # otherwise the peg clips the rim and never actually descends.
            hole_xy, mouth_z = s["hole_xy"], s["hole_mouth_z"]
            clear_pt = np.array([hole_xy[0], hole_xy[1], mouth_z + 0.08])
            tgt = clear_pt + (ee - peg)   # flange<->peg offset, same trick as CARRY
            self._peg_target_pt = clear_pt   # "pegframe" mode's own desired-peg-point (see act()'s tail)
            grip = 1.0
            xy_err = float(np.linalg.norm(peg[:2] - hole_xy))
            # don't commit to INSERT mid-correction -- "pegframe" gates on the
            # MEASURED tilt actually holding near-vertical (closed loop);
            # "slerp" keeps the original open-loop timer; "none" never
            # reorients, so there is nothing to gate on.
            if self.reorient_mode == "none":
                tilt_ok = True
                xy_gate, height_gate = 0.0035, 0.015
            elif self.reorient_mode == "pegframe":
                tilt_ok = self._tilt_hold >= _TILT_HOLD_STEPS
                # see _PEGFRAME_ALIGN_XY_TOL's comment -- the original 3.5mm/
                # 1.5cm bars, measured directly, never co-occur with tilt_ok
                # under active reorientation even though each is individually
                # reachable; committing at the tolerances success itself
                # needs (not an extra pre-reorientation-era margin) instead
                xy_gate, height_gate = _PEGFRAME_ALIGN_XY_TOL, _PEGFRAME_ALIGN_HEIGHT_TOL
            else:
                tilt_ok = self.reorient_progress() >= 1.0
                xy_gate, height_gate = 0.0035, 0.015
            if xy_err < xy_gate and abs(float(peg[2]) - clear_pt[2]) < height_gate and tilt_ok:
                self.state = "INSERT"
                self._best_depth = 0.0
                self._stuck_steps = 0
                self._align_clear_z = float(clear_pt[2])
                self._descend_ramp_t = 0.0
        elif self.state == "INSERT":
            # ALIGN's entry gate (xy < 3.5 mm, now also tilt-converged) already
            # guarantees a good start; the insert target keeps re-aiming at
            # hole_xy exactly, so small drift during the descent self-corrects.
            #
            # COMPLIANT-INSERTION FIX 2/2: real force/torque feedback was
            # tried first and dropped -- measured directly (n=15 episodes),
            # the raw wrist `ft` reaction is dominated by motion-induced
            # transients, not contact: free-carry (LIFT/ALIGN, no obstruction)
            # already reads median 223N (p90 558N), fully overlapping INSERT's
            # own median 425N even after 10-step smoothing. Not cleanly
            # separable without per-episode calibration this scripted
            # controller has no principled way to do. DEPTH STALL is used
            # instead -- strictly reliable (peg not going deeper for
            # `_STUCK_STEPS` consecutive steps IS the definition of being
            # stuck, whatever the cause) and needs no threshold tuning.
            # Once stuck, run a growing spiral search in xy (classic RCC-style
            # compliant-insertion strategy) instead of grinding statically
            # against the rim -- capped at `_SEARCH_MAX_RADIUS` so it never
            # wanders outside a plausible catch zone.
            hole_xy, mouth_z = s["hole_xy"], s["hole_mouth_z"]
            depth_target = float(self.cfg.insert_success_depth) + 0.01
            depth = float(s.get("peg_depth", 0.0))
            if depth > self._best_depth + 1e-4:
                self._best_depth = depth
                self._stuck_steps = 0
            else:
                self._stuck_steps += 1
            search_xy = np.zeros(2)
            if self._stuck_steps > _STUCK_STEPS:
                k = self._stuck_steps - _STUCK_STEPS
                radius = min(_SEARCH_MAX_RADIUS, _SEARCH_RADIUS_RATE * k)
                angle = _SEARCH_ANGULAR_RATE * k
                search_xy = radius * np.array([np.cos(angle), np.sin(angle)])
            # Ramp the Z target smoothly from ALIGN's own clearance height
            # down to the full insertion depth instead of jumping there
            # instantly the first INSERT step (see _DESCEND_RATE's comment --
            # the instant jump was found, via direct tracing, to destabilize
            # an already-converged xy/tilt commit).
            self._descend_ramp_t = min(1.0, self._descend_ramp_t + self.descend_rate)
            z_start = getattr(self, "_align_clear_z", mouth_z - depth_target)
            z_target = z_start + (mouth_z - depth_target - z_start) * self._descend_ramp_t
            insert_pt = np.array([hole_xy[0] + search_xy[0], hole_xy[1] + search_xy[1],
                                  z_target])
            if self.reorient_mode == "pegframe" and self.xy_feedback_gain != 0.0:
                # Proportional correction using the CURRENT measured peg xy
                # error -- tests whether the descent-phase drift (found
                # tracing a committed pegframe episode: xy_err climbs from
                # <1cm to 5-6cm over ~60 INSERT steps even with tilt/the
                # z-jump already fixed) is a correctable bias (this should
                # collapse it) or genuine per-step noise (this won't help,
                # cheaply ruling out a whole explanation class). 0.0
                # (default) = today's open-loop behavior, unchanged.
                insert_pt[:2] += self.xy_feedback_gain * (hole_xy - peg[:2])
            tgt = insert_pt + (ee - peg)
            self._peg_target_pt = insert_pt   # "pegframe" mode's own desired-peg-point (see act()'s tail)
            grip = 1.0
            # no explicit hold/terminate logic -- the env's own success
            # predicate (depth/xy/tilt held for insert_hold_steps) ends the
            # episode once this is sustained; the expert just keeps aiming here.
        elif self.state == "CARRY":
            # IK positions the FLANGE, but the grasped peg sits ~2 cm below it
            # (offset fixed at grasp time). Aim the flange at goal + that offset
            # so the *peg* lands on the target, not the flange.
            tgt = goal + (ee - peg)
            grip = 1.0
            if np.linalg.norm(peg - goal) < 0.6 * self.cfg.place_success_dist:
                self.state = "RELEASE"
        else:  # RELEASE (pick_place only)
            tgt = goal + (ee - peg)
            grip = -1.0

        # recovery: peg slipped out mid-task -> go pick it up again
        if self.state in ("LIFT", "CARRY", "ALIGN", "INSERT") and not grasped and \
                np.linalg.norm(ee - peg) > 0.12:
            self.state = "APPROACH"
            tgt = peg + [0, 0, 0.10]
            grip = -1.0

        avoid = self._avoid_offset([ee, peg] if grasped else [ee])
        tgt = tgt + avoid

        # Advance the gradual reorientation (see the GRASP->LIFT transition
        # above) every step it's active, regardless of which of LIFT/ALIGN/
        # INSERT we're in -- one continuous correction across all three,
        # not restarted per-state.
        target_quat = None
        if self.reorient_mode == "slerp" and self.insertion and hasattr(self, "_grip_quat_target"):
            # Slerp from the FIXED start snapshot (not the previous step's
            # own output) so `_reorient_t`'s linear ramp is the EXACT
            # geodesic progress fraction -- repeatedly slerping toward a
            # fixed target by a constant rate each step converges
            # geometrically, not linearly (after k steps of rate r you're at
            # fraction 1-(1-r)^k, not k*r), which would silently desync from
            # `_reorient_t` and make ALIGN's `reorient_progress() >= 1.0`
            # gate fire well before the peg is actually vertical.
            self._reorient_t = min(1.0, self._reorient_t + _REORIENT_RATE)
            self._grip_quat = _slerp(self._grip_quat_start, self._grip_quat_target, self._reorient_t)
            target_quat = self._grip_quat

        if (self.reorient_mode == "pegframe" and self.insertion
                and hasattr(self, "_peg_in_flange") and hasattr(self, "_peg_target_pt")
                and self.state in ("ALIGN", "INSERT")):
            # Peg-frame FK control (see the module-level comment above
            # _TILT_OK_DEG): solve for the FLANGE pose that puts the PEG at
            # its desired point AND orientation simultaneously, instead of
            # the position-only `pt + (ee - peg)` trick above combined with
            # a separately-slerped flange orientation. Composing
            # T_flange = T_peg_desired . inverse(T_peg_in_flange) means
            # position and orientation come from the SAME desired peg pose
            # and cannot disagree -- this is the actual fix for the 1.2-
            # 2.25cm drift the "slerp" scheme suffers at the diagnosed tilts.
            self._peg_reorient_t = min(1.0, self._peg_reorient_t + _REORIENT_RATE)
            peg_quat_des = _slerp(self._peg_quat_start, self._peg_quat_target, self._peg_reorient_t)
            peg_pos_des = self._peg_target_pt + avoid   # same repulsive nudge, applied in peg-space
            inv_p, inv_q = p.invertTransform(*self._peg_in_flange)
            flange_pos, flange_quat = p.multiplyTransforms(
                list(peg_pos_des), list(peg_quat_des), inv_p, inv_q)
            tgt, target_quat = np.asarray(flange_pos), flange_quat

        jt = self._ik(tgt, target_quat)
        arm = np.clip((jt - q) / DELTA_Q_SCALE, -1.0, 1.0)
        return np.concatenate([arm, [grip]]).astype(np.float32)


# backward-compatible alias (Phase-2 scripts imported this name)
ScriptedPickPlace = ScriptedManipulator


def run_episode(env: ManipulaRLEnv, expert: ScriptedManipulator, max_steps=None):
    """Run one episode, return (obs_list, act_list, success). obs are RAW env obs."""
    max_steps = max_steps or env.max_steps
    obs, _ = env.reset()
    expert.reset()
    obs_l, act_l = [], []
    success = False
    for _ in range(max_steps):
        s = env._state_dict()
        a = expert.act(s)
        obs_l.append(np.asarray(obs, np.float32))
        act_l.append(a)
        obs, _r, term, trunc, info = env.step(a)
        if info.get("success"):
            success = True
        if term or trunc:
            break
    return obs_l, act_l, success


if __name__ == "__main__":
    # quick self-test: success rate of the scripted expert over N episodes
    #   python scripts/scripted_expert.py [n_episodes] [phase]
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    phase = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    env = ManipulaRLEnv(phase=phase, split="train", seed=0, grasp_curriculum=0.0)
    exp = ScriptedManipulator(env, phase=phase)
    wins = sum(run_episode(env, exp)[2] for _ in range(n))
    print(f"scripted expert (phase {phase}): {wins}/{n} = {100*wins/n:.0f}% success")
    env.close()
