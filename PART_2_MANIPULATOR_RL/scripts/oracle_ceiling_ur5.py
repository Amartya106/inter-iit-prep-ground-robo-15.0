#!/usr/bin/env python3
"""
oracle_ceiling_ur5.py -- UR5 (6-DOF) PORT of scripts/oracle_ceiling.py, measuring the SAME
funnel on the UR5 environment (manipularl_ur5) for direct, apples-to-apples comparison against
every KUKA (7-DOF) oracle number logged in EXPERIMENTS.md (E55/E65/E66/E68) -- the whole point
being to test whether removing the redundant 7th DOF changes the descent-phase xy-drift finding
from E68. Identical instrumentation/CLI to the original; only the imports are repointed.

oracle_ceiling.py -- measure the ENVIRONMENT's attainable Phase-4 success rate.

Every experiment E1-E54 tuned the LEARNER (reward weights, curricula, BC, GAIL,
distillation, model-based dreaming) and none moved Phase 4 above 0.020. This
script asks the question none of them asked: what success rate is attainable in
this environment AT ALL?

The yardstick is `scripts/scripted_expert.py`'s ScriptedManipulator -- a
privileged controller with exact sim state (peg, hole, obstacle poses), IK, and
a hand-written textbook insertion routine. It has no exploration problem, no
credit-assignment problem and no sample-complexity problem. Whatever it scores
is a fair lower bound on what the environment permits, and a ceiling that a
learned policy has no structural reason to exceed by much.

The baseline this reproduces is already on disk, in `runs/collect_demos_phase4.log`:

    [collect] scripted expert 7/800 = 1% success        <- Phase 4
    [collect] scripted expert 438/800 = 55% success     <- Phase 3

0.875% on Phase 4 versus 55% on Phase 3 -- reach/grasp/avoid is fine, the entire
collapse is insertion. The learned policies (0.015-0.020) have been BEATING this
oracle by ~2x for a long time.

What this script adds over that one bottom-line number is per-stage attribution,
so the next fix is aimed at the stage that actually fails rather than guessed at:

    grasp_ok  -> the peg was ever picked up
    align_ok  -> grasped, still above the mouth plane, and within the success xy
                 tolerance (insert_xy_tol) -- i.e. the arm could position for a
                 descent at all (reachability + control-resolution ceiling)
    enter_ok  -> the peg centre got >5mm below the mouth plane (did it enter, or
                 did it sit on the rim?)
    depth_ok  -> reached insert_success_depth (the insertion-physics ceiling)
    success   -> the env's own predicate: depth AND xy AND tilt, held for
                 insert_hold_steps consecutive steps, no obstacle collision

plus, for failures, WHY the descent stalled: peg-tip height at max depth, xy
error at the first contact with the hole ring, and which ring box it landed on.
`tests/test_env.py:test_hole_is_solvable` does NOT answer this -- it detaches the
peg from the robot entirely (peg-to-WORLD constraint), teleports it to exact hole
centre, and drives it down with an infinitely stiff constraint. It proves the
bore is not degenerate; it says nothing about whether the ARM can insert.

    PYTHONPATH=. .venv/bin/python scripts/oracle_ceiling.py --episodes 200

Runs on the EVAL seed partition by default, the same layouts `evaluate.py` uses,
so the number is directly comparable to the logged policy results.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p

from manipularl_ur5.env import ManipulaRLEnv, _PEG_HEIGHT
from manipularl_ur5.configs import get_phase
from scripts.scripted_expert_ur5 import ScriptedManipulator


def run_one(env, expert, max_steps):
    """One instrumented episode. Returns a dict of per-episode diagnostics."""
    env.reset()
    expert.reset()
    cfg = env.cfg

    rec = dict(
        success=False, collided=False, ever_grasped=False,
        align_ok=False, enter_ok=False, depth_ok=False,
        reached_insert_state=False,
        max_depth=0.0, min_xy_err_grasped=np.inf, min_xy_err_above_mouth=np.inf,
        steps=0, ring_contact_steps=0,
        first_ring_contact_xy_err=None, first_ring_contact_box=None,
        tip_z_at_max_depth=None, final_state="APPROACH",
        tilt_at_insert_deg=None,
    )
    states_seen = set()
    prev_state = expert.state

    for t in range(max_steps):
        s = env._state_dict()
        a = expert.act(s)
        # Peg tilt (measured, not the reorientation timer's own belief) at
        # the exact ALIGN->INSERT transition -- the same quantity E56's
        # diagnosis reported (median 18.4deg, max 59.8deg, n=20) for the
        # pre-fix scheme; comparable across --reorient-mode arms since it's
        # read from `s`, the state that triggered the transition inside
        # expert.act(), regardless of which mode decided to commit.
        if prev_state != "INSERT" and expert.state == "INSERT" and rec["tilt_at_insert_deg"] is None:
            rec["tilt_at_insert_deg"] = float(np.degrees(s.get("peg_tilt_rad", 0.0)))
        prev_state = expert.state
        states_seen.add(expert.state)
        _obs, _r, term, trunc, info = env.step(a)
        rec["steps"] = t + 1

        s2 = env._state_dict()
        grasped = bool(s2["grasped"])
        peg_pos = np.asarray(s2["peg_pos"], float)
        xy_err = float(np.linalg.norm(peg_pos[:2] - s2["hole_xy"]))
        mouth_z = float(s2["hole_mouth_z"])
        depth = float(s2["peg_depth"])

        if grasped:
            rec["ever_grasped"] = True
            rec["min_xy_err_grasped"] = min(rec["min_xy_err_grasped"], xy_err)
            # "could it position for a descent": still above the mouth plane,
            # and inside the tolerance success itself demands.
            if peg_pos[2] >= mouth_z:
                rec["min_xy_err_above_mouth"] = min(rec["min_xy_err_above_mouth"], xy_err)
                if xy_err < cfg.insert_xy_tol:
                    rec["align_ok"] = True

        if depth > rec["max_depth"]:
            rec["max_depth"] = depth
            rec["tip_z_at_max_depth"] = float(peg_pos[2]) - _PEG_HEIGHT / 2.0

        # contact between the held peg and the hole ring (+ the Stage-1
        # chamfer ring, when cfg.hole_chamfer is on) -- the rim-landing
        # failure mode. Not a "collision" as the env defines it (collisions
        # are obstacles only, env.py:971-976), so nothing else records it.
        ring_bodies = list(env.hole_box_ids) + list(getattr(env, "hole_chamfer_ids", []))
        for bi, bid in enumerate(ring_bodies):
            if p.getContactPoints(bodyA=env.peg_id, bodyB=bid,
                                  physicsClientId=env.client):
                rec["ring_contact_steps"] += 1
                if rec["first_ring_contact_xy_err"] is None:
                    rec["first_ring_contact_xy_err"] = xy_err
                    rec["first_ring_contact_box"] = bi
                break

        if info.get("success"):
            rec["success"] = True
        if info.get("episode_collided") or info.get("collision"):
            rec["collided"] = True
        if term or trunc:
            break

    rec["final_state"] = expert.state
    rec["reached_insert_state"] = "INSERT" in states_seen
    rec["enter_ok"] = rec["max_depth"] > 0.005
    rec["depth_ok"] = rec["max_depth"] >= cfg.insert_success_depth
    for k in ("min_xy_err_grasped", "min_xy_err_above_mouth"):
        if not np.isfinite(rec[k]):
            rec[k] = None
    return rec


def _frac(rows, key):
    return float(np.mean([bool(r[key]) for r in rows])) if rows else 0.0


def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                best=float(a.min()), worst=float(a.max()))


def summarize(rows, cfg, phase, args):
    n = len(rows)
    print(f"\n{'='*66}")
    print(f"ORACLE CEILING -- phase {phase}, {args.split} split, N={n}")
    print(f"privileged scripted IK expert (scripts/scripted_expert.py)")
    print(f"{'='*66}")
    print(f"\nOVERALL")
    print(f"  success        : {_frac(rows,'success'):.4f}   "
          f"({sum(r['success'] for r in rows)}/{n})")
    print(f"  collided       : {_frac(rows,'collided'):.3f}")
    print(f"  mean steps     : {np.mean([r['steps'] for r in rows]):.1f}")

    print(f"\nFUNNEL (fraction of ALL {n} episodes reaching each stage)")
    stages = [
        ("grasp_ok  peg ever picked up", "ever_grasped"),
        ("align_ok  above mouth, xy < %.0fmm" % (cfg.insert_xy_tol * 1000), "align_ok"),
        ("enter_ok  peg centre >5mm past mouth", "enter_ok"),
        ("depth_ok  reached %.0fmm depth" % (cfg.insert_success_depth * 1000), "depth_ok"),
        ("success   + tilt, held %d steps, no collision" % cfg.insert_hold_steps, "success"),
    ]
    prev = 1.0
    for label, key in stages:
        f = _frac(rows, key)
        cond = f / prev if prev > 1e-9 else 0.0
        print(f"  {label:<46} {f:.3f}   (conditional {cond:.3f})")
        prev = f

    tilts = [r["tilt_at_insert_deg"] for r in rows if r["tilt_at_insert_deg"] is not None]
    st = _stat(tilts)
    if st:
        print(f"\nPEG TILT AT ALIGN->INSERT TRANSITION ({len(tilts)}/{n} reached it)")
        print(f"  mean={st['mean']:.1f}deg  median={st['median']:.1f}deg  "
              f"best={st['best']:.1f}deg  worst={st['worst']:.1f}deg")
        print(f"    (E56's pre-fix baseline: median 18.4deg, max 59.8deg, n=20)")

    print(f"\nEXPERT STATE MACHINE (its own final state)")
    for st, c in Counter(r["final_state"] for r in rows).most_common():
        print(f"  {st:<12} {c:>4}  ({100*c/n:.1f}%)")
    print(f"  ever entered INSERT (its own xy<3.5mm gate fired): "
          f"{_frac(rows,'reached_insert_state'):.3f}")

    grasped = [r for r in rows if r["ever_grasped"]]
    print(f"\nPRECISION (of {len(grasped)} grasped episodes)")
    for label, key in [("min xy_err while grasped", "min_xy_err_grasped"),
                       ("min xy_err while above mouth", "min_xy_err_above_mouth")]:
        st = _stat([r[key] for r in grasped])
        if st:
            print(f"  {label:<30} mean={100*st['mean']:.2f}cm  "
                  f"median={100*st['median']:.2f}cm  best={100*st['best']:.2f}cm")
    st = _stat([r["max_depth"] for r in grasped])
    if st:
        print(f"  {'max_depth':<30} mean={100*st['mean']:.2f}cm  "
              f"median={100*st['median']:.2f}cm  best={100*st['worst']:.2f}cm")

    print(f"\nWHY THE DESCENT STALLS")
    ring = [r for r in rows if r["first_ring_contact_xy_err"] is not None]
    print(f"  episodes with peg-vs-hole-ring contact: {len(ring)}/{n} "
          f"({100*len(ring)/n:.1f}%)")
    st = _stat([r["first_ring_contact_xy_err"] for r in ring])
    if st:
        print(f"    xy_err at FIRST ring contact: mean={100*st['mean']:.2f}cm  "
              f"median={100*st['median']:.2f}cm  best={100*st['best']:.2f}cm")
    st = _stat([r["ring_contact_steps"] for r in ring])
    if st:
        print(f"    steps in ring contact:        mean={st['mean']:.1f}  "
              f"median={st['median']:.1f}  worst={st['worst']:.0f}")
    st = _stat([r["tip_z_at_max_depth"] for r in rows])
    if st:
        print(f"  peg-TIP height at max depth: mean={100*st['mean']:.2f}cm  "
              f"median={100*st['median']:.2f}cm  lowest={100*st['best']:.2f}cm")
        print(f"    (mouth plane is at z=10.00cm; tip must reach z=3.00cm for success)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--max-steps", type=int, default=None,
                    help="default: the phase's own max_steps")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="EpisodeSampler index seed. Omit to reproduce the exact "
                         "historical layout sequence (evaluate.py's default).")
    ap.add_argument("--frozen-env", action="store_true", default=True,
                    help="measure the UNMODIFIED environment (current default; "
                         "any of the flags below opt OUT of it for a Stage-1 "
                         "ablation, so the frozen baseline stays reproducible "
                         "whenever none of them are passed).")
    ap.add_argument("--hole-chamfer", action="store_true",
                    help="Stage 1b: enable cfg.hole_chamfer (funnel lead-in above "
                         "the bore, see env._place_chamfer).")
    ap.add_argument("--grasp-max-force", type=float, default=None,
                    help="Stage 1c: enable cfg.grasp_max_force (compliant grasp).")
    ap.add_argument("--fine-control-frac", type=float, default=None,
                    help="Stage 1a: enable cfg.fine_control_frac (slower motion "
                         "near the hole).")
    ap.add_argument("--joint-max-velocity", type=float, default=None,
                    help="Bugfix: cap cfg.joint_max_velocity (rad/s) to prevent "
                         "the stiff-contact ejection artifact (see EXPERIMENTS.md).")
    ap.add_argument("--reorient-mode", default="slerp", choices=["none", "slerp", "pegframe"],
                    help="scripts/scripted_expert.py's ScriptedManipulator reorient_mode: "
                         "'none' = pre-E58 baseline (no reorientation), 'slerp' = E58's "
                         "open-loop flange-orientation timer (default, current behavior), "
                         "'pegframe' = FK-based peg-pose control closed on measured tilt "
                         "(see EXPERIMENTS.md's peg-frame FK entry).")
    ap.add_argument("--ik-iters", type=int, default=80,
                    help="ScriptedManipulator's _ik() maxNumIterations (default: the "
                         "original, never-revisited value). Candidate fix for the "
                         "descent-phase xy-drift follow-up -- env.py's own "
                         "_ik_converge/_ik_limited use up to 200 for a different "
                         "(teleport) use case in this codebase; the scripted expert's "
                         "own IK was never upgraded to match.")
    ap.add_argument("--ik-threshold", type=float, default=1e-4,
                    help="ScriptedManipulator's _ik() residualThreshold (default: the "
                         "original value; try 1e-5 to match _ik_limited).")
    ap.add_argument("--descend-rate", type=float, default=None,
                    help="Override scripted_expert.py's _DESCEND_RATE (default: the "
                         "module constant, 1/40). A slower ramp (e.g. 0.01 = 1/100) "
                         "tests whether the descent-phase xy-drift correlates with "
                         "how fast the z-target changes per step.")
    ap.add_argument("--xy-feedback-gain", type=float, default=0.0,
                    help="'pegframe' mode's INSERT state only: proportional "
                         "correction on the measured peg xy error each step (default "
                         "0.0 = today's open-loop behavior, aim at hole_xy and trust "
                         "IK to hold it). See ScriptedManipulator's docstring.")
    ap.add_argument("--out", default=None, help="optional JSON path for per-episode rows")
    args = ap.parse_args()

    overrides = {}
    if args.hole_chamfer:
        overrides["hole_chamfer"] = True
    if args.grasp_max_force is not None:
        overrides["grasp_max_force"] = args.grasp_max_force
    if args.fine_control_frac is not None:
        overrides["fine_control_frac"] = args.fine_control_frac
    if args.joint_max_velocity is not None:
        overrides["joint_max_velocity"] = args.joint_max_velocity
    env = ManipulaRLEnv(phase=args.phase, split=args.split, seed=0,
                        index_seed=args.eval_seed,
                        cfg_overrides=overrides or None)
    expert_kwargs = dict(reorient_mode=args.reorient_mode, ik_iters=args.ik_iters,
                         ik_threshold=args.ik_threshold, xy_feedback_gain=args.xy_feedback_gain)
    if args.descend_rate is not None:
        expert_kwargs["descend_rate"] = args.descend_rate
    expert = ScriptedManipulator(env, phase=args.phase, **expert_kwargs)
    max_steps = args.max_steps or env.max_steps
    cfg = env.cfg

    rows = []
    for ep in range(args.episodes):
        rows.append(run_one(env, expert, max_steps))
        if (ep + 1) % 25 == 0:
            sr = np.mean([r["success"] for r in rows])
            dr = np.mean([r["depth_ok"] for r in rows])
            print(f"  {ep+1}/{args.episodes}  success={sr:.3f}  depth_ok={dr:.3f}")
    env.close()

    summarize(rows, cfg, args.phase, args)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2, default=float))
        print(f"[oracle] per-episode rows -> {args.out}")


if __name__ == "__main__":
    main()
