"""
evaluate.py -- fixed-episode evaluation of a trained policy, reported as
aggregate metrics per condition (never single-run numbers), per the PS.

    python evaluate.py --run runs/phase4_tqc --phase 4 --episodes 200
    python evaluate.py --run runs/phase4_tqc --phase 4 --noise-sweep

Generalization (Phase 5, dynamics randomization -- peg mass, friction, joint
damping, restitution, bore clearance): NOT a `--conditions` value, despite an
earlier version of this docstring/README claiming
`--conditions in_dist,unseen_dynamics` -- `build_eval_env`'s condition
argument only ever gates the *noise* wrappers (see `condition.startswith
("noise")` below); there never was a code path reading "unseen_dynamics" as
a string, so that invocation silently produced two IDENTICAL rows. Dynamics
randomization is controlled entirely by WHICH PHASE'S CONFIG IS LOADED
(`PHASES[5].randomize_dynamics=True`, `manipularl/configs.py`) -- get it by
running the SAME checkpoint under `--phase 4` (in-distribution) and
`--phase 5` (unseen dynamics + wider obstacle range) as two separate
invocations and comparing the two result rows directly:

    python evaluate.py --run runs/phase4_tqc --phase 4 --episodes 200 --out results/x_in_dist.csv
    python evaluate.py --run runs/phase4_tqc --phase 5 --episodes 200 --out results/x_unseen_dynamics.csv

Conditions (for `--conditions`/`--noise-sweep`)
  in_dist          the phase's own eval-split env (unseen layouts already,
                   since train/eval episode seeds are disjoint)
  noise=<sigma>     eval-time observation+action noise + peg perturbations
                   (Phase 6); --noise-sweep runs a ladder of magnitudes

Metrics (aggregate over --episodes independently randomized episodes):
  success_rate, collision_rate, mean_steps (completion time),
  path_efficiency (straight-line / travelled EE path), final_pos_error,
  grasp_failure_rate, insert_depth_mean, insert_tilt_deg_mean
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from sb3_contrib import TQC
from stable_baselines3 import PPO, SAC

from manipularl_ur5.env import ManipulaRLEnv
from manipularl.wrappers import NoisyObservation, NoisyAction, PegPerturbation
from manipularl_ur5.make_env import make_vec_env
from manipularl_ur5.configs import get_phase, EVAL_SEED_LO


def load_policy(run_dir: str, algo: str):
    run = Path(run_dir)
    model_path = run / "model.zip"
    if not model_path.exists():
        model_path = run / "best" / "best_model.zip"
    Model = {"tqc": TQC, "sac": SAC}.get(algo, PPO)
    return Model.load(str(model_path), device="cpu"), str(run / "vecnormalize.pkl")


def build_eval_env(phase, condition, sigma, n_stack, norm_path, seed, cfg_overrides=None,
                    index_seed=None):
    """Single (non-vec) env with the right perturbations, wrapped to match training obs."""
    env = ManipulaRLEnv(phase=phase, split="eval", seed=seed, cfg_overrides=cfg_overrides,
                        index_seed=index_seed)
    if condition.startswith("noise"):
        env = NoisyObservation(env, sigma)
        env = NoisyAction(env, sigma)
        # BUGFIX (found while preparing the Phase-6 noise sweep for the first
        # time): this used to floor the peg-perturbation force at 2.0N even
        # at sigma=0.0, so the sweep's own "baseline" rung was never actually
        # disturbance-free -- it always applied a 2N peg impulse every 25
        # steps (PegPerturbation.period, wrappers.py), undermining the exact
        # "characterize degradation from a clean baseline" measurement the
        # PS's Phase 6 asks for. PegPerturbation itself already treats
        # force<=0.0 as a true no-op (wrappers.py's own `if self.force > 0.0`
        # guard), so removing the floor here is sufficient -- no wrapper
        # change needed.
        env = PegPerturbation(env, force=sigma * 300)
    # reuse the vec wrapper stack for identical normalisation + frame stacking
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

    venv = DummyVecEnv([lambda: env])
    if n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack)
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv, env


def rollout(model, venv, base_env, episodes, phase_cfg, start_index):
    rows = []
    for ep in range(episodes):
        obs = venv.reset()
        done = False
        ee_path = 0.0
        prev_ee = None
        first_ee = None
        steps = 0
        last_info = {}
        final_err = float("nan")
        goal = None
        while not done:
            # snapshot the terminal-ish state BEFORE stepping, so the last one
            # captured is the true final state (the vec-env auto-resets on done)
            s = base_env.unwrapped._state_dict()
            ee = s["ee_pos"]
            goal = s["goal_pos"]
            final_err = float(np.linalg.norm(ee - goal))
            if first_ee is None:
                first_ee = ee.copy()
            if prev_ee is not None:
                ee_path += float(np.linalg.norm(ee - prev_ee))
            prev_ee = ee.copy()

            action, _ = model.predict(obs, deterministic=True)
            obs, _r, dones, infos = venv.step(action)
            done = bool(dones[0])
            last_info = infos[0]
            steps += 1
        straight = float(np.linalg.norm(goal - first_ee)) if (first_ee is not None and goal is not None) else 0.0
        path_eff = straight / ee_path if ee_path > 1e-6 else 0.0
        ga = last_info.get("grasp_attempts", 0)
        gf = last_info.get("grasp_failures", 0)
        rows.append(dict(
            success=int(bool(last_info.get("success", False))),
            collided=int(bool(last_info.get("episode_collided", False))),
            steps=steps,
            path_efficiency=path_eff,
            final_pos_error=final_err,
            grasp_failure_rate=(gf / ga) if ga else 0.0,
            insert_depth=float(last_info.get("max_depth", 0.0)),
            insert_tilt_deg=float(last_info.get("min_tilt_deg", np.nan)),
        ))
    return rows


def summarize(rows):
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    arr["insert_tilt_deg"][~np.isfinite(arr["insert_tilt_deg"])] = np.nan
    return dict(
        n=len(rows),
        success_rate=float(arr["success"].mean()),
        collision_rate=float(arr["collided"].mean()),
        mean_steps=float(arr["steps"].mean()),
        path_efficiency=float(np.nanmean(arr["path_efficiency"])),
        final_pos_error=float(arr["final_pos_error"].mean()),
        grasp_failure_rate=float(arr["grasp_failure_rate"].mean()),
        insert_depth_mean=float(arr["insert_depth"].mean()),
        insert_tilt_deg_mean=float(np.nanmean(arr["insert_tilt_deg"])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--algo", default="tqc", choices=["tqc", "ppo", "sac"])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--conditions", default="in_dist")
    ap.add_argument("--noise-sweep", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--align-only", action="store_true",
                    help="eval the Task-1 (reach+align) variant: cfg_overrides={align_only: True}")
    ap.add_argument("--insert-only", action="store_true",
                    help="eval the Task-2 (insert) variant: cfg_overrides={insert_only: True} "
                         "-- every episode starts pre-aligned in the insertion zone at depth 0")
    ap.add_argument("--descend-only", action="store_true",
                    help="eval the Task-1b (descend) variant: cfg_overrides={descend_only: True} "
                         "-- every episode starts pre-grasped + xy-aligned at a random height "
                         "in the airspace column above the hole")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="draw an INDEPENDENT held-out episode set for repeated-eval "
                         "variance checks (controls EpisodeSampler._index_rng via the new "
                         "index_seed param). Default (omitted) reproduces the EXACT "
                         "standard eval set every prior logged result used -- the eval "
                         "episode sequence was previously always default_rng(1) "
                         "regardless of any seed passed in; this flag is the first way "
                         "to actually change it")
    ap.add_argument("--cfg-json", default=None,
                    help='generic PhaseConfig cfg_overrides as a JSON dict, e.g. '
                         '\'{"hole_chamfer": true, "grasp_max_force": 15.0}\' -- for the '
                         'Stage-1/E56 fixed-env track. Merged with --align-only/'
                         '--insert-only/--descend-only if both given.')
    args = ap.parse_args()

    cfg_overrides = None
    if args.align_only:
        cfg_overrides = {"align_only": True}
    elif args.insert_only:
        cfg_overrides = {"insert_only": True}
    elif args.descend_only:
        cfg_overrides = {"descend_only": True}
    if args.cfg_json:
        import json
        extra = json.loads(args.cfg_json)
        cfg_overrides = {**(cfg_overrides or {}), **extra}

    model, norm_path = load_policy(args.run, args.algo)
    cfg = get_phase(args.phase)

    conditions = args.conditions.split(",")
    if args.noise_sweep:
        conditions = [f"noise={s:.3f}" for s in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)]

    out_path = Path(args.out or (Path(args.run) / "evaluation.csv"))
    results = []
    for cond in conditions:
        sigma = float(cond.split("=")[1]) if cond.startswith("noise=") else 0.0
        venv, base = build_eval_env(args.phase, cond, sigma, args.frame_stack, norm_path,
                                    seed=EVAL_SEED_LO, cfg_overrides=cfg_overrides,
                                    index_seed=args.eval_seed)
        rows = rollout(model, venv, base, args.episodes, cfg, EVAL_SEED_LO)
        summ = summarize(rows)
        summ["condition"] = cond
        summ["phase"] = args.phase
        results.append(summ)
        print(f"[{cond:>14}] success={summ['success_rate']:.3f} "
              f"collision={summ['collision_rate']:.3f} steps={summ['mean_steps']:.0f} "
              f"path_eff={summ['path_efficiency']:.2f} "
              f"depth={summ['insert_depth_mean']:.3f} tilt={summ['insert_tilt_deg_mean']:.1f}")
        venv.close()

    fields = ["phase", "condition", "n", "success_rate", "collision_rate", "mean_steps",
              "path_efficiency", "final_pos_error", "grasp_failure_rate",
              "insert_depth_mean", "insert_tilt_deg_mean"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
