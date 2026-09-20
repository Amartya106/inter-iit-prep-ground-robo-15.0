"""
train.py -- train one phase of ManipulaRL with TQC (primary) or PPO (baseline).

    python train.py --config configs/phase1.yaml
    python train.py --config configs/phase2.yaml --warm-start runs/phase1/model.zip
    python train.py --config configs/phase4.yaml --timesteps 3_000_000 --n-envs 8

Cross-phase transfer: --warm-start loads the policy/critic weights from an
earlier phase (the observation/action interface is identical across phases,
so the load is exact). The replay buffer is NOT carried over.

Everything phase-specific lives in the YAML; this script is the same for all
six phases and for both algorithms.
"""

import argparse
import os
from pathlib import Path

# CPU throughput: the killer here is thread oversubscription -- N SubprocVecEnv
# workers each spinning up BLAS threads while the learner does torch gradient
# steps. Pin BLAS/OMP to 1 (pybullet stepping is single-threaded anyway) and
# give torch a fixed intra-op pool. This alone took TQC from ~80 to ~500 fps.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
_NT = int(os.environ.get("MANIPULARL_TORCH_THREADS", "8"))
import torch

torch.set_num_threads(_NT)

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_contrib import TQC

from manipularl_ur5.make_env import make_vec_env
from manipularl_ur5.configs import get_phase
from manipularl.callbacks import (
    ManipulaRLTensorboardCallback, GraspCurriculumAnneal, ObstacleCountCurriculum,
    AlignCurriculumAnneal, GraspTriggerCurriculum, InsertDepthCurriculum,
    InsertStartDepthAnneal, XyErrWeightAnneal, GraspDropPenaltyAnneal, AirspaceHeightAnneal,
    CollisionCurriculum, DreamAugmentCallback, SuccessGatedAnneal,
)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _soften_warm_start(model, log_std: float = -0.7):
    """Keep the warm-started *policy* net, but reset the *value* net + optimiser
    and raise the action noise. Rationale: when the reward changes between
    phases (e.g. phase 3 adds obstacles + collision_is_failure), the inherited
    critic is systematically wrong and a confident low-noise policy resists the
    new detours it needs to learn. Reinit-value + more entropy lets it adapt
    without throwing away the manipulation skill."""
    import torch
    import torch.nn as nn

    pol = model.policy
    value_mods = [m for name in ("value_net",)
                  for m in [getattr(pol.mlp_extractor, name, None), getattr(pol, name, None)]
                  if m is not None]
    n = 0
    for mod in value_mods:
        for m in mod.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
                n += 1
    with torch.no_grad():
        pol.log_std.data.fill_(float(log_std))
    pol.optimizer = pol.optimizer_class(pol.parameters(), lr=model.lr_schedule(1.0),
                                        **pol.optimizer_kwargs)
    print(f"[train] SOFT warm-start: reinit {n} value-net layers, "
          f"log_std -> {log_std}, optimiser reset")


def build_model(algo: str, venv, cfg: dict, seed: int, tb_dir: str):
    hp = dict(cfg.get("hyperparams", {}))
    policy_kwargs = hp.pop("policy_kwargs", {})
    if algo == "tqc":
        return TQC("MlpPolicy", venv, seed=seed, verbose=1, tensorboard_log=tb_dir,
                   policy_kwargs=policy_kwargs, **hp)
    if algo == "ppo":
        return PPO("MlpPolicy", venv, seed=seed, verbose=1, tensorboard_log=tb_dir,
                   policy_kwargs=policy_kwargs, **hp)
    if algo == "sac":
        return SAC("MlpPolicy", venv, seed=seed, verbose=1, tensorboard_log=tb_dir,
                   policy_kwargs=policy_kwargs, **hp)
    raise ValueError(f"unknown algo {algo!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--warm-start", default=None, help="checkpoint .zip from an earlier phase")
    ap.add_argument("--warm-start-norm", default=None,
                    help="vecnormalize.pkl from the warm-start phase (keeps obs stats continuous)")
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    ap.add_argument("--n-envs", type=int, default=None, help="override n_envs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--no-subproc", action="store_true")
    ap.add_argument("--progress", action="store_true",
                    help="force the rich progress bar (default: only when stdout is a TTY)")
    ap.add_argument("--soft-warm-start", action="store_true",
                    help="PPO: keep the warm-started policy net but re-initialise the value "
                         "net + optimiser and raise the action log-std, so a confident prior "
                         "policy can still adapt to a changed reward (e.g. obstacles).")
    ap.add_argument("--soft-log-std", type=float, default=-0.7,
                    help="log-std to reset the action noise to under --soft-warm-start")
    ap.add_argument("--reset-obstacle-obs-norm", action="store_true",
                    help="UR5 Phase 3 port fix: when warm-starting into a phase with "
                         "obstacles from a phase that had none (e.g. Phase 2, "
                         "num_obstacles=0), the source VecNormalize's obstacle-feature "
                         "dims have ~zero recorded variance -- real obstacle features "
                         "then get divided by sqrt(~0) and saturate at clip_obs's +/-10 "
                         "rails from step 0, on top of first-layer weights that were "
                         "never gradient-shaped for a nonzero input. Resets those dims' "
                         "running mean/var to 0/1 (a sane starting scale) right after "
                         "the warm-started VecNormalize loads, leaving every other "
                         "obs dim's carried-over statistics untouched.")
    ap.add_argument("--reset-depth-obs-norm", action="store_true",
                    help="Same mechanism as --reset-obstacle-obs-norm, for the single "
                         "peg_depth obs dim (_assemble_obs index 56 per frame): it is "
                         "unconditionally 0.0 whenever cfg.use_hole is False (Phases "
                         "1-3), so a Phase-3-trained VecNormalize records ~zero variance "
                         "there too, and Phase 4 (use_hole=True, where peg_depth is the "
                         "core insertion-progress signal) would see it arrive railed.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    phase = int(cfg["phase"])
    algo = cfg.get("algo", "tqc").lower()
    n_envs = args.n_envs or int(cfg.get("n_envs", 6))
    total_timesteps = args.timesteps or int(cfg["total_timesteps"])
    frame_stack = int(cfg.get("frame_stack", 3))
    grasp_curr = cfg.get("grasp_curriculum", None)
    cfg_overrides = cfg.get("cfg_overrides", None)
    insert_xy_jitter = cfg.get("insert_xy_jitter", None)
    insert_tilt_jitter_deg = cfg.get("insert_tilt_jitter_deg", None)
    linger_pen_w = cfg.get("linger_pen_w", None)
    settle_bonus_w = cfg.get("settle_bonus_w", None)
    arm_jitter_pen_w = cfg.get("arm_jitter_pen_w", None)
    replay_path = cfg.get("replay_path", None)
    replay_curriculum = cfg.get("replay_curriculum", None)
    keypoint_w = cfg.get("keypoint_w", None)

    out_dir = Path(args.out_dir or f"runs/phase{phase}_{algo}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = str(out_dir / "tb")

    print(f"[train] phase={phase} algo={algo} n_envs={n_envs} steps={total_timesteps:,} "
          f"frame_stack={frame_stack} -> {out_dir}")

    venv = make_vec_env(phase, n_envs=n_envs, split="train", seed=args.seed,
                        training=True, n_stack=frame_stack,
                        norm_path=args.warm_start_norm,
                        subproc=not args.no_subproc, grasp_curriculum=grasp_curr,
                        cfg_overrides=cfg_overrides, insert_xy_jitter=insert_xy_jitter,
                        insert_tilt_jitter_deg=insert_tilt_jitter_deg,
                        linger_pen_w=linger_pen_w, settle_bonus_w=settle_bonus_w,
                        arm_jitter_pen_w=arm_jitter_pen_w,
                        replay_path=replay_path, replay_curriculum=replay_curriculum,
                        keypoint_w=keypoint_w)

    if args.reset_obstacle_obs_norm:
        if not args.warm_start_norm:
            raise SystemExit("--reset-obstacle-obs-norm requires --warm-start-norm "
                             "(nothing to reset otherwise)")
        from manipularl_ur5.configs import MAX_OBSTACLES_IN_OBS
        obs_rms = venv.obs_rms
        total_dim = obs_rms.mean.shape[0]
        d = total_dim // frame_stack
        assert d * frame_stack == total_dim, (
            f"obs dim {total_dim} not divisible by frame_stack {frame_stack}")
        obstacle_dim = MAX_OBSTACLES_IN_OBS * 9   # see env.py's _obstacle_features/_assemble_obs
        stage_dim = 4                              # stage_onehot, the final block in _assemble_obs
        lo_in_frame = d - stage_dim - obstacle_dim
        hi_in_frame = d - stage_dim
        n_reset = 0
        for k in range(frame_stack):
            lo, hi = k * d + lo_in_frame, k * d + hi_in_frame
            obs_rms.mean[lo:hi] = 0.0
            obs_rms.var[lo:hi] = 1.0
            n_reset += hi - lo
        print(f"[train] --reset-obstacle-obs-norm: reset {n_reset} obs dims "
              f"({obstacle_dim} per frame x {frame_stack} frames) to mean=0/var=1")

    if args.reset_depth_obs_norm:
        if not args.warm_start_norm:
            raise SystemExit("--reset-depth-obs-norm requires --warm-start-norm "
                             "(nothing to reset otherwise)")
        obs_rms = venv.obs_rms
        total_dim = obs_rms.mean.shape[0]
        d = total_dim // frame_stack
        assert d * frame_stack == total_dim
        depth_idx_in_frame = 56   # verified via a phase3-vs-phase4 dry-run obs diff
        for k in range(frame_stack):
            idx = k * d + depth_idx_in_frame
            obs_rms.mean[idx] = 0.0
            obs_rms.var[idx] = 1.0
        print(f"[train] --reset-depth-obs-norm: reset peg_depth dim (index "
              f"{depth_idx_in_frame} per frame x {frame_stack} frames) to mean=0/var=1")

    eval_venv = make_vec_env(phase, n_envs=1, split="eval", seed=args.seed + 10_000,
                             training=False, n_stack=frame_stack, subproc=False,
                             cfg_overrides=cfg_overrides, insert_xy_jitter=insert_xy_jitter,
                             insert_tilt_jitter_deg=insert_tilt_jitter_deg,
                             linger_pen_w=linger_pen_w, settle_bonus_w=settle_bonus_w,
                             arm_jitter_pen_w=arm_jitter_pen_w,
                             replay_path=replay_path, replay_curriculum=replay_curriculum,
                             keypoint_w=keypoint_w)

    if args.warm_start:
        print(f"[train] warm-starting from {args.warm_start}")
        Model = {"tqc": TQC, "sac": SAC}.get(algo, PPO)
        model = Model.load(args.warm_start, env=venv, tensorboard_log=tb_dir)
        model.seed = args.seed
        # A warm-started model's hyperparameters come from the LOADED zip,
        # not this config's `hyperparams` block (build_model(), which reads
        # that block, is only called in the fresh-start branch below) -- so
        # `target_kl` has to be set directly on the loaded object to have
        # any effect. E57's writeup (EXPERIMENTS.md) flagged exactly this
        # gap: its retrain (several simultaneous reward/env changes, warm-
        # started, no target_kl) showed `approx_kl=0.81` late in training,
        # vs PPO's typical ~0.01-0.05 -- an unrelated optimization-stability
        # issue plausibly confounding that result, and recommended "a lower
        # learning rate or an explicit target_kl" as the next lever. A new
        # reward term (like keypoint_w) changes the same reward landscape,
        # so applying that recommendation here is cheap, safe (only kicks in
        # if the config sets it), and protects a multi-hour run from the
        # same failure mode instead of discovering it again after the fact.
        if algo == "ppo" and cfg.get("target_kl") is not None:
            model.target_kl = float(cfg["target_kl"])
            print(f"[train] warm-start target_kl override -> {model.target_kl}")
        if args.soft_warm_start:
            if algo != "ppo":
                raise SystemExit("--soft-warm-start is PPO-only")
            _soften_warm_start(model, log_std=args.soft_log_std)
    else:
        model = build_model(algo, venv, cfg, args.seed, tb_dir)

    callbacks = [
        CheckpointCallback(save_freq=max(args.eval_freq, total_timesteps // 10) // n_envs,
                           save_path=str(out_dir / "checkpoints"),
                           name_prefix="model", save_vecnormalize=True),
        EvalCallback(eval_venv, best_model_save_path=str(out_dir / "best"),
                     log_path=str(out_dir / "eval"), eval_freq=max(args.eval_freq // n_envs, 1),
                     n_eval_episodes=20, deterministic=True, render=False),
        ManipulaRLTensorboardCallback(),
    ]

    if cfg.get("anneal_grasp_curriculum") and grasp_curr:
        af = float(cfg.get("anneal_frac", 0.6))
        callbacks.append(GraspCurriculumAnneal(
            start=float(grasp_curr), end=0.0, anneal_frac=af,
            total_timesteps=total_timesteps))
        print(f"[train] grasp-curriculum anneal {grasp_curr} -> 0.0 over {af:.0%} of training")

    if cfg.get("obstacle_curriculum"):
        oaf = float(cfg.get("obstacle_curriculum_frac", 0.6))
        ostart = tuple(cfg.get("obstacle_curriculum_start", [0, 1]))
        _pc = get_phase(phase)
        oend = tuple(cfg.get("obstacle_curriculum_end", [_pc.obstacle_min, _pc.obstacle_max]))
        callbacks.append(ObstacleCountCurriculum(
            start=ostart, end=oend, anneal_frac=oaf, total_timesteps=total_timesteps))
        print(f"[train] obstacle-count curriculum {ostart} -> {oend} over {oaf:.0%} of training")

    align_curr = cfg.get("align_curriculum", None)
    if cfg.get("anneal_align_curriculum") and align_curr:
        aaf = float(cfg.get("align_anneal_frac", 0.5))
        callbacks.append(AlignCurriculumAnneal(
            start=float(align_curr), end=0.0, anneal_frac=aaf,
            total_timesteps=total_timesteps))
        print(f"[train] align-curriculum anneal {align_curr} -> 0.0 over {aaf:.0%} of training")

    insert_curr = cfg.get("insert_curriculum", None)
    if cfg.get("anneal_insert_curriculum") and insert_curr:
        iaf = float(cfg.get("insert_anneal_frac", 0.5))
        callbacks.append(InsertDepthCurriculum(
            start=float(insert_curr), end=0.0, anneal_frac=iaf,
            total_timesteps=total_timesteps))
        print(f"[train] insert-curriculum anneal {insert_curr} -> 0.0 over {iaf:.0%} of training")

    if cfg.get("insert_start_depth_curriculum"):
        isdf = float(cfg.get("insert_start_depth_frac", 0.5))
        callbacks.append(InsertStartDepthAnneal(
            start=1.0, end=0.0, anneal_frac=isdf, total_timesteps=total_timesteps))
        print(f"[train] insert-start-depth anneal 1.0 -> 0.0 over {isdf:.0%} of training "
              f"(Task 2 insert-only)")

    if cfg.get("airspace_height_curriculum"):
        ahf = float(cfg.get("airspace_height_anneal_frac", 0.5))
        callbacks.append(AirspaceHeightAnneal(
            start=0.0, end=1.0, anneal_frac=ahf, total_timesteps=total_timesteps))
        print(f"[train] airspace-height anneal 0.0 -> 1.0 over {ahf:.0%} of training "
              f"(Task 1b descend-only)")

    if cfg.get("dream_model_path"):
        dhorizon = int(cfg.get("dream_horizon", 8))
        dfrac = float(cfg.get("dream_env_frac", 0.25))
        dbranches = int(cfg.get("dream_n_branches", 1))
        callbacks.append(DreamAugmentCallback(
            model_path=cfg["dream_model_path"], horizon=dhorizon, dream_env_frac=dfrac,
            n_branches=dbranches))
        print(f"[train] Dyna-style dream augmentation: model={cfg['dream_model_path']} "
              f"horizon={dhorizon} dream_env_frac={dfrac:.0%} n_branches={dbranches}")

    if cfg.get("xy_err_weight_curriculum"):
        xstart = float(cfg.get("xy_err_weight_start", 1.5))
        xend = float(cfg.get("xy_err_weight_end", 1.0))
        xfrac = float(cfg.get("xy_err_weight_anneal_frac", 0.6))
        callbacks.append(XyErrWeightAnneal(
            start=xstart, end=xend, anneal_frac=xfrac, total_timesteps=total_timesteps))
        print(f"[train] xy_err_weight anneal {xstart} -> {xend} over {xfrac:.0%} of training")

    if cfg.get("success_gated_curriculum"):
        sg = cfg["success_gated_curriculum"]
        callbacks.append(SuccessGatedAnneal(
            setter_method=sg["setter_method"], start=float(sg["start"]), end=float(sg["end"]),
            step_size=float(sg["step_size"]),
            advance_thresh=float(sg.get("advance_thresh", 0.5)),
            regress_thresh=float(sg.get("regress_thresh", 0.2)),
            window=int(sg.get("window", 100)), check_every=int(sg.get("check_every", 1))))
        print(f"[train] success-gated curriculum: {sg['setter_method']} "
              f"{sg['start']} -> {sg['end']} (step {sg['step_size']}, "
              f"advance>={sg.get('advance_thresh', 0.5)}, regress<{sg.get('regress_thresh', 0.2)})")

    if cfg.get("grasp_drop_pen_curriculum"):
        gstart = float(cfg.get("grasp_drop_pen_start", 0.3))
        gend = float(cfg.get("grasp_drop_pen_end", 2.0))
        gfrac = float(cfg.get("grasp_drop_pen_anneal_frac", 0.5))
        callbacks.append(GraspDropPenaltyAnneal(
            start=gstart, end=gend, anneal_frac=gfrac, total_timesteps=total_timesteps))
        print(f"[train] grasp_drop_pen anneal {gstart} -> {gend} over {gfrac:.0%} of training")

    if cfg.get("collision_curriculum"):
        cpstart = float(cfg.get("collision_pen_mult_start", 0.3))
        cpend = float(cfg.get("collision_pen_mult_end", 1.0))
        cpfrac = float(cfg.get("collision_pen_anneal_frac", 0.3))
        ctfrac = float(cfg.get("collision_terminal_at_frac", 0.6))
        callbacks.append(CollisionCurriculum(
            pen_mult_start=cpstart, pen_mult_end=cpend, pen_anneal_frac=cpfrac,
            terminal_at_frac=ctfrac, total_timesteps=total_timesteps))
        print(f"[train] collision curriculum: pen_mult {cpstart}->{cpend} over {cpfrac:.0%}, "
              f"terminal ON at {ctfrac:.0%} of training")

    if cfg.get("grasp_trigger_curriculum"):
        tstart = float(cfg.get("grasp_trigger_start_cm", 15.0)) / 100.0
        tend = float(cfg.get("grasp_trigger_end_cm", 9.0)) / 100.0
        tfrac = float(cfg.get("grasp_trigger_frac", 0.5))
        callbacks.append(GraspTriggerCurriculum(
            start=tstart, end=tend, anneal_frac=tfrac, total_timesteps=total_timesteps))
        print(f"[train] grasp-trigger curriculum {tstart*100:.0f}cm -> {tend*100:.0f}cm "
              f"over {tfrac:.0%} of training")

    import sys

    show_bar = args.progress or sys.stdout.isatty()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=show_bar)

    model.save(str(out_dir / "model.zip"))
    venv.save(str(out_dir / "vecnormalize.pkl"))
    print(f"[train] saved {out_dir/'model.zip'} and {out_dir/'vecnormalize.pkl'}")
    venv.close(); eval_venv.close()


if __name__ == "__main__":
    main()
