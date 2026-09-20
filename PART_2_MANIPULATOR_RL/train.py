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

import numpy as np
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_contrib import TQC

from stable_baselines3.common.callbacks import BaseCallback

from manipularl.make_env import make_vec_env
from manipularl.configs import get_phase
from manipularl.callbacks import (
    ManipulaRLTensorboardCallback, GraspCurriculumAnneal, ObstacleCountCurriculum,
    AlignCurriculumAnneal, GraspTriggerCurriculum, InsertDepthCurriculum,
    InsertStartDepthAnneal, XyErrWeightAnneal, GraspDropPenaltyAnneal, AirspaceHeightAnneal,
    CollisionCurriculum, DreamAugmentCallback, SuccessGatedAnneal,
)


class PerDimStdLoggerCallback(BaseCallback):
    """Phase 4 H1 diagnosis (plan: parsed-plotting-allen.md) -- SB3's own
    tensorboard output only ever logs the MEAN of exp(log_std) across all
    action dimensions ("std" in the console table), which is exactly what let
    the wrist-specific blow-up (j4/j5/j6 reaching 2-3 orders of magnitude
    above the arm-motion dims) go unremarked through 69 logged experiments --
    a mean of ~11 looks merely "elevated," not "three dimensions are at 30+
    while three others are below 1." Logs every dimension separately so this
    can never hide inside an averaged number again. Deliberately kept local
    to train.py (a port-owned/no-shared-invariant file) rather than added to
    manipularl/callbacks.py's ManipulaRLTensorboardCallback, which must stay
    byte-identical to what the UR5 port depends on."""

    def __init__(self, log_every: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every < self.training_env.num_envs:
            log_std = self.model.policy.log_std.data.detach().cpu().numpy()
            stds = np.exp(log_std)
            for i, s in enumerate(stds):
                self.logger.record(f"std_per_dim/j{i}" if i < len(stds) - 1 else "std_per_dim/grip",
                                   float(s))
        return True


class LogStdBandClamp(BaseCallback):
    """Combined-fix plan (parsed-plotting-allen.md) -- tonight's two
    ent_coef attempts failed in OPPOSITE directions: 0.0 let the gripper's
    mean action collapse with nothing to pull it back (no entropy pressure
    at all once a bad early gradient pushed it into a dead zone); 0.003
    apparently damped exploration just enough to prevent the policy from
    ever committing to grasping, replicated across two seeds. A single
    unconstrained scalar controls both "does exploration collapse" and
    "does it blow up," and no value of it guarantees neither. This clamps
    policy.log_std into a fixed band EVERY step, independent of ent_coef --
    mechanically incapable of reproducing either failure. Band defaults
    (std in [0.3, 2.0]) grounded in tonight's own measurements: floor above
    the ent_coef=0.0 run's collapsed values, ceiling below every
    pathological wrist reading measured tonight (7.7-245) and close to
    Phase 3's own proven-successful range (mean 1.68, max ~2.4 non-wrist).
    """

    def __init__(self, lo: float = 0.3, hi: float = 2.0,
                 grip_lo: "float | None" = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_lo = float(np.log(lo))
        self.log_hi = float(np.log(hi))
        # Plan v2 (parsed-plotting-allen.md) -- an optional HIGHER floor for
        # just the gripper dimension (index -1 / ACTION_DIM-1): grasping is
        # a single binary-ish threshold crossing (env.py's _handle_grasp,
        # grip_a > _GRASP_THRESH), which a few noisy early advantage
        # estimates (see reinit_value_net's docstring) can extinguish
        # permanently -- unlike the continuous arm dimensions, which keep
        # getting exercised regardless of noise level. A dedicated, higher
        # floor on just this one dimension is a second, independent
        # safeguard against that specific collapse mode, on top of
        # reinit_value_net addressing the root cause.
        self.log_grip_lo = float(np.log(grip_lo)) if grip_lo is not None else self.log_lo

    def _on_step(self) -> bool:
        with torch.no_grad():
            ls = self.model.policy.log_std.data
            ls[:-1].clamp_(self.log_lo, self.log_hi)
            ls[-1].clamp_(self.log_grip_lo, self.log_hi)
        return True


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _reinit_value_net_layers(pol) -> int:
    """Orthogonal-reinit every Linear layer in the policy's value trunk +
    head (mlp_extractor.value_net and the final value_net head). Shared by
    _soften_warm_start and reinit_value_net below -- same reinit logic,
    different callers decide what else (log_std, optimiser) to touch
    alongside it."""
    import torch.nn as nn

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
    return n


def _soften_warm_start(model, log_std: float = -0.7):
    """Keep the warm-started *policy* net, but reset the *value* net + optimiser
    and raise the action noise. Rationale: when the reward changes between
    phases (e.g. phase 3 adds obstacles + collision_is_failure), the inherited
    critic is systematically wrong and a confident low-noise policy resists the
    new detours it needs to learn. Reinit-value + more entropy lets it adapt
    without throwing away the manipulation skill."""
    pol = model.policy
    n = _reinit_value_net_layers(pol)
    with torch.no_grad():
        pol.log_std.data.fill_(float(log_std))
    pol.optimizer = pol.optimizer_class(pol.parameters(), lr=model.lr_schedule(1.0),
                                        **pol.optimizer_kwargs)
    print(f"[train] SOFT warm-start: reinit {n} value-net layers, "
          f"log_std -> {log_std}, optimiser reset")


def reinit_value_net(model) -> int:
    """Plan v2 (parsed-plotting-allen.md) -- the value-net-reinit HALF of
    _soften_warm_start, decoupled from its own log_std reset, so it composes
    cleanly with --warm-start-pad + LogStdBandClamp instead of conflicting
    with them. Directly targets the measured mechanism behind E74/E75's
    total behavioural freeze (grasp_failure_rate stuck at ~1.0 for the
    ENTIRE training run, on both the full monolithic task and the isolated
    Align sub-task): warm_start_pad() deliberately copies the value net
    verbatim from the source checkpoint (Phase 3), but Phase 3's reward
    doesn't have Phase 4's terms (xy_err/tilt/depth potential, ALIGN_BONUS,
    DEPTH_PROGRESS_BONUS, ring repulsion) -- so that copied V(s) is
    initially WRONG for this task. Measured directly: E74's
    explained_variance started at -0.429 (worse than predicting the mean)
    and only recovered to 0.969 by ~53% through training. PPO's advantage
    estimates are GAE(rewards, V(s)); a badly-wrong V(s) during the first
    several updates contaminates the POLICY gradient too, even though the
    policy net itself started in a genuinely good place (verified function-
    identical to the source checkpoint, torch.allclose, exact 0.0
    difference, at padding time). This uniquely threatens the grasp
    decision -- a single binary-ish threshold crossing (env.py's
    _handle_grasp, grip_a > _GRASP_THRESH) that a few contaminated early
    updates can extinguish permanently, unlike the continuous arm
    dimensions which keep getting exercised regardless of noise level.
    insert_only (E73, the one real result of the whole investigation) never
    showed this collapse because it starts pre-grasped -- the one
    vulnerable decision is removed from the task by construction, which is
    the natural experiment that motivates this fix rather than a guess."""
    pol = model.policy
    n = _reinit_value_net_layers(pol)
    pol.optimizer = pol.optimizer_class(pol.parameters(), lr=model.lr_schedule(1.0),
                                        **pol.optimizer_kwargs)
    print(f"[train] reinit-value-net: reinit {n} value-net layers, optimiser reset "
          f"(log_std untouched -- left for --log-std-min/--log-std-max to manage)")
    return n


def warm_start_pad(old_path: str, model, frame_stack: int) -> int:
    """Combined-fix plan (parsed-plotting-allen.md) -- H3 widened the
    observation from 123 to 131 dims/frame (8 new dims APPENDED at the end
    of each frame's slice -- see manipularl/env.py's _assemble_obs), so no
    checkpoint saved before that change has a first-layer shape matching
    this env's observation space, and a normal --warm-start (Model.load)
    would reject it outright. This is a function-preserving ("net2net"-
    style) surgery instead: `model` must already be a freshly-built model
    (via build_model()) at the NEW width; its weights get overwritten in
    place from `old_path`'s checkpoint everywhere the shapes still agree --
    action_net/value_net heads, log_std, and every deeper trunk layer copy
    verbatim (nothing about them depends on observation width). Only
    policy_net[0]/value_net[0] (the sole shape-changed layers) get their
    OLD columns copied into the matching [0:old_d] sub-slice of each of the
    `frame_stack` new, wider frames; the new (H3) columns are left at
    exactly zero. Zero-init means the padded model's output is IDENTICAL
    to the old model's, for any input, at step 0 -- the new dims contribute
    nothing to any pre-activation until gradients actually move them.
    Verified directly (torch.allclose) on a dummy old/new pair before this
    was ever used for a real run.
    Returns the number of new (zero-init) dims per frame, for logging."""
    import torch

    Model = PPO  # this surgery is PPO-specific (policy_net/value_net/log_std
                 # layout below); raise loudly rather than silently
                 # mis-copying if ever pointed at a non-PPO checkpoint.
    old = Model.load(old_path, device="cpu", print_system_info=False)
    old_pol, new_pol = old.policy, model.policy

    old_total = old_pol.mlp_extractor.policy_net[0].weight.data.shape[1]
    new_total = new_pol.mlp_extractor.policy_net[0].weight.data.shape[1]
    assert old_total % frame_stack == 0 and new_total % frame_stack == 0, (
        f"obs width must divide evenly by frame_stack={frame_stack} "
        f"(old={old_total}, new={new_total})")
    old_d = old_total // frame_stack
    new_d = new_total // frame_stack
    assert new_d >= old_d, f"padded per-frame width {new_d} must be >= old width {old_d}"
    n_new = new_d - old_d

    def _pad_first_layer(old_layer, new_layer):
        with torch.no_grad():
            new_layer.bias.data.copy_(old_layer.bias.data)
            new_layer.weight.data.zero_()
            for k in range(frame_stack):
                old_lo, old_hi = k * old_d, (k + 1) * old_d
                new_lo = k * new_d
                new_layer.weight.data[:, new_lo:new_lo + old_d] = \
                    old_layer.weight.data[:, old_lo:old_hi]

    _pad_first_layer(old_pol.mlp_extractor.policy_net[0], new_pol.mlp_extractor.policy_net[0])
    _pad_first_layer(old_pol.mlp_extractor.value_net[0], new_pol.mlp_extractor.value_net[0])

    with torch.no_grad():
        for name in ("policy_net", "value_net"):
            old_seq = getattr(old_pol.mlp_extractor, name)
            new_seq = getattr(new_pol.mlp_extractor, name)
            assert len(old_seq) == len(new_seq), (
                f"{name}: old/new net_arch must match (surgery only handles the "
                f"first layer's shape changing) -- got {len(old_seq)} vs {len(new_seq)} modules")
            for i in range(2, len(old_seq)):   # [0] handled above; skip Tanh (no .weight)
                if hasattr(old_seq[i], "weight"):
                    new_seq[i].weight.data.copy_(old_seq[i].weight.data)
                    new_seq[i].bias.data.copy_(old_seq[i].bias.data)
        new_pol.action_net.weight.data.copy_(old_pol.action_net.weight.data)
        new_pol.action_net.bias.data.copy_(old_pol.action_net.bias.data)
        new_pol.value_net.weight.data.copy_(old_pol.value_net.weight.data)
        new_pol.value_net.bias.data.copy_(old_pol.value_net.bias.data)
        new_pol.log_std.data.copy_(old_pol.log_std.data)

    print(f"[train] warm-start-pad: {old_path} ({old_d} dims/frame) -> this env "
          f"({new_d} dims/frame) -- {n_new} new (zero-init) dims x {frame_stack} frames "
          f"= {n_new * frame_stack} new columns; action_net/value_net/log_std/deeper "
          f"trunk layers copied verbatim")
    return n_new


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
    # Phase 4 H1 diagnosis (plan: parsed-plotting-allen.md) -- the warm-start
    # branch above loads log_std verbatim from the source checkpoint and
    # never resets it (only model.seed and, since E67, model.target_kl are
    # touched). Measured directly: it ratchets up every phase transition
    # (phase2 ~1.0 -> phase3_best ~3.3, wrist dims 7.7/8.2 -> phase4_full_dream
    # wrist dims 33.4/32.7 -> phase4_stage3_fixedenv wrist dims 134/119),
    # because entropy's gradient w.r.t. log_std is a constant +1 per
    # dimension with nothing in phases 1-3's reward ever opposing it on the
    # wrist (no term references peg orientation before phase 4). These two
    # flags make that inherited state resettable, independent of
    # --soft-warm-start (which also reinits the value net + optimiser and
    # so can't isolate this one variable).
    ap.add_argument("--reset-log-std", type=float, default=None,
                    help="PPO warm-start only: overwrite EVERY action dimension's "
                         "log_std to this value after loading (applied before "
                         "--reset-log-std-dims, so that flag can override a subset "
                         "on top of this blanket reset).")
    ap.add_argument("--reset-log-std-dims", nargs=2, default=None,
                    metavar=("IDX_CSV", "VALUE"),
                    help="PPO warm-start only: overwrite specific action dimensions' "
                         "log_std, e.g. '--reset-log-std-dims 4,5,6 0.2' targets the "
                         "KUKA wrist joints (0-indexed: j4/j5/j6) to std~=0.2 without "
                         "touching the others. IDX_CSV is comma-separated joint "
                         "indices into the ACTION_DIM=8 vector (7 arm + 1 gripper); "
                         "VALUE is the log_std to set them to.")
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="PPO warm-start only: override the loaded model's ent_coef "
                         "post-load. Config-file hyperparams are silently ignored on "
                         "warm-start (build_model(), which reads them, is only called "
                         "in the from-scratch branch below) -- every Phase 4 run has "
                         "so far inherited ent_coef=0.01 from Phase 2/3 regardless of "
                         "what its own config specified. This is the direct fix, "
                         "mirroring the existing target_kl override below.")
    # Combined-fix plan (parsed-plotting-allen.md) -- H3's observation-width
    # change (123->131 dims/frame) makes every existing checkpoint
    # incompatible with a normal --warm-start. --warm-start-pad performs a
    # function-preserving surgery instead (see warm_start_pad()'s
    # docstring) so a Phase-4 retrain can still start from Phase 3's actual
    # competence rather than from scratch. Mutually exclusive with
    # --warm-start in practice (both build a model from a checkpoint); if
    # both are passed, --warm-start-pad wins.
    ap.add_argument("--warm-start-pad", default=None,
                    help="checkpoint .zip from a phase with a NARROWER observation "
                         "space (e.g. pre-H3 Phase 3) -- builds a fresh model at this "
                         "config's own (wider) width and copies the old weights in via "
                         "warm_start_pad(), zero-initialising only the new dims.")
    ap.add_argument("--log-std-min", type=float, default=None,
                    help="PPO only: clamp policy.log_std's per-dimension std to at "
                         "least this value, every step, via LogStdBandClamp -- pairs "
                         "with --log-std-max. Neither alone enables the callback; both "
                         "must be passed together.")
    ap.add_argument("--log-std-max", type=float, default=None,
                    help="PPO only: clamp policy.log_std's per-dimension std to at "
                         "most this value, every step, via LogStdBandClamp.")
    ap.add_argument("--log-std-min-grip", type=float, default=None,
                    help="Plan v2 (parsed-plotting-allen.md): a HIGHER floor for just "
                         "the gripper dimension (last action dim), on top of "
                         "--log-std-min/--log-std-max. Grasping is a single binary-ish "
                         "threshold crossing that a few noisy early advantage estimates "
                         "can extinguish permanently (see reinit_value_net's docstring "
                         "for the full mechanism) -- a dedicated floor structurally "
                         "prevents that dimension's exploration from ever shrinking "
                         "enough to get stuck there, independent of reinit_value_net "
                         "addressing the root cause. Requires --log-std-min/-max too.")
    ap.add_argument("--reinit-value-net", action="store_true",
                    help="Plan v2 (parsed-plotting-allen.md), --warm-start-pad only: "
                         "after warm_start_pad() copies the value net verbatim from the "
                         "source checkpoint, reinitialise it (orthogonal init + fresh "
                         "optimiser) -- the value-net-only half of the existing "
                         "_soften_warm_start/--soft-warm-start, decoupled from that "
                         "flag's own log_std reset so it composes with "
                         "--log-std-min/--log-std-max instead of conflicting. See "
                         "reinit_value_net()'s docstring for why this is needed: E74's "
                         "explained_variance measured -0.429 early in training (the "
                         "copied critic is wrong for Phase 4's different reward), which "
                         "plausibly explains that run's total grasp-failure freeze.")
    args = ap.parse_args()
    if (args.log_std_min is None) != (args.log_std_max is None):
        raise SystemExit("--log-std-min and --log-std-max must be passed together")
    if args.log_std_min_grip is not None and args.log_std_min is None:
        raise SystemExit("--log-std-min-grip requires --log-std-min/--log-std-max")
    if args.reinit_value_net and not args.warm_start_pad:
        raise SystemExit("--reinit-value-net requires --warm-start-pad")

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
    satisficing_proximity = cfg.get("satisficing_proximity", None)
    risk_seeking_depth = cfg.get("risk_seeking_depth", None)

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
                        keypoint_w=keypoint_w,
                        satisficing_proximity=satisficing_proximity,
                        risk_seeking_depth=risk_seeking_depth)
    eval_venv = make_vec_env(phase, n_envs=1, split="eval", seed=args.seed + 10_000,
                             training=False, n_stack=frame_stack, subproc=False,
                             cfg_overrides=cfg_overrides, insert_xy_jitter=insert_xy_jitter,
                             insert_tilt_jitter_deg=insert_tilt_jitter_deg,
                             linger_pen_w=linger_pen_w, settle_bonus_w=settle_bonus_w,
                             arm_jitter_pen_w=arm_jitter_pen_w,
                             replay_path=replay_path, replay_curriculum=replay_curriculum,
                             keypoint_w=keypoint_w,
                             satisficing_proximity=satisficing_proximity,
                             risk_seeking_depth=risk_seeking_depth)

    if args.warm_start and not args.warm_start_pad:
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
        if args.ent_coef is not None:
            if algo != "ppo":
                raise SystemExit("--ent-coef override is PPO-only (SAC/TQC's ent_coef "
                                 "is usually 'auto' and not a simple post-load overwrite)")
            model.ent_coef = float(args.ent_coef)
            print(f"[train] warm-start ent_coef override -> {model.ent_coef}")
        if args.reset_log_std is not None or args.reset_log_std_dims is not None:
            if algo != "ppo":
                raise SystemExit("--reset-log-std(-dims) is PPO-only "
                                 "(policy.log_std is PPO's DiagGaussianDistribution "
                                 "parameter; SAC/TQC don't expose an equivalent)")
            import torch as _torch
            with _torch.no_grad():
                before = model.policy.log_std.data.exp().cpu().numpy().round(3).tolist()
                if args.reset_log_std is not None:
                    model.policy.log_std.data.fill_(float(args.reset_log_std))
                if args.reset_log_std_dims is not None:
                    idx_csv, val_str = args.reset_log_std_dims
                    idxs = [int(x) for x in idx_csv.split(",")]
                    val = float(val_str)
                    for i in idxs:
                        model.policy.log_std.data[i] = val
                after = model.policy.log_std.data.exp().cpu().numpy().round(3).tolist()
            print(f"[train] log_std reset: std {before} -> {after}")
    elif args.warm_start_pad:
        if algo != "ppo":
            raise SystemExit("--warm-start-pad is PPO-only (see warm_start_pad()'s "
                             "docstring -- it assumes policy_net/value_net/log_std)")
        if args.soft_warm_start:
            raise SystemExit("--soft-warm-start reinitialises the value net; "
                             "--warm-start-pad already copies it from the old "
                             "checkpoint verbatim -- combining them is contradictory")
        print(f"[train] warm-start-pad from {args.warm_start_pad}")
        model = build_model(algo, venv, cfg, args.seed, tb_dir)
        model.seed = args.seed
        warm_start_pad(args.warm_start_pad, model, frame_stack)
        if args.reinit_value_net:
            reinit_value_net(model)
        if cfg.get("target_kl") is not None:
            model.target_kl = float(cfg["target_kl"])
            print(f"[train] warm-start-pad target_kl override -> {model.target_kl}")
        if args.ent_coef is not None:
            model.ent_coef = float(args.ent_coef)
            print(f"[train] warm-start-pad ent_coef override -> {model.ent_coef}")
        if args.reset_log_std is not None or args.reset_log_std_dims is not None:
            import torch as _torch
            with _torch.no_grad():
                before = model.policy.log_std.data.exp().cpu().numpy().round(3).tolist()
                if args.reset_log_std is not None:
                    model.policy.log_std.data.fill_(float(args.reset_log_std))
                if args.reset_log_std_dims is not None:
                    idx_csv, val_str = args.reset_log_std_dims
                    idxs = [int(x) for x in idx_csv.split(",")]
                    val = float(val_str)
                    for i in idxs:
                        model.policy.log_std.data[i] = val
                after = model.policy.log_std.data.exp().cpu().numpy().round(3).tolist()
            print(f"[train] log_std reset: std {before} -> {after}")
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

    if algo == "ppo":
        callbacks.append(PerDimStdLoggerCallback())
    if args.log_std_min is not None:
        if algo != "ppo":
            raise SystemExit("--log-std-min/--log-std-max is PPO-only")
        callbacks.append(LogStdBandClamp(lo=args.log_std_min, hi=args.log_std_max,
                                         grip_lo=args.log_std_min_grip))
        grip_note = (f", grip floor {args.log_std_min_grip}"
                    if args.log_std_min_grip is not None else "")
        print(f"[train] LogStdBandClamp active: std in [{args.log_std_min}, "
              f"{args.log_std_max}]{grip_note}")

    import sys

    show_bar = args.progress or sys.stdout.isatty()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=show_bar)

    model.save(str(out_dir / "model.zip"))
    venv.save(str(out_dir / "vecnormalize.pkl"))
    print(f"[train] saved {out_dir/'model.zip'} and {out_dir/'vecnormalize.pkl'}")
    venv.close(); eval_venv.close()


if __name__ == "__main__":
    main()
