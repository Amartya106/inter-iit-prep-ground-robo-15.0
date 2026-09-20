"""
callbacks.py -- TensorBoard logging of the task-level metrics the PS asks for.

SB3's Monitor already logs episode return/length. This callback pulls the
per-episode `info` fields our env emits (success, collision, grasp failures,
insertion depth/tilt, ...) off the vec-env buffer and writes rolling means to
TensorBoard so training curves show *task* progress, not just return.
"""

from collections import deque

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .world_model import DynamicsModel, predict_next_raw_obs


class DreamAugmentCallback(BaseCallback):
    """Part 2 of the "dreaming"/model-based RL plan (Task 1b/descend_only):
    Dyna-style rollout-buffer augmentation using the learned dynamics model
    (manipularl/world_model.py, trained separately on scripts/
    collect_dynamics_transitions.py's data -- see runs/dynamics_model.pt and
    its reported held-out accuracy, checked BEFORE this callback is ever
    used, per the plan).

    Each rollout, AFTER real on-policy collection finishes and BEFORE the
    policy update runs: for a fraction of envs, pick a real timestep in the
    just-collected buffer as a branch point, then unroll the dynamics model
    for `horizon` steps under the CURRENT policy, overwriting that many
    consecutive buffer slots (same env column) with the synthetic
    trajectory. `rollout_buffer.compute_returns_and_advantage` is called
    AGAIN over the whole (now-mixed) buffer afterward so GAE stays exactly
    consistent with the overwritten entries (not an approximation -- this
    recomputation is exact, just needs the same last_values/dones SB3 itself
    used, cached from self.locals).

    Known simplifications, disclosed: (1) the dynamics model predicts raw,
    un-stacked single-frame observations (see world_model.py's own
    docstring for why), so a synthetic branch's 3-frame stack history is
    initialized by repeating the real branch-point frame 3x rather than
    using its true preceding history -- reasonable since frame-stacking is
    a noise-smoothing aid here, not load-bearing for the Markov property.
    (2) the model's done/collision predictor was flagged as unreliable at
    training time (99.5% accuracy on a 99.4%-not-done dataset is not
    actually informative) -- imagined rollouts still use it to end early,
    but this is a known weak point, not a verified-accurate signal.
    """

    def __init__(self, model_path: str, horizon: int = 8, dream_env_frac: float = 0.25,
                 n_branches: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.horizon = int(horizon)
        self.dream_env_frac = float(dream_env_frac)
        # Branches per selected env per rollout: each branch starts from the
        # SAME real current live state (there's only one "now" per env) but
        # samples the policy stochastically, so multiple branches diverge
        # into different imagined futures from that one anchor -- standard
        # practice for getting more synthetic coverage per real env without
        # touching dream_env_frac/horizon (which carry their own tradeoffs:
        # frac=1.0 has no more envs to give; horizon growth compounds model
        # error). Branches are planted at distinct, spread-out t_start
        # windows (not just piled on the same slots) so they contribute
        # n_branches x as many distinct buffer slots, not repeats of one.
        self.n_branches = int(n_branches)
        # weights_only=False: torch>=2.6 (e.g. .venv_gpu_test's 2.14) defaults
        # this to True, which rejects the plain numpy arrays (obs_mean etc.)
        # saved alongside the model's state_dict in world_model.py's training
        # script. Safe here -- this checkpoint is produced by this same
        # project's own training code, not an external or untrusted source.
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.dyn_model = DynamicsModel(
            obs_dim=ckpt["obs_mean"].shape[0], action_dim=ckpt["act_mean"].shape[0])
        self.dyn_model.load_state_dict(ckpt["state_dict"])
        self.dyn_model.eval()
        self.obs_mean, self.obs_std = ckpt["obs_mean"], ckpt["obs_std"]
        self.act_mean, self.act_std = ckpt["act_mean"], ckpt["act_std"]
        self._n_dream_calls = 0
        self._n_dream_steps = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        n_steps, n_envs = buf.buffer_size, buf.n_envs
        horizon = min(self.horizon, n_steps)
        n_dream = max(1, int(n_envs * self.dream_env_frac))
        dream_envs = np.random.choice(n_envs, size=n_dream, replace=False)

        vec_normalize = self.training_env  # VecNormalize is the outermost wrapper
        device = self.model.device

        for env_idx in dream_envs:
            # Distinct, spread-out t_start windows for this env's n_branches --
            # divide the buffer into n_branches roughly-equal segments and
            # pick one random start within each, so branches land on
            # DIFFERENT buffer slots rather than piling up on/overwriting
            # each other. Falls back to fully independent random starts if
            # the buffer's too short to spread them out cleanly.
            if n_steps <= horizon:
                t_starts = [0] * self.n_branches
            else:
                usable = n_steps - horizon
                if usable >= self.n_branches:
                    seg = usable // self.n_branches
                    t_starts = [int(np.random.randint(b * seg, max(b * seg + 1, (b + 1) * seg)))
                               for b in range(self.n_branches)]
                else:
                    t_starts = [int(np.random.randint(0, usable + 1)) for _ in range(self.n_branches)]

            # Real raw current-frame obs for this env, straight from the live
            # (post-real-rollout) environment -- not reconstructed from the
            # buffer, avoids any normalize/clip round-trip loss. Every branch
            # for this env starts from this SAME real anchor state (there's
            # only one "now" per env) and diverges via the policy's own
            # stochastic sampling -- multiple independent imagined futures
            # from one real point, not multiple real starting points.
            raw_obs_anchor = np.asarray(
                self.training_env.env_method("_assemble_obs", indices=[int(env_idx)])[0],
                dtype=np.float32)

            for t_start in t_starts:
                frame_history = [raw_obs_anchor.copy(), raw_obs_anchor.copy(), raw_obs_anchor.copy()]
                for h in range(horizon):
                    t = t_start + h
                    stacked_raw = np.concatenate(frame_history, axis=0)
                    stacked_norm = vec_normalize.normalize_obs(stacked_raw[None, :])[0]
                    obs_tensor = torch.as_tensor(stacked_norm, dtype=torch.float32,
                                                 device=device).unsqueeze(0)
                    with torch.no_grad():
                        action_t, value_t, log_prob_t = self.model.policy(obs_tensor)
                    action_np = action_t.cpu().numpy()[0]
                    action_clipped = np.clip(action_np, self.model.action_space.low,
                                             self.model.action_space.high)

                    next_raw_obs, reward, done_prob = predict_next_raw_obs(
                        self.dyn_model, frame_history[-1], action_clipped,
                        self.obs_mean, self.obs_std, self.act_mean, self.act_std)

                    buf.observations[t, env_idx] = stacked_norm
                    buf.actions[t, env_idx] = action_np
                    buf.rewards[t, env_idx] = reward
                    buf.episode_starts[t, env_idx] = 0.0
                    buf.values[t, env_idx] = value_t.item()
                    buf.log_probs[t, env_idx] = log_prob_t.item()
                    self._n_dream_steps += 1

                    frame_history = frame_history[1:] + [next_raw_obs.astype(np.float32)]
                    if done_prob > 0.5:
                        break
                self._n_dream_calls += 1

        # Exact re-derivation of advantages/returns over the now-mixed
        # buffer, using the SAME last_values/dones SB3's own collect_rollouts
        # computed right before calling on_rollout_end (cached in self.locals).
        last_values = self.locals.get("values")
        dones = self.locals.get("dones")
        if last_values is not None and dones is not None:
            buf.compute_returns_and_advantage(last_values=last_values, dones=dones)

        self.logger.record("manipularl/dream_calls_cumulative", self._n_dream_calls)
        self.logger.record("manipularl/dream_steps_cumulative", self._n_dream_steps)


class GraspCurriculumAnneal(BaseCallback):
    """Linearly decay the pre-grasped-start fraction start -> end over the
    first `anneal_frac` of training, then hold at `end`. Forces the policy
    off the curriculum crutch and onto cold-start grasping while it still has
    the pre-grasped competence to bootstrap from.
    """

    def __init__(self, start: float, end: float = 0.0, anneal_frac: float = 0.6,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = float(start), float(end)
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        frac = self.start + (self.end - self.start) * prog
        # only push updates on a meaningful change (every ~0.02)
        if self._last_set is None or abs(frac - self._last_set) >= 0.02 or prog >= 1.0:
            self.training_env.env_method("set_grasp_curriculum", frac)
            self._last_set = frac
        self.logger.record("manipularl/grasp_curriculum", frac)
        return True


class ObstacleCountCurriculum(BaseCallback):
    """Linearly ramp the per-episode obstacle-count range UP from a small
    `start` (e.g. 0-1) to the phase's real `end` (e.g. 2-5) over the first
    `anneal_frac` of training, then hold. The Phase-2/3 warm-start has never
    seen an obstacle; this re-consolidates pick-place first and learns
    avoidance against a manageable count at each stage, instead of the full
    clutter field cold. Mirrors GraspCurriculumAnneal but ramping up, not down.
    """

    def __init__(self, start=(0, 1), end=(2, 5), anneal_frac: float = 0.6,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start_lo, self.start_hi = start
        self.end_lo, self.end_hi = end
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        lo = int(round(self.start_lo + (self.end_lo - self.start_lo) * prog))
        hi = int(round(self.start_hi + (self.end_hi - self.start_hi) * prog))
        hi = max(lo, hi)
        key = (lo, hi)
        if self._last_set != key:
            self.training_env.env_method("set_obstacle_range", lo, hi)
            self._last_set = key
        self.logger.record("manipularl/obstacle_range_lo", lo)
        self.logger.record("manipularl/obstacle_range_hi", hi)
        return True


class AlignCurriculumAnneal(BaseCallback):
    """peg_in_hole only: linearly decay the pre-aligned-start fraction
    start -> end over the first `anneal_frac` of training, then hold.
    Mirrors GraspCurriculumAnneal one stage further down the task chain --
    bootstraps descend/insert/hold first, then forces the policy off the
    curriculum crutch and onto the full transit+align+insert task.
    """

    def __init__(self, start: float, end: float = 0.0, anneal_frac: float = 0.5,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = float(start), float(end)
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        frac = self.start + (self.end - self.start) * prog
        if self._last_set is None or abs(frac - self._last_set) >= 0.02 or prog >= 1.0:
            self.training_env.env_method("set_align_curriculum", frac)
            self._last_set = frac
        self.logger.record("manipularl/align_curriculum", frac)
        return True


class InsertDepthCurriculum(BaseCallback):
    """peg_in_hole only: linearly decay the pre-inserted-start fraction
    start -> end over the first `anneal_frac` of training, then hold.
    Mirrors AlignCurriculumAnneal one stage further down the task chain --
    bootstraps holding a peg already past the insertion depth first (the
    one experience four independent Phase-4 attempts found the policy never
    gets on its own, cold or curriculum-assisted at the align stage), then
    forces the policy off the curriculum crutch onto the full task.
    """

    def __init__(self, start: float, end: float = 0.0, anneal_frac: float = 0.5,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = float(start), float(end)
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        frac = self.start + (self.end - self.start) * prog
        if self._last_set is None or abs(frac - self._last_set) >= 0.02 or prog >= 1.0:
            self.training_env.env_method("set_insert_curriculum", frac)
            self._last_set = frac
        self.logger.record("manipularl/insert_curriculum", frac)
        return True


class InsertStartDepthAnneal(BaseCallback):
    """cfg.insert_only (Task 2, "insert") only: linearly decay the starting
    insertion depth given for free at reset, 1.0 (starts already past
    insert_success_depth -- trivial, just hold) -> 0.0 (starts at depth
    exactly 0, peg tip just touching the mouth -- the real task, genuine
    push-through required), over the first `anneal_frac` of training, then
    hold at 0.0. Unlike InsertDepthCurriculum (Track A), this does not
    control WHETHER episodes start pre-aligned -- cfg.insert_only makes that
    unconditional, every episode, both splits -- only how much of the depth
    is already given away. Eval (build_eval_env / make_vec_env split='eval')
    always forces depth_frac=0.0 regardless of this callback's schedule
    (enforced in env.reset() itself), so eval numbers are never flattered.
    """

    def __init__(self, start: float = 1.0, end: float = 0.0, anneal_frac: float = 0.5,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = float(start), float(end)
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        frac = self.start + (self.end - self.start) * prog
        if self._last_set is None or abs(frac - self._last_set) >= 0.02 or prog >= 1.0:
            self.training_env.env_method("set_insert_start_depth_frac", frac)
            self._last_set = frac
        self.logger.record("manipularl/insert_start_depth_frac", frac)
        return True


class AirspaceHeightAnneal(BaseCallback):
    """cfg.descend_only (Task 1b, "descend") only: linearly ramp UP how much
    of the airspace column's height range _start_pre_airspace() samples
    from, 0.0 (easiest -- start height always ~_AIRSPACE_LOW_M, a trivial
    short descent) -> 1.0 (hardest/real -- full [_AIRSPACE_LOW_M,
    _AIRSPACE_HIGH_M] range), over the first `anneal_frac` of training, then
    hold at 1.0. Opposite ramp direction from InsertStartDepthAnneal (which
    decays a free-credit amount DOWN) because this instead ramps a
    DIFFICULTY RANGE up -- same up-ramping shape as GraspDropPenaltyAnneal/
    ObstacleCountCurriculum. Added after the first (no-curriculum, full
    3-15cm range from step 0) attempt showed insert_depth_mean stuck at
    exactly 0.0 through 27%+ of training across all three jitter-multiplier
    variants -- consistent with the established pattern that a
    full-difficulty-from-cold-start task (no curriculum) tends to never get
    off the ground (Track A's diluted insert_curriculum, E16/E17), while an
    anneal that gives early exploration a genuinely easy on-ramp is what
    actually broke through Phase 4's original wall (E33's insert_only,
    itself curriculum'd this same way via InsertStartDepthAnneal). Eval
    always forces frac=1.0 regardless (enforced in env.py's
    _start_pre_airspace itself), so eval numbers reflect the real, full-
    range skill, never the training crutch.
    """

    def __init__(self, start: float = 0.0, end: float = 1.0, anneal_frac: float = 0.5,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = float(start), float(end)
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        frac = self.start + (self.end - self.start) * prog
        if self._last_set is None or abs(frac - self._last_set) >= 0.02 or prog >= 1.0:
            self.training_env.env_method("set_airspace_height_frac", frac)
            self._last_set = frac
        self.logger.record("manipularl/airspace_height_frac", frac)
        return True


class XyErrWeightAnneal(BaseCallback):
    """peg_in_hole only: linearly decay the dense potential-shaping's
    xy_err coefficient (RewardComputer.xy_err_weight, rewards.py's
    _potential) start -> end over the first `anneal_frac` of training,
    then hold at end. Default start=1.5 matches the value used
    unconditionally throughout; end=1.0 softens the pull-toward-the-hole
    gradient as training progresses -- a strong attraction term can itself
    drive overshoot/oscillation right at the target (classic high-gain-
    near-setpoint intuition), consistent with Task 1 v5's diagnostic
    showing imprecise, unsettled hovering concentrated near the hole
    (mean xy_err 5.7cm despite a 0.18cm best case) rather than during
    transit. `anneal_frac` deliberately long (e.g. 0.6-0.8) relative to
    other curricula -- this should stay near its stronger starting value
    while the policy is still learning to reach the hole at all, only
    softening once transit is largely mastered.
    """

    def __init__(self, start: float = 1.5, end: float = 1.0, anneal_frac: float = 0.6,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = float(start), float(end)
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        w = self.start + (self.end - self.start) * prog
        if self._last_set is None or abs(w - self._last_set) >= 0.02 or prog >= 1.0:
            self.training_env.env_method("set_xy_err_weight", w)
            self._last_set = w
        self.logger.record("manipularl/xy_err_weight", w)
        return True


class GraspDropPenaltyAnneal(BaseCallback):
    """peg_in_hole only: linearly ramp the grasp-drop penalty
    (RewardComputer.grasp_drop_pen, rewards.py's `_GRASP_DROP_PEN_DEFAULT`)
    UP from a small `start` to its full `end` strength over the first
    `anneal_frac` of training, then hold at end. Mirrors GraspCurriculumAnneal/
    ObstacleCountCurriculum's ramping-up pattern (not ramping down like most
    curricula here) -- E40/E41's fixed, full-strength-from-step-0 drop
    penalty fixed grasp-flicker cleanly but plausibly drove grasp-shyness
    as a side effect (a harsh, ever-present cost during early exploration,
    when accidental bad grasps are common, may make "never risk it" the
    locally-optimal response before the policy has a confident grasp habit
    worth protecting). Ramping it up lets early exploration tolerate cheap
    mis-grasps, reinforcing hold-stability only once grasping itself is
    already comfortable.
    """

    def __init__(self, start: float = 0.3, end: float = 2.0, anneal_frac: float = 0.5,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = float(start), float(end)
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        w = self.start + (self.end - self.start) * prog
        if self._last_set is None or abs(w - self._last_set) >= 0.02 or prog >= 1.0:
            self.training_env.env_method("set_grasp_drop_pen", w)
            self._last_set = w
        self.logger.record("manipularl/grasp_drop_pen", w)
        return True


class CollisionCurriculum(BaseCallback):
    """Stage collision consequences up over training, mirroring Phase 3's
    own real history: E9-E23 had collision preclude success but NOT end the
    episode early; E24 later added immediate termination once the
    underlying approach/grasp/carry skill was already decent (0.68-0.73
    success at the time). User-directed for Task 1, where collision rates
    have stayed high (0.30-0.84) and instant termination while the policy
    is still fumbling grasp attempts may be cutting it off before it
    experiences enough of the later episode to learn approach/align at all.

    Three stages (fractions of total_timesteps):
      1. [0, pen_anneal_frac]:      collision_terminal=False, collision_pen_mult
                                     ramps linearly pen_mult_start -> pen_mult_end
      2. (pen_anneal_frac, terminal_at_frac]: collision_terminal=False,
                                     collision_pen_mult held at pen_mult_end
      3. (terminal_at_frac, 1.0]:   collision_terminal=True, pen_mult at
                                     pen_mult_end (full strength)
    Collision ALWAYS precludes success (cfg.collision_is_failure, unchanged,
    fixed) in every stage -- only the per-step penalty magnitude and whether
    it also ends the episode immediately are staged.
    """

    def __init__(self, pen_mult_start: float = 0.3, pen_mult_end: float = 1.0,
                 pen_anneal_frac: float = 0.3, terminal_at_frac: float = 0.6,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.pen_mult_start = float(pen_mult_start)
        self.pen_mult_end = float(pen_mult_end)
        self.pen_anneal_steps = max(1, int(pen_anneal_frac * total_timesteps))
        self.terminal_steps = max(1, int(terminal_at_frac * total_timesteps))
        self._last_pen = None
        self._last_terminal = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.pen_anneal_steps)
        pen_mult = self.pen_mult_start + (self.pen_mult_end - self.pen_mult_start) * prog
        if self._last_pen is None or abs(pen_mult - self._last_pen) >= 0.02 or prog >= 1.0:
            self.training_env.env_method("set_collision_pen_mult", pen_mult)
            self._last_pen = pen_mult

        terminal = self.num_timesteps >= self.terminal_steps
        if terminal != self._last_terminal:
            self.training_env.env_method("set_collision_terminal", terminal)
            self._last_terminal = terminal

        self.logger.record("manipularl/collision_pen_mult", pen_mult)
        self.logger.record("manipularl/collision_terminal", float(terminal))
        return True


class GraspTriggerCurriculum(BaseCallback):
    """Linearly anneal the EE-peg grasp-catch radius start -> end (e.g.
    15cm -> the real 9cm _GRASP_TRIGGER) over the first `anneal_frac` of
    training, then hold at end. Same "legitimate difficulty ramp, disclosed,
    eval at the real value" pattern as GraspCurriculumAnneal/
    ObstacleCountCurriculum -- grasps land easily early (fast positive
    signal), tightening to the real catch radius the eval env always uses.
    """

    def __init__(self, start: float, end: float, anneal_frac: float = 0.5,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = float(start), float(end)
        self.anneal_steps = max(1, int(anneal_frac * total_timesteps))
        self._last_set = None

    def _on_step(self) -> bool:
        prog = min(1.0, self.num_timesteps / self.anneal_steps)
        radius = self.start + (self.end - self.start) * prog
        if self._last_set is None or abs(radius - self._last_set) >= 0.005 or prog >= 1.0:
            self.training_env.env_method("set_grasp_trigger", radius)
            self._last_set = radius
        self.logger.record("manipularl/grasp_trigger_cm", radius * 100.0)
        return True


class SuccessGatedAnneal(BaseCallback):
    """Stage-3 fix (E55/E56's diagnosis): every curriculum in this file
    anneals on a STEP-COUNT TIMER (`anneal_frac * total_timesteps`), which
    walks off the difficulty cliff on schedule whether or not the policy
    actually kept up -- if training is behind schedule at that point, the
    curriculum tightens anyway and the policy never recovers; if it's ahead,
    time is wasted at an already-mastered difficulty. This is a structural
    replacement, not another weight: it advances a single env parameter
    (`setter_method`, any `set_*` method in `env.py`, e.g. `set_xy_err_weight`)
    one `step_size` toward `end` only when the ROLLING TRAIN SUCCESS RATE
    (read the same way ManipulaRLTensorboardCallback already does -- per-
    episode `info["success"]` off the Monitor-wrapped vec-env buffer, no new
    instrumentation needed) exceeds `advance_thresh`, and REGRESSES one step
    back toward `start` when it drops below `regress_thresh` -- a genuine
    safety net no prior curriculum here had. `check_every` rollouts between
    adjustments (not every step) keeps this from thrashing on a single noisy
    episode; `window` matches the tensorboard callback's own rolling-success
    definition so "50% success" means the same thing in both places.
    """

    def __init__(self, setter_method: str, start: float, end: float,
                 step_size: float, advance_thresh: float = 0.5,
                 regress_thresh: float = 0.2, window: int = 100,
                 check_every: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.setter_method = setter_method
        self.start, self.end, self.step_size = float(start), float(end), float(step_size)
        self.advance_thresh, self.regress_thresh = advance_thresh, regress_thresh
        self.check_every = max(1, int(check_every))
        self._succ = deque(maxlen=window)
        self._cur = self.start
        self._n_rollouts = 0

    def _on_training_start(self) -> None:
        self.training_env.env_method(self.setter_method, self._cur)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self._succ.append(float(info.get("success", False)))
        return True

    def _on_rollout_end(self) -> None:
        self._n_rollouts += 1
        if self._n_rollouts % self.check_every != 0 or len(self._succ) < self.window_min():
            self._log()
            return
        rate = float(np.mean(self._succ))
        toward_end = 1.0 if self.end >= self.start else -1.0
        moved = False
        if rate >= self.advance_thresh and self._cur != self.end:
            self._cur = self._clip(self._cur + toward_end * self.step_size)
            moved = True
        elif rate < self.regress_thresh and self._cur != self.start:
            self._cur = self._clip(self._cur - toward_end * self.step_size)
            moved = True
        if moved:
            self.training_env.env_method(self.setter_method, self._cur)
        self._log(rate)

    def window_min(self) -> int:
        # don't act on a near-empty window early in training
        return max(10, self._succ.maxlen // 4)

    def _clip(self, v: float) -> float:
        lo, hi = (self.start, self.end) if self.end >= self.start else (self.end, self.start)
        return float(np.clip(v, lo, hi))

    def _log(self, rate: float = None) -> None:
        self.logger.record(f"manipularl/success_gated_{self.setter_method}", self._cur)
        if rate is not None:
            self.logger.record(f"manipularl/success_gated_{self.setter_method}_trigger_rate", rate)


class ManipulaRLTensorboardCallback(BaseCallback):
    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._succ = deque(maxlen=window)
        self._coll = deque(maxlen=window)
        self._depth = deque(maxlen=window)
        self._tilt = deque(maxlen=window)
        self._grasp_fail = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:      # only on episode end (Monitor adds "episode")
                continue
            self._succ.append(float(info.get("success", False)))
            self._coll.append(float(info.get("episode_collided", False)))
            if "max_depth" in info:
                self._depth.append(float(info["max_depth"]))
            if "min_tilt_deg" in info and np.isfinite(info["min_tilt_deg"]):
                self._tilt.append(float(info["min_tilt_deg"]))
            ga, gf = info.get("grasp_attempts", 0), info.get("grasp_failures", 0)
            if ga > 0:
                self._grasp_fail.append(gf / ga)

        if self._succ:
            self.logger.record("manipularl/success_rate", float(np.mean(self._succ)))
            self.logger.record("manipularl/collision_rate", float(np.mean(self._coll)))
        if self._depth:
            self.logger.record("manipularl/insert_depth_mean", float(np.mean(self._depth)))
        if self._tilt:
            self.logger.record("manipularl/insert_tilt_deg_mean", float(np.mean(self._tilt)))
        if self._grasp_fail:
            self.logger.record("manipularl/grasp_failure_rate", float(np.mean(self._grasp_fail)))
        return True
