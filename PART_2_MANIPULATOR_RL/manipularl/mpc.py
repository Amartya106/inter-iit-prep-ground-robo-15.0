"""
mpc.py -- CEM (Cross-Entropy Method) model-predictive control over the
learned DynamicsModel (world_model.py), for expert-demo generation.

User-directed: "is there a better way to get expert demos than IK" -> "Try
1. And 3." (1 = the compliant scripted expert in scripts/scripted_expert.py;
this is 3). Motivation: scripts/scripted_expert.py is a purely KINEMATIC,
memoryless controller -- solve IK for a target pose, apply it, repeat. It has
no notion of contact, no way to plan ahead, and (per E56/E58's diagnosis) no
way to converge position and orientation together rather than as competing
per-step corrections. MPC over a learned dynamics model is a genuinely
different kind of demonstrator: at every real step it samples many candidate
FUTURE action sequences, simulates each one THROUGH THE LEARNED MODEL, scores
the predicted outcomes, and only then commits to one action -- planning
ahead using a model of consequences, not reacting to a static kinematic
target.

WHY NOT the model's own reward head for scoring candidates: DynamicsModel's
reward head is trained on scripts/collect_dynamics_transitions.py's RANDOM-
ACTION rollouts (see world_model.py), where a random policy essentially never
grasps cleanly, let alone inserts -- insertion-success-scale reward events
are near-absent from its training data. It almost certainly learned the
common, dense, always-present shaping terms and nothing about the rare,
high-value states that matter for THIS task. Using it to rank candidates
would silently bias planning toward whatever the model happens to think a
"normal" reward looks like, not toward genuine task progress. Instead, this
module computes a HAND-CRAFTED planning cost directly from the model's
predicted RAW observation, using the env's own known observation layout
(manipularl/env.py's _assemble_obs, confirmed via the project's own
exploration -- see EXPERIMENTS.md) -- the exact same quantities
(xy_err/depth/tilt/grasped) the real reward is built from, not a learned
proxy for them.
"""
import numpy as np
import torch

from .world_model import DynamicsModel

# manipularl/env.py's _assemble_obs() layout (123-dim, single raw frame --
# see EXPERIMENTS.md's exploration report for the full index table):
_OBS_GRASPED = 30
_OBS_PEG_ROT6D = slice(34, 40)     # peg's rot6d orientation (2 columns of R)
_OBS_EE_TO_PEG = slice(40, 43)     # peg_pos - ee_pos
_OBS_XY_ERR_VEC = slice(55, 57)    # goal_xy - peg_xy
_OBS_HEIGHT_GAP = 57               # goal_z - peg_z (signed; +ve = peg above goal)
_OBS_DEPTH = 58                    # gated peg_depth


def _tilt_from_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
    """Decode a batch of rot6d vectors (..., 6) -> tilt from vertical (radians),
    matching env.py's own convention (arccos(|R[2,2]|), R[:,2] = the body's
    world-frame z-axis). Standard Gram-Schmidt reconstruction of the 3rd
    column (Zhou et al. 2019's continuous rotation representation) -- exactly
    how manipularl/env.py's own _rot6d() was built (first two columns of R),
    just run in reverse.
    """
    a, b = rot6d[..., 0:3], rot6d[..., 3:6]
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b - (a * b).sum(-1, keepdim=True) * a
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    c = torch.cross(a, b, dim=-1)
    return torch.arccos(torch.clamp(c[..., 2].abs(), 0.0, 1.0))


def planning_score(obs: torch.Tensor, action: torch.Tensor,
                   w_reach=3.0, w_xy=1.5, w_height=0.8, w_tilt=1.2,
                   w_depth=2.0, w_grasped=5.0, w_act=0.01) -> torch.Tensor:
    """Hand-crafted per-step planning reward from a (batch of) predicted RAW
    observation(s), shape (..., 123). Weights loosely mirror the real
    RewardComputer._potential's own terms (xy_err_weight~1.5, tilt_weight~1.2,
    z_gap~0.8, depth~2.0) for consistency with what the real reward actually
    values, plus a reach term (only active pre-grasp) and a grasped bonus
    neither of which _potential needs (grasping itself is handled by
    _handle_grasp's mechanics there, not a shaping term)."""
    grasped = obs[..., _OBS_GRASPED].clamp(0.0, 1.0)
    ee_to_peg_dist = obs[..., _OBS_EE_TO_PEG].norm(dim=-1)
    xy_err = obs[..., _OBS_XY_ERR_VEC].norm(dim=-1)
    height_gap = obs[..., _OBS_HEIGHT_GAP].clamp(min=0.0)
    depth = obs[..., _OBS_DEPTH]
    tilt = _tilt_from_rot6d(obs[..., _OBS_PEG_ROT6D])
    act_reg = (action[..., :7] ** 2).sum(-1)
    return (
        -w_reach * ee_to_peg_dist * (1.0 - grasped)
        - w_xy * xy_err
        - w_height * height_gap
        - w_tilt * tilt
        + w_depth * depth
        + w_grasped * grasped
        - w_act * act_reg
    )


class CEMPlanner:
    """Cross-Entropy Method MPC over a DynamicsModel. Receding-horizon: call
    `plan(raw_obs)` once per real env step, execute the returned first
    action, then call `plan()` again on the new real observation (this
    resamples the whole horizon fresh each call -- no warm-start carried
    between real steps, kept simple and robust rather than fastest; the
    model is tiny so full replanning every step is cheap, see __init__'s
    n_candidates/horizon defaults).
    """

    def __init__(self, model: DynamicsModel, obs_mean, obs_std, act_mean, act_std,
                 action_dim: int, horizon: int = 12, n_candidates: int = 256,
                 n_elites: int = 32, n_iters: int = 4, gamma: float = 0.95,
                 device: str = "cpu", seed: int = 0):
        self.model = model.to(device).eval()
        self.device = device
        self.obs_mean = torch.as_tensor(obs_mean, dtype=torch.float32, device=device)
        self.obs_std = torch.as_tensor(obs_std, dtype=torch.float32, device=device)
        self.act_mean = torch.as_tensor(act_mean, dtype=torch.float32, device=device)
        self.act_std = torch.as_tensor(act_std, dtype=torch.float32, device=device)
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.gamma = gamma
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def plan(self, raw_obs: np.ndarray) -> np.ndarray:
        H, K, A = self.horizon, self.n_candidates, self.action_dim
        mean = torch.zeros(H, A, device=self.device)
        std = torch.ones(H, A, device=self.device) * 0.6   # start near-full-range

        for _ in range(self.n_iters):
            noise = torch.randn(K, H, A, generator=self.gen).to(self.device)
            candidates = torch.clamp(mean.unsqueeze(0) + noise * std.unsqueeze(0), -1.0, 1.0)

            obs = torch.as_tensor(raw_obs, dtype=torch.float32, device=self.device)
            obs = obs.unsqueeze(0).repeat(K, 1)   # (K, obs_dim)
            total_score = torch.zeros(K, device=self.device)
            discount = 1.0
            for t in range(H):
                act = candidates[:, t, :]
                obs_n = (obs - self.obs_mean) / self.obs_std
                act_n = (act - self.act_mean) / self.act_std
                delta_n, _model_reward, _done_logit = self.model(obs_n, act_n)
                next_obs = obs + delta_n * self.obs_std
                total_score += discount * planning_score(next_obs, act)
                discount *= self.gamma
                obs = next_obs

            elite_idx = torch.topk(total_score, self.n_elites).indices
            elites = candidates[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=0.05)

        return mean[0].cpu().numpy().astype(np.float32)
