"""
world_model.py -- Part 2b of the "dreaming"/model-based RL plan (Task 1b/
descend_only). A small MLP dynamics model, (obs, action) -> (delta_obs,
reward, done_logit), trained via ordinary supervised regression/BCE on real
collected transitions (scripts/collect_dynamics_transitions.py).

Deliberately NOT a full Dreamer-style RSSM/latent-imagination model -- this
project's observations are already compact vectors (see
ManipulaRLEnv._assemble_obs, no images anywhere), so there's no encoder to
learn and no latent state to maintain; a direct feedforward model on the raw
observation is the right-sized adaptation (closer to MBPO's single-step
dynamics model than to Dreamer proper). Predicting delta_obs (next - current)
rather than next_obs directly is a standard, better-conditioned target since
most observation dimensions barely change in one 1/30s-scale control step.
"""
import numpy as np
import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.delta_obs_head = nn.Linear(hidden, obs_dim)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)   # logit, BCEWithLogitsLoss

    def forward(self, obs, action):
        x = self.net(torch.cat([obs, action], dim=-1))
        return self.delta_obs_head(x), self.reward_head(x).squeeze(-1), self.done_head(x).squeeze(-1)

    def predict_next_obs(self, obs, action):
        delta, reward, done_logit = self.forward(obs, action)
        return obs + delta, reward, torch.sigmoid(done_logit)


def train_dynamics_model(transitions_path: str, epochs: int = 60, batch_size: int = 256,
                          lr: float = 1e-3, val_frac: float = 0.15, seed: int = 0,
                          device: str = "cpu"):
    """Supervised-train a DynamicsModel on collected transitions. Returns
    (model, obs_mean, obs_std, action_mean, action_std, history) -- inputs
    are standardized (helps a plain MLP converge; predictions are
    de-standardized by callers via predict_next_obs on RAW obs/action, since
    the model itself normalizes internally -- see DynamicsModel usage in
    imagine_rollout, which passes raw obs/action through a wrapper).
    Reports train/val loss every 10 epochs and the FINAL held-out numbers --
    per the plan, this must be checked/reported BEFORE the model is wired
    into any policy training.
    """
    data = np.load(transitions_path)
    obs, action, next_obs = data["obs"], data["action"], data["next_obs"]
    reward, done = data["reward"], data["done"]
    n = obs.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-6
    act_mean, act_std = action.mean(0), action.std(0) + 1e-6

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    obs_n = (obs - obs_mean) / obs_std
    act_n = (action - act_mean) / act_std
    delta_obs = (next_obs - obs) / obs_std   # standardize the delta target too

    obs_t, act_t = to_t(obs_n), to_t(act_n)
    delta_t, rew_t, done_t = to_t(delta_obs), to_t(reward), to_t(done)

    model = DynamicsModel(obs.shape[1], action.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        perm = train_idx[rng.permutation(len(train_idx))]
        ep_loss = 0.0
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            pred_delta, pred_rew, pred_done_logit = model(obs_t[b], act_t[b])
            loss = mse(pred_delta, delta_t[b]) + mse(pred_rew, rew_t[b]) + bce(pred_done_logit, done_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()) * len(b)
        ep_loss /= len(train_idx)

        model.eval()
        with torch.no_grad():
            pd, pr, pdl = model(obs_t[val_idx], act_t[val_idx])
            val_delta_mse = float(mse(pd, delta_t[val_idx]).item())
            val_rew_mse = float(mse(pr, rew_t[val_idx]).item())
            val_done_bce = float(bce(pdl, done_t[val_idx]).item())
            val_done_acc = float(((torch.sigmoid(pdl) > 0.5).float() == done_t[val_idx]).float().mean().item())
        history.append(dict(epoch=epoch, train_loss=ep_loss, val_delta_mse=val_delta_mse,
                            val_rew_mse=val_rew_mse, val_done_bce=val_done_bce, val_done_acc=val_done_acc))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}: train_loss={ep_loss:.5f} val_delta_mse={val_delta_mse:.5f} "
                  f"val_rew_mse={val_rew_mse:.5f} val_done_acc={val_done_acc:.3f}")

    return model, obs_mean, obs_std, act_mean, act_std, history


def predict_next_raw_obs(model: "DynamicsModel", raw_obs: np.ndarray, raw_action: np.ndarray,
                         obs_mean: np.ndarray, obs_std: np.ndarray,
                         act_mean: np.ndarray, act_std: np.ndarray, device: str = "cpu"):
    """Single-step prediction in RAW (unstandardized) units -- the model
    itself operates on standardized inputs/targets (see train_dynamics_model:
    delta_obs target is divided by obs_std), so this wrapper standardizes
    obs/action going in and de-standardizes the delta coming out. Reward is
    NOT standardized during training (plain MSE on raw reward), so the
    reward head's output is already in raw units, no conversion needed.
    Returns (next_raw_obs, reward, done_prob) as numpy arrays/floats.
    """
    with torch.no_grad():
        obs_t = torch.as_tensor((raw_obs - obs_mean) / obs_std, dtype=torch.float32, device=device)
        act_t = torch.as_tensor((raw_action - act_mean) / act_std, dtype=torch.float32, device=device)
        if obs_t.dim() == 1:
            obs_t, act_t = obs_t.unsqueeze(0), act_t.unsqueeze(0)
        delta_n, reward, done_logit = model(obs_t, act_t)
        next_raw_obs = raw_obs + delta_n.cpu().numpy()[0] * obs_std
        return next_raw_obs, float(reward.item()), float(torch.sigmoid(done_logit).item())
