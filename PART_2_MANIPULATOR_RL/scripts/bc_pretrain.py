#!/usr/bin/env python3
"""
bc_pretrain.py -- behaviour-clone a PPO policy on the scripted-expert demos,
so RL finetune starts already able to reach -> grasp -> carry -> place instead
of having to discover the gated grasp by exploration.

    python scripts/bc_pretrain.py --demos runs/demos_phase2 --out runs/phase2_bc \
        --config configs/ppo_phase2.yaml --epochs 40

Then finetune:
    python train.py --config configs/ppo_phase2.yaml \
        --warm-start runs/phase2_bc/model.zip \
        --warm-start-norm runs/demos_phase2/vecnormalize.pkl
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from manipularl.env import ManipulaRLEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="runs/demos_phase2")
    ap.add_argument("--config", default="configs/ppo_phase2.yaml")
    ap.add_argument("--out", default="runs/phase2_bc")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    d = np.load(Path(args.demos) / "demos.npz")
    O = torch.as_tensor(d["obs"], dtype=torch.float32)
    A = torch.as_tensor(np.clip(d["act"], -1.0, 1.0), dtype=torch.float32)
    N = O.shape[0]
    print(f"[bc] {N} transitions, obs {tuple(O.shape[1:])}, act {tuple(A.shape[1:])}")

    cfg = yaml.safe_load(open(args.config))
    pk = dict(cfg.get("hyperparams", {}).get("policy_kwargs", {}))
    fs = int(cfg.get("frame_stack", 3))

    # a throwaway env only to shape the policy (obs/action spaces must match)
    venv = DummyVecEnv([lambda: ManipulaRLEnv(phase=int(cfg["phase"]), split="train", seed=0)])
    if fs > 1:
        venv = VecFrameStack(venv, n_stack=fs)
    model = PPO("MlpPolicy", venv, policy_kwargs=pk, device="cpu", verbose=0)
    assert model.observation_space.shape[0] == O.shape[1], \
        f"obs dim mismatch: policy {model.observation_space.shape[0]} vs demos {O.shape[1]}"

    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)

    n_val = int(args.val_frac * N)
    perm = torch.randperm(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    for ep in range(1, args.epochs + 1):
        pol.train()
        order = tr_idx[torch.randperm(tr_idx.numel())]
        tot = 0.0
        for i in range(0, order.numel(), args.batch):
            b = order[i:i + args.batch]
            _, logp, ent = pol.evaluate_actions(O[b], A[b])
            mean_a = pol.get_distribution(O[b]).distribution.mean
            loss = -logp.mean() + 0.5 * F.mse_loss(mean_a, A[b]) - 1e-3 * ent.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            tot += float(loss) * b.numel()
        if ep % 5 == 0 or ep == 1:
            pol.eval()
            with torch.no_grad():
                vpred = pol.get_distribution(O[val_idx]).distribution.mean
                vmse = float(F.mse_loss(vpred, A[val_idx]))
            print(f"  epoch {ep:3d}  train_loss {tot/order.numel():+.4f}  val_action_mse {vmse:.4f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "model.zip"))
    print(f"[bc] saved {out}/model.zip")
    venv.close()


if __name__ == "__main__":
    main()
