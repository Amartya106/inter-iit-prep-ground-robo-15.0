"""
plot_eval.py -- charts from the evaluation CSVs.

    PYTHONPATH=. .venv/bin/python scripts/plot_eval.py --csv runs/*/evaluation.csv --out plots/

Produces:
  * per_phase_summary.png   -- success / collision rate per phase (in-dist)
  * generalization.png       -- success rate broken out by phase (in_dist =
                               --phase 4, unseen_dynamics = the SAME
                               checkpoint's --phase 5 result -- dynamics
                               randomization comes from which phase's config
                               is loaded, not a --conditions value; see
                               evaluate.py's own docstring), per PS
  * noise_degradation.png    -- success + collision vs noise magnitude (Phase 6)
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csvs):
    frames = []
    for pat in csvs:
        for f in glob.glob(pat):
            df = pd.read_csv(f)
            df["run"] = Path(f).parent.name
            frames.append(df)
    if not frames:
        raise SystemExit("no evaluation CSVs matched")
    return pd.concat(frames, ignore_index=True)


def per_phase_summary(df, out):
    d = df[df["condition"] == "in_dist"].sort_values("phase")
    if d.empty:
        return
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, d["success_rate"], 0.4, label="success rate", color="#2a9d8f")
    ax.bar(x + 0.2, d["collision_rate"], 0.4, label="collision rate", color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels([f"P{int(p)}" for p in d["phase"]])
    ax.set_ylim(0, 1); ax.set_ylabel("rate"); ax.set_title("In-distribution performance by phase")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "per_phase_summary.png", dpi=130)
    plt.close(fig)


def generalization(df, out):
    d = df[~df["condition"].str.startswith("noise")]
    if d["condition"].nunique() < 2:
        return
    piv = d.pivot_table(index="phase", columns="condition", values="success_rate")
    ax = piv.plot(kind="bar", figsize=(8, 4), colormap="viridis")
    ax.set_ylim(0, 1); ax.set_ylabel("success rate")
    ax.set_title("Success rate by generalization condition (drop = generalization gap)")
    ax.figure.tight_layout(); ax.figure.savefig(out / "generalization.png", dpi=130)
    plt.close(ax.figure)


def noise_degradation(df, out):
    d = df[df["condition"].str.startswith("noise=")].copy()
    if d.empty:
        return
    d["sigma"] = d["condition"].str.split("=").str[1].astype(float)
    d = d.sort_values("sigma")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d["sigma"], d["success_rate"], "o-", color="#2a9d8f", label="success rate")
    ax.plot(d["sigma"], d["collision_rate"], "s--", color="#e76f51", label="collision rate")
    ax.set_xlabel("noise magnitude (obs+act sigma)"); ax.set_ylabel("rate")
    ax.set_ylim(0, 1); ax.set_title("Degradation under eval-time noise (Phase 6)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "noise_degradation.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load(args.csv)
    per_phase_summary(df, out)
    generalization(df, out)
    noise_degradation(df, out)
    df.to_csv(out / "all_evaluations.csv", index=False)
    print(f"[plot] wrote charts + all_evaluations.csv to {out}/")


if __name__ == "__main__":
    main()
