"""
record_rollout.py -- render trained-policy episodes to an mp4.

    PYTHONPATH=. .venv/bin/python scripts/record_rollout.py \
        --run runs/phase4_tqc --phase 4 --episodes 3 --out media/phase4.mp4
    # stress-test rollout (Phase 6 eval-time perturbations):
    PYTHONPATH=. .venv/bin/python scripts/record_rollout.py \
        --run runs/phase4_tqc --phase 6 --episodes 2 --noise 0.02 --out media/phase6_stress.mp4

Runs headless (PyBullet DIRECT + software renderer, no display needed) and
grabs a camera image each control step.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pybullet as p

from sb3_contrib import TQC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from manipularl.env import ManipulaRLEnv
from manipularl.wrappers import NoisyObservation, NoisyAction, PegPerturbation
from manipularl.configs import EVAL_SEED_LO

W, H = 640, 480


def cam(client):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.45, 0.0, 0.05], distance=1.7,
        yaw=52, pitch=-32, roll=0, upAxisIndex=2, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(fov=55, aspect=W / H, nearVal=0.05, farVal=4.0,
                                        physicsClientId=client)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER,
                           physicsClientId=client)
    rgb = np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--algo", default="tqc", choices=["tqc", "ppo"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--failures-only", action="store_true",
                    help="only keep episodes where info['success'] is False "
                         "(collision or otherwise) -- rolls extra episodes "
                         "as needed until --episodes failures are collected, "
                         "up to --max-attempts.")
    ap.add_argument("--successes-only", action="store_true",
                    help="only keep episodes where info['success'] is True -- "
                         "rolls extra episodes as needed until --episodes "
                         "successes are collected, up to --max-attempts. "
                         "Mutually exclusive with --failures-only.")
    ap.add_argument("--max-attempts", type=int, default=200,
                    help="safety cap on total episodes rolled when "
                         "--failures-only or --successes-only is set.")
    ap.add_argument("--align-only", action="store_true",
                    help="Task-1 (reach+align) variant: cfg_overrides={align_only: True} "
                         "-- mirrors evaluate.py's flag")
    ap.add_argument("--insert-only", action="store_true",
                    help="Task-2 (insert) variant: cfg_overrides={insert_only: True} "
                         "-- mirrors evaluate.py's flag")
    args = ap.parse_args()
    if args.failures_only and args.successes_only:
        raise SystemExit("--failures-only and --successes-only are mutually exclusive")

    run = Path(args.run)
    model_path = run / "model.zip"
    if not model_path.exists():
        model_path = run / "best" / "best_model.zip"
    Model = TQC if args.algo == "tqc" else PPO
    model = Model.load(str(model_path), device="cpu")

    cfg_overrides = None
    if args.align_only:
        cfg_overrides = {"align_only": True}
    elif args.insert_only:
        cfg_overrides = {"insert_only": True}

    base = ManipulaRLEnv(phase=args.phase, split="eval", seed=EVAL_SEED_LO,
                         cfg_overrides=cfg_overrides)
    env = base
    if args.noise > 0.0:
        env = NoisyObservation(env, args.noise)
        env = NoisyAction(env, args.noise)
        # Same fix as evaluate.py's build_eval_env: no force floor, so a
        # small --noise value doesn't get disproportionately amplified by a
        # 2N minimum peg impulse (this call site is already guarded by
        # `if args.noise > 0.0`, so args.noise*300 is never literally 0 here,
        # but the floor was still inflating small-noise recordings).
        env = PegPerturbation(env, force=args.noise * 300)
    venv = DummyVecEnv([lambda: env])
    if args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack)
    venv = VecNormalize.load(str(run / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(tempfile.mkdtemp(prefix="rollout_"))
    n_succ, fi = 0, 0
    kept_eps = 0
    attempt = 0
    target = args.episodes
    filtering = args.failures_only or args.successes_only
    max_attempts = args.max_attempts if filtering else args.episodes
    while kept_eps < target and attempt < max_attempts:
        attempt += 1
        obs = venv.reset()
        done = False
        steps = 0
        info = {}
        ep_frames = []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, dones, infos = venv.step(a)
            done = bool(dones[0]); info = infos[0]
            ep_frames.append(cam(base.client))
            steps += 1
        success = bool(info.get("success", False))
        n_succ += int(success)
        if args.failures_only:
            keep = not success
        elif args.successes_only:
            keep = success
        else:
            keep = True
        if keep:
            kept_eps += 1
            for frame in ep_frames:
                Image.fromarray(frame).save(frame_dir / f"f{fi:06d}.png"); fi += 1
            for _ in range(args.fps // 2):    # hold the last frame briefly
                Image.fromarray(ep_frames[-1]).save(frame_dir / f"f{fi:06d}.png"); fi += 1
            print(f"  [kept {kept_eps}/{target}] attempt {attempt}: steps={steps} success={success}")
        else:
            reason = "success" if args.failures_only else "failure"
            print(f"  [discard, {reason}] attempt {attempt}: steps={steps}")
    if kept_eps < target:
        print(f"[record] WARNING: only collected {kept_eps}/{target} matching episodes "
              f"after {attempt} attempts (hit --max-attempts)")
    venv.close()

    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(args.fps), "-i", str(frame_dir / "f%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         str(args.out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(frame_dir, ignore_errors=True)
    if args.failures_only:
        print(f"[record] {kept_eps} failure episodes kept out of {attempt} attempts "
              f"({n_succ} successes discarded) -> {args.out}")
    else:
        print(f"[record] {n_succ}/{args.episodes} successful -> {args.out}")


if __name__ == "__main__":
    main()
