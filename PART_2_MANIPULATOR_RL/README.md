# Part 2, Manipulator RL

One PyBullet plus Gymnasium environment, with six task phases built on top: Reaching, Pick and Place, Obstacle Aware Manipulation, Peg in Hole Insertion, Dynamics Generalization, and a Robustness and Stress Test. All the decision making is learned through RL. The only non-learned part is PyBullet's own low-level joint position servo.

Robot: KUKA iiwa, a 7-DOF arm, bundled with `pybullet_data`.

## What I had to rebuild from the assignment scaffold

The scaffold runs, but cannot satisfy the problem statement as given. I rebuilt the parts that mattered, inside `manipularl/`.

| What was wrong | What I did about it |
|---|---|
| Stage 4 had no hole, so peg-in-hole was physically impossible. | Built a bore socket in `manipularl/env.py`, a ring of flat box segments (default 8, an octagon). PyBullet has no way to cut a true circular hole into a solid, so this is an approximation; `cfg.hole_segments` can raise the segment count for a rounder one, but that is a real physics change, not yet retrained against. Success needs a grasp, no collision, depth past 3cm, xy/tilt within tolerance, held for 10 steps. |
| The policy could not see obstacles at all. | Added the 6 nearest obstacles to the observation, fixed size, zero-filled when unused. |
| Stage 2 success only required the peg to still be grasped, so it could never release. | Success now requires an actual release, with the peg settled. |
| The obstacle RNG was never reseeded, so train and eval layouts were not separated. | `EpisodeSampler` draws every episode from a fixed seed range: train [0, 1e6), eval [1e6, 2e6). |
| No domain randomization, noise, metrics, or evaluation harness. | Added dynamics randomization (phase 5), noise/perturbation wrappers (phase 6, eval only), and `evaluate.py`. |
| The sim reset and reloaded all URDFs every episode, at 240Hz with one physics tick per step. | Made the scene persistent (25-30x faster resets). Control now runs at 20Hz, 12 physics substeps per step. |
| Actions were raw joint velocity commands. | Switched to delta joint position control. |

`manipularl/obstacles.py` is the scaffold's own obstacle generator, left unchanged.

## Setup

```bash
uv venv --python 3.10 .venv          # or: python3.10 -m venv .venv
uv pip install -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt
```

Needs Python 3.10, since that's what the PyBullet package is built for. Torch runs on CPU on purpose: the networks are small (about 100k parameters), so a small laptop GPU would actually be slower once you count the time spent moving data to and from it.

## Files

| Path | What's in it |
|---|---|
| `manipularl/env.py` | `ManipulaRLEnv`: persistent scene, delta position control, hole socket, obstacle collisions, observation assembly |
| `manipularl/configs.py` | the six `PhaseConfig` objects, interface constants, train/eval seed ranges |
| `manipularl/rewards.py` | the staged, potential-based reward and success rules per phase, the core RL design piece |
| `manipularl/randomization.py` | `EpisodeSampler`, builds per-episode layout and phase 5 dynamics |
| `manipularl/wrappers.py` | `NoisyObservation`, `NoisyAction`, `PegPerturbation`, eval only |
| `manipularl/make_env.py` | the vectorized environment wrapper stack |
| `manipularl/callbacks.py` | TensorBoard logging for task metrics |
| `train.py` | trains one phase (TQC or PPO) from a YAML config, with `--warm-start` |
| `evaluate.py` | runs evaluation episodes, writes an aggregate metrics CSV, `--noise-sweep` for phase 6 |
| `configs/` | phase configs, PPO is primary, TQC kept alongside |
| `scripts/` | chain runner, diagnostics, scripted expert, demo collection, behavior cloning, rollout recording, plotting |
| `EXPERIMENTS.md` | full run-by-run log: every attempt, every result, where the files live |
| `tests/test_env.py` | API compliance, determinism, seed separation, obstacle count, hole-solvability checks |

## Observation and action spaces (same shape across every phase)

**Action**: `Box(-1, 1, (8,))`. Seven delta joint position commands (0.05 rad/step), plus a gripper signal (above 0.5 closes, at or below opens).

**Observation**: `Box(-inf, inf, (123,))`, 369 after a 3-frame stack. Covers joint state, end effector pose and velocity, gripper/grasp state, peg pose and velocity (absolute and relative to the end effector), goal pose and the peg-to-goal vector, insertion depth, wrist force/torque, the 6 nearest obstacles (9 values each), and a one-hot phase indicator.

While working on Phase 4, I tried adding more signals to the observation (peg tilt, ring distance, a shaped depth signal, ring contact) to see if it helped fine insertion control. It didn't help, and it broke every checkpoint trained before that change, including all four checkpoints used here. So I moved that code to `manipularl/obs_h3_extras.py` instead, where it sits unused, and kept the real observation the same size as before. That way `runs/phase1_ppo`, `runs/phase2_ppo`, `runs/phase3_ppo_best`, and `runs/phase4_full_dream` all still load and run directly. The videos in `media/` and the numbers in the table below were freshly re-recorded and re-checked against these exact checkpoints to confirm that.

## How the reward is put together

I use potential-based shaping: `F = k * (gamma * Phi(s') - Phi(s))`. This form is proven not to change what the best policy is (Ng et al. 1999), so it's safe to add. `Phi` is the negative distance to the current sub-goal. As the episode moves along, the sub-goal switches through one-time bonuses: reach, grasp, carry, align, insert.

On top of that there's a penalty for jerky motion, a time penalty (so the policy can't just hover near the goal forever for free reward), a per-step penalty for colliding, and a large bonus for actually finishing the task. From phase 3 onward, avoiding collisions is required for success, not just a nice-to-have.

## Training

```bash
# Phase 1 (solved):
PYTHONPATH=. .venv/bin/python train.py --config configs/ppo_phase1.yaml --out-dir runs/phase1_ppo

# Phase 2 (solved), FROM SCRATCH, not warm-started, a phase-1 warm start is
# actually negative transfer here, see report §4.2 or EXPERIMENTS.md:
PYTHONPATH=. .venv/bin/python train.py --config configs/ppo_phase2_scratch.yaml --out-dir runs/phase2_ppo
PYTHONPATH=. .venv/bin/python scripts/diagnose_phase2.py --run runs/phase2_ppo --mode coldstart

# Phases 3-5, warm-started chain, detached:
setsid bash -c 'bash scripts/chain_p3p4p5.sh > runs/chain_p3p4p5.log 2>&1' </dev/null & disown
tensorboard --logdir runs
```

On algorithm choice: I planned to use TQC as the main algorithm, since it's usually a good fit for PyBullet tasks with lots of contact. But on this CPU (no CUDA, and a small laptop GPU would actually be slower at this model size) it only managed 120-330 steps per second. It solved phase 1 (0.935) but got stuck at 0.01 on phase 2, so I used PPO for every phase instead, which runs much faster: roughly 1,200-2,000 steps per second with 16 environments running in parallel. Configs for both are in `configs/`. Every run, including the ones that failed, is logged in `EXPERIMENTS.md`.

## Evaluating

```bash
PYTHONPATH=. .venv/bin/python evaluate.py --run runs/phase4_ppo --phase 4 --episodes 200
# Generalization check: same checkpoint, two separate --phase calls (dynamics
# randomization comes from which phase's config gets loaded, not from a
# --conditions value, see evaluate.py's own docstring):
PYTHONPATH=. .venv/bin/python evaluate.py --run runs/phase4_ppo --phase 5 --episodes 200
PYTHONPATH=. .venv/bin/python evaluate.py --run runs/phase4_ppo --phase 4 --noise-sweep
PYTHONPATH=. .venv/bin/python scripts/plot_eval.py --csv "results/eval_phase[1-4].csv" --out plots/
PYTHONPATH=. .venv/bin/python scripts/record_rollout.py --run runs/phase3_ppo_best --phase 3 \
    --episodes 5 --out media/phase3_obstacle_aware.mp4
```

Metrics per condition: success rate, collision rate, completion time, path efficiency, final position error, grasp failure rate, insertion depth, tilt.

## Where things landed (200 eval-split episodes, unseen layouts, deterministic policy)

| Phase | success | collision | what happened |
|---|---|---|---|
| 1 Reaching | 1.00 (PPO), 0.935 (TQC) | 0.00 | solved |
| 2 Pick & Place | 0.995 | 0.00 | solved from scratch with reward shaping and an annealed grasp curriculum. Report §4.2 |
| 3 Obstacle-Aware | 0.86 | 0.08 | best result after a long series of reward-shaping attempts. EXPERIMENTS.md |
| 4 Peg-in-Hole | 0.015-0.02 full task, 0.073 insert-only sub-task | 0.23-0.43 | genuinely hard, still not solved on the full task. I found and fixed a real bug in the grasp code (E79). A hand-coded controller with perfect information confirmed the fix works, tripling its success rate, but training an RL policy against the fix still did not move the full-task number (E80). An earlier finding, that a redundant arm joint adds difficulty, still holds too; the bug fix does not overturn it, just adds to the picture. Training on just the insertion step by itself did reach a new best on that smaller piece (0.073, up from 0.040). Full story: EXPERIMENTS.md E1-E80 |
| 5 Dynamics Gen. | phase 4 checkpoint, evaluated | EXPERIMENTS.md E62 | success held flat under unseen dynamics, collision got meaningfully worse |
| 6 Robustness | phase 4 checkpoint, evaluated | EXPERIMENTS.md E63 | collision degrades gracefully with noise, but the hard-won fine-precision behavior is brittle and does not survive any tested noise level |

Chart: `plots/per_phase_summary.png`. Full log, every experiment, every dead end: `EXPERIMENTS.md`.

## Tests

```bash
PYTHONPATH="" .venv/bin/python -m pytest -q tests/test_env.py
```

(Empty `PYTHONPATH` avoids a broken ROS-supplied pytest plugin, in case ROS is sourced in the same shell.)
