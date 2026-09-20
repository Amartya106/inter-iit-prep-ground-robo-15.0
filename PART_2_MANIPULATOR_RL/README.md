
# Part 2: Manipulator Reinforcement Learning

A PyBullet plus Gymnasium environment for a 7-DOF KUKA iiwa arm, with six manipulation tasks: Reaching, Pick and Place, Obstacle-Aware Manipulation, Peg-in-Hole Insertion, Dynamics Generalization, and Robustness Testing.

I built this to train and test RL policies on manipulation tasks that get harder step by step, from simple reaching up to precise insertion under noise and changing conditions.

**Robot:** KUKA iiwa, 7 DOF, bundled with `pybullet_data`.

**Framework:** PyBullet, Gymnasium, Stable-Baselines3.

All the decision making is learned through RL. The only non-learned part is PyBullet's own low-level joint position servo.

## What I Rebuilt from the Assignment Scaffold

The scaffold I was given ran, but several parts needed fixing or building from scratch to do what the tasks needed.

| Issue in the scaffold | Change |
|---|---|
| Phase 4 had no physical hole, making peg-in-hole insertion impossible | Built an approximate bore socket using flat box segments |
| The policy could not observe obstacles | Added the six nearest obstacles to the observation |
| Phase 2 success only required the peg to remain grasped | Changed success to require an actual release and a settled peg |
| Obstacle RNG was not reseeded | Added separate train and evaluation seed ranges |
| No domain randomization, noise, metrics, or evaluation harness | Added dynamics randomization, evaluation-only perturbation wrappers, and `evaluate.py` |
| Simulation reset reloaded URDFs every episode | Made the scene persistent, which makes resets about 25–30x faster |
| Actions used raw joint velocity commands | Switched to delta joint position control |

The scaffold's obstacle generator, `manipularl/obstacles.py`, is unchanged.

### Peg-in-Hole Geometry

The original scaffold had no hole for Phase 4. I built a socket in `manipularl/env.py` out of a ring of flat box segments.

PyBullet has no way to cut a true circular hole into a solid, so this socket is an approximation. The default uses eight segments, which looks like an octagon.

You can raise the segment count through `cfg.hole_segments` for a rounder hole, but that changes the physics, and I have not retrained against it yet.

Insertion success requires:
- A grasped peg
- No collision
- Insertion depth greater than 3 cm
- XY position and tilt within tolerance
- The conditions to remain satisfied for 10 steps

## Setup

This project needs Python 3.10, since that's what the PyBullet package is built for.

Create a virtual environment and install dependencies:

```bash
uv venv --python 3.10 .venv

uv pip install -r requirements.txt
```

Or with the standard Python virtual environment:

```bash
python3.10 -m venv .venv

.venv/bin/pip install -r requirements.txt
```

### Compute

I used both CPU and GPU across this project, not one exclusively. Phases 1-4's main checkpoints (`.venv`) trained on CPU. The later Phase 4 insertion investigation (R1-R6, `.venv_gpu_test`) trained on GPU. I benchmarked the two directly (`runs/bench_cpu_clean.log` vs `runs/bench_gpu_clean.log`): about 1,450-1,490 steps/s on CPU versus 1,500-1,550 on GPU. PyBullet's own physics stepping is the real bottleneck here, not the ~100k-parameter network, so which one you use barely changes the number.

## Project Structure

| Path | Description |
|---|---|
| `manipularl/env.py` | `ManipulaRLEnv`: persistent scene, delta position control, hole socket, obstacle collisions, observation assembly |
| `manipularl/configs.py` | Six phase configurations, interface constants, train/evaluation seed ranges |
| `manipularl/rewards.py` | Staged potential-based rewards and phase-specific success rules |
| `manipularl/randomization.py` | `EpisodeSampler`, episode layouts, Phase 5 dynamics randomization |
| `manipularl/wrappers.py` | `NoisyObservation`, `NoisyAction`, and `PegPerturbation` wrappers |
| `manipularl/make_env.py` | Vectorized environment wrapper stack |
| `manipularl/callbacks.py` | TensorBoard logging for task metrics |
| `train.py` | Trains one phase using PPO or TQC, with optional warm-starting |
| `evaluate.py` | Evaluation episodes, aggregate metrics CSV, and noise sweeps |
| `configs/` | Phase-specific PPO and TQC configurations |
| `scripts/` | Training chain, diagnostics, scripted expert, demonstrations, behavior cloning, rollout recording, and plotting |
| `tests/test_env.py` | API compliance, determinism, seed separation, obstacle count, and hole-solvability checks |
| `EXPERIMENTS.md` | Run-by-run experiment log, including failed attempts and results |
| `media/` | Recorded demonstrations |
| `runs/` | Training runs and checkpoints |
| `results/` | Evaluation metrics |
| `plots/` | Generated evaluation plots |

## Observation and Action Spaces

The observation and action spaces have the same shape across all six phases.

### Action

```text
Box(-1, 1, (8,))
```

- Seven joint delta-position commands, scaled to 0.05 rad per step.
- One gripper signal:
  - Above 0.5: close
  - At or below 0.5: open

### Observation

```text
Box(-inf, inf, (123,))
```

A three-frame stack produces 369 values.

The observation includes:

- Joint state
- End-effector pose and velocity
- Gripper and grasp state
- Peg pose and velocity, both absolute and relative to the end effector
- Goal pose and peg-to-goal vector
- Insertion depth
- Wrist force and torque
- Six nearest obstacles, with nine values per obstacle
- One-hot phase indicator

### Observation Design During Phase 4

While working on Phase 4, I experimented with adding peg tilt, ring distance, shaped depth, and ring contact signals to improve insertion control.

These extra signals did not help, and they broke every checkpoint trained before that change.

I moved the experimental observation code to `manipularl/obs_h3_extras.py` and kept the main observation shape unchanged.

That way these checkpoints still load and run directly:

```text
runs/phase1_ppo
runs/phase2_ppo
runs/phase3_ppo_best
runs/phase4_full_dream
```

I re-recorded the videos and re-checked the numbers below against these exact checkpoints.

## Reward Design

I use potential-based shaping:

```
F(s, s') = k * (gamma * Phi(s') - Phi(s))
```

where `Phi` is the negative distance to the current sub-goal.

The sub-goal moves through the task in order:

```text
Reach → Grasp → Carry → Align → Insert
```

The reward also includes:

- A penalty for jerky motion
- A time penalty
- A per-step collision penalty
- A large bonus for completing the task

From Phase 3 onward, avoiding collisions is required for success, not just a nice-to-have.

This form is proven not to change what the best policy is (Ng et al. 1999), so it's safe to add.

## Training

I use PPO as the main algorithm, and kept TQC configs around for comparison.

### Phase 1: Reaching

```bash
PYTHONPATH=. .venv/bin/python train.py \
  --config configs/ppo_phase1.yaml \
  --out-dir runs/phase1_ppo
```

### Phase 2: Pick and Place

I train Phase 2 from scratch instead of warm-starting from Phase 1. In my tests, a Phase 1 warm start actually hurt, not helped.

```bash
PYTHONPATH=. .venv/bin/python train.py \
  --config configs/ppo_phase2_scratch.yaml \
  --out-dir runs/phase2_ppo

PYTHONPATH=. .venv/bin/python scripts/diagnose_phase2.py \
  --run runs/phase2_ppo \
  --mode coldstart
```

### Phases 3–5: Warm-Started Training Chain

```bash
setsid bash -c \
  'bash scripts/chain_p3p4p5.sh > runs/chain_p3p4p5.log 2>&1' \
  </dev/null & disown
```

Watch training with TensorBoard:

```bash
tensorboard --logdir runs
```

### Algorithm Choice

I originally planned to use TQC as the main algorithm, since it's usually a good fit for continuous-control tasks.

But on this CPU, TQC only managed about 120–330 steps per second. It solved Phase 1 (0.935) but got stuck around 0.01 on Phase 2.

So I used PPO for the rest, which runs much faster here: about 1,200–2,000 steps per second with 16 environments running in parallel.

These are numbers from this specific setup, not a general claim that PPO always beats TQC.

Every run, including the ones that failed, is logged in `EXPERIMENTS.md`.

## Evaluation

Evaluation uses different episode seeds than training, and the policy acts deterministically.

Run evaluation for a trained checkpoint:

```bash
PYTHONPATH=. .venv/bin/python evaluate.py \
  --run runs/phase4_ppo \
  --phase 4 \
  --episodes 200
```

### Dynamics Generalization

Evaluate the same checkpoint under Phase 5 dynamics:

```bash
PYTHONPATH=. .venv/bin/python evaluate.py \
  --run runs/phase4_ppo \
  --phase 5 \
  --episodes 200
```

Which phase config you load decides how much dynamics randomization gets used.

### Robustness Testing

Run the noise sweep:

```bash
PYTHONPATH=. .venv/bin/python evaluate.py \
  --run runs/phase4_ppo \
  --phase 4 \
  --noise-sweep
```

### Plot Evaluation Results

```bash
PYTHONPATH=. .venv/bin/python scripts/plot_eval.py \
  --csv "results/eval_phase[1-4].csv" \
  --out plots/
```

### Record a Demonstration

```bash
PYTHONPATH=. .venv/bin/python scripts/record_rollout.py \
  --run runs/phase3_ppo_best \
  --phase 3 \
  --episodes 5 \
  --out media/phase3_obstacle_aware.mp4
```

### Evaluation Metrics

Evaluation records:

- Success rate
- Collision rate
- Completion time
- Path efficiency
- Final position error
- Grasp failure rate
- Insertion depth
- Peg tilt

## Results

These results are from 200 evaluation episodes on unseen layouts, with the policy acting deterministically.

| Phase | Success rate | Collision rate | Outcome |
|---|---:|---:|---|
| 1: Reaching | 1.00 PPO; 0.935 TQC | 0.00 | Solved |
| 2: Pick and Place | 0.995 | 0.00 | Solved from scratch |
| 3: Obstacle-Aware | 0.86 | 0.08 | Partial success |
| 4: Peg-in-Hole | 0.015–0.02 full task | 0.23–0.43 | Not solved |
| 5: Dynamics Generalization | See `EXPERIMENTS.md` E62 | See experiment log | Evaluated using Phase 4 checkpoint |
| 6: Robustness | See `EXPERIMENTS.md` E63 | See experiment log | Evaluated using Phase 4 checkpoint |

### Phase 1: Reaching

PPO solved reaching, with a success rate of 1.00. TQC reached 0.935.

### Phase 2: Pick and Place

This task reached 0.995, with zero collisions in evaluation.

Training from scratch worked better here than warm-starting from Phase 1.

### Phase 3: Obstacle-Aware Manipulation

The best result I got was 0.86 success, 0.08 collision.

That came after a long series of reward-shaping attempts, all logged in `EXPERIMENTS.md`.

### Phase 4: Peg-in-Hole Insertion

The full peg-in-hole task is still not solved.

The full-task success rate is about 1.5–2%. An insertion-only sub-task reached 7.3%, up from 4.0%.

While debugging, I found and fixed a bug in the grasp code (E79). A hand-coded controller with perfect information confirmed the fix worked and raised its success rate.

But training an RL policy against the fix still did not improve the full-task result (E80).

An earlier experiment also found that the extra arm joint adds difficulty. The grasp fix doesn't overturn that, it just adds another factor.

The insertion-only result is progress on a smaller piece, not a solution to the whole task.

See `EXPERIMENTS.md` E1–E80 for the complete history.

### Phase 5: Dynamics Generalization

I evaluated the Phase 4 checkpoint under Phase 5 dynamics.

Success stayed about flat under unseen dynamics, but collisions got worse.

Full numbers are in `EXPERIMENTS.md` E62.

### Phase 6: Robustness and Stress Testing

I evaluated the same checkpoint with observation noise, action noise, and peg perturbations.

Collisions got gracefully worse with noise, but the fine, hard-won insertion precision was brittle and didn't survive any tested noise level.

Full numbers are in `EXPERIMENTS.md` E63.

### Results Visualization

```text
plots/per_phase_summary.png
```

`EXPERIMENTS.md` has the full record: every attempt, every result, every dead end.

## Tests

Run the environment tests:

```bash
PYTHONPATH="" .venv/bin/python -m pytest -q tests/test_env.py
```

An empty `PYTHONPATH` avoids a broken ROS-supplied pytest plugin, in case ROS is sourced in the same shell.

The tests cover:

- Gymnasium API compliance
- Determinism
- Train/evaluation seed separation
- Obstacle count
- Hole-solvability checks

## Limitations

- **Peg-in-hole is still not solved.** The full task has a low success rate even after a lot of experimenting.
- **The socket shape is only an approximation.** The hole is built from flat box segments, not a true circular bore.
- **Extra observation signals didn't help.** The insertion-specific signals I tried adding did not improve the policy and broke checkpoint compatibility.
- **Generalization has a real gap.** Collisions got worse under unseen dynamics.
- **Robustness is limited.** The fine insertion precision did not survive any tested noise level.
- **Compute was a limit.** Training speed was capped by CPU performance, especially for TQC.

These are limits of what I built and tried, not a claim about what's possible with a different algorithm, observation design, or training setup.

## Experiment Log

`EXPERIMENTS.md` contains the full run-by-run record, including:

- Training configurations
- Algorithm comparisons
- Reward-shaping attempts
- Failed experiments
- Checkpoint locations
- Phase 4 debugging
- Generalization and robustness evaluations

## Summary

This project is one shared PyBullet and Gymnasium environment across six RL task phases.

It has delta joint position control, obstacle-aware observations, staged reward shaping, dynamics randomization, evaluation-only perturbations, and a reproducible evaluation setup.

Reaching and Pick and Place are solved with high success rates. Obstacle-Aware Manipulation is partially solved. Peg-in-Hole is still an open problem, though the insertion-only sub-task shows some progress.

The main result is a working setup for studying manipulation tasks that get progressively harder, with clear metrics and a full record of what worked and what didn't.
