# Part 2, Manipulator RL: Technical Report

Inter-IIT Tech Meet 15.0 Prepathon, Ground Robotics

---

## 1. Task

There is one PyBullet + Gymnasium environment and one observation/action interface. It has **six phases**. Each phase adds something new on top of the last one. None of them redefine what came before.

| Phase | Task | Adds |
|---|---|---|
| 1 | Reaching | move the end-effector to a randomized target from a randomized arm pose |
| 2 | Pick & Place | reach, grasp a peg, carry it to a target, then **release** it |
| 3 | Obstacle-Aware Manipulation | randomized static obstacles are added; a run that collides is not counted as a success |
| 4 | Peg-in-Hole Insertion | reach, grasp, move past obstacles, align, then insert the peg to a target depth within position and angle tolerance |
| 5 | Standard & Dynamics Generalization | the Phase-4 task, evaluated on layouts and physical parameters the policy has not seen |
| 6 | Robustness / Stress Test | the Phase-4 policy tested under observation noise, action noise, and physical perturbations at evaluation time |

The core decision-making is learned with reinforcement learning. Traditional control is only used as low-level plumbing, through PyBullet's `POSITION_CONTROL` joint servo, nothing more.

Robot: a **KUKA iiwa** arm with 7 degrees of freedom. The extra joint helps it avoid obstacles in Phase 3, and it has no gripper geometry to get in the way of the hole plate during insertion. It comes bundled with `pybullet_data`, so no external assets are needed.

---

## 2. What the scaffold could not do, and the rebuild

The assignment scaffold runs, but as shipped it cannot do what the problem statement asks for. I rebuilt the environment (`manipularl/`) to fix the gaps below.

| Scaffold gap | Consequence | Rebuild |
|---|---|---|
| Stage 4's target is a point at `z = 0`. Success needs the peg to be 3 cm below that point, but **no hole geometry exists at all** | peg-in-hole is physically impossible | I added a **bore socket** on the table, made from a ring of flat box segments (8 by default, so it looks like an octagon). PyBullet cannot cut a true circular hole into a solid shape, so this is an approximation. `cfg.hole_segments` can raise the count for a rounder shape, though that is a real change to the physics and needs its own retrain, which I have not done. Depth is measured from the top of the plate. Success now needs several things at once (grasp, no collision, depth of at least 3 cm, xy position within tolerance, tilt within tolerance), and all of them must **hold for 10 consecutive control steps** |
| The 32-dimensional observation has **no obstacle information** in it | Phase 3 avoidance, and generalizing to unseen layouts, cannot be learned | the observation now includes the **6 nearest obstacles** (their position relative to the arm, their size, a type label, and a flag saying if the slot is used), sorted by distance and padded with zeros when there are fewer than 6. This part of the observation always stays the same size |
| Stage 2 success requires the peg to **still be grasped** | the policy can never successfully "release" the peg; a released peg also simply falls, so a mid-air "settle" requirement is impossible | pick-and-place success now means the peg was brought within tolerance of the target while held, **and then released** within a short window afterward |
| The obstacle random number generator was never reseeded on `reset()` | there was no separation between training and evaluation layouts | I added an `EpisodeSampler`. Training uses seeds `[0, 10^6)`, evaluation uses `[10^6, 2*10^6)`, so "unseen" is now guaranteed by the code, not just something I hoped for |
| there was no domain randomization, no noise, no metrics, and no evaluation harness | Phases 5 and 6 could not be evaluated as described | dynamics randomization for Phase 5, perturbation wrappers for Phase 6 (evaluation-time only), and an `evaluate.py` script that reports metrics per condition |
| the scaffold reset by reloading 3 URDFs every episode, at a single 240 Hz tick | poor throughput and control rate | a **persistent scene**: built once, bodies teleported instead of reloaded, measured **25 to 30 times faster resets**. Control now runs at **20 Hz over 12 physics substeps** |
| the action was a raw joint velocity command | jittery, hard to align finely | switched to **delta joint position** control (`q_target = clip(q + a * 0.05, q_lo, q_hi)`), fully joint-space and smooth |

`manipularl/obstacles.py` is the scaffold's own `obstacle_generator.py`, unchanged.

---

## 3. Environment

### 3.1 Robot model and control

The robot has 7 revolute arm joints. Grasping works like a snap: once the gripper signal closes within a trigger distance, a fixed `p.createConstraint` locks the wrist link to the peg. I kept this from the scaffold since it is a stable grasp model for early-stage RL, not a finger-force simulation. Control runs at 20 Hz, and each `step()` advances 12 substeps of the underlying 240 Hz physics.

### 3.2 Action space: `Box(-1, 1, (8,))`

7 delta joint position commands (each scaled by 0.05 radians per step), plus 1 gripper signal (above 0.5 closes, at or below 0.5 opens).

Why this design: the problem statement requires the decision-making to be learned through RL, with normal planning or control used only as supporting infrastructure. Small joint-space steps tracked by a low-level servo, with no IK solver or motion planner involved, is the safest way to meet that rule. It is also smoother and learns faster than the scaffold's raw velocity command.

### 3.3 Observation space: `Box(-inf, inf, (123,))`, stacked 3 frames to 369

Joint position (7) and velocity (7), end-effector position (3) and a 6-value rotation (6) plus its linear (3) and angular (3) velocity, gripper command (1) and grasp flag (1), the peg's position (3) and rotation (6), its position relative to the end-effector (3) and velocity (3), the goal position (3) and rotation (6) and the vector from peg to goal (3), insertion depth as a fraction (1), wrist force and torque (6), the 6 nearest obstacles (9 values each), and a one-hot phase indicator (4).

Design choices:
- I use a **6-value rotation representation** (the first two columns of the rotation matrix) instead of quaternions. It is continuous, it avoids the double-cover problem, and it is easier for a network to learn.
- **Wrist force and torque**: the scaffold turns these sensors on but never reads them, even though contact force is one of the most useful signals during alignment and insertion.
- I use a **fixed-size, sorted list of the nearest obstacles**. This is cheaper than ray casts or a voxel grid, and sorting keeps obstacles from swapping slots between steps. A **3-frame stack** lets the policy filter out noise and figure out contact state on its own (useful in Phase 6), without needing a recurrent network.
- **One observation space for every phase.** Fields that do not apply, for example the peg in Phase 1, are simply zero. The interface is only ever extended, never redefined, as the problem statement requires.

### 3.4 Reward design (the core RL-design piece the problem statement asks about)

I use potential-based shaping: `F(s, s') = k * (gamma * Phi(s') - Phi(s))`. A well-known result (Ng, Harada & Russell, 1999) proves this form never changes what the best policy is, no matter what `Phi` is. So the agent cannot farm the dense reward. It can only earn more by making real progress.

`Phi` is the negative distance to the **current sub-goal**, which switches automatically through one-time bonuses as the task moves forward:

```
reach peg --(+grasp bonus)--> carry to goal --(+align bonus)--> insert (depth term)
```

For peg-in-hole, while the peg is held, `Phi` combines horizontal distance to the bore, vertical gap above the plate, peg tilt, and insertion depth (negative sign).

A few smaller terms sit on top, all small compared to the main task reward. There is an action/jerk penalty for smooth motion. There is a small time penalty, which also cancels a side effect of discounted shaping: without it, the policy could get rewarded just for hovering near the goal without ever arriving. From Phase 3 onward there is a per-step collision penalty. And there is a large bonus for actually reaching the goal tolerance, so the policy commits instead of camping just outside it.

**Success conditions**
- Reach: end-effector within 3 cm of the goal.
- Pick & Place: peg within 3 cm of the target while held, **then released** within 15 steps (a mid-air target cannot be "settled" on). From Phase 3 onward, any run that collided is `success = False` even if it reached the goal.
- Peg-in-Hole: grasped, no forbidden collision, depth at least 3 cm, horizontal error under 1 cm, tilt under 10 degrees, all held for 10 consecutive steps so a single lucky frame does not count.

### 3.5 Randomization and the train/eval split

`EpisodeSampler` builds one `EpisodeConfig` for each episode, using a PCG64 random stream keyed by the episode index. Training uses indices in `[0, 10^6)`, evaluation uses `[10^6, 2*10^6)`, so a policy is never evaluated on a layout it could have trained on. Randomized: peg, hole and goal positions, peg yaw and tilt, starting joint configuration, and obstacle count/placement. Phase 5 additionally randomizes peg mass, lateral friction, joint damping, restitution, and bore clearance.

Phase 6 perturbations (`wrappers.py`, evaluation only): additive Gaussian observation noise, additive Gaussian action noise (re-clipped), and random impulses on the peg.

---

## 4. Training method

I train the phases in sequence and warm-start each one from the last. Since every phase shares the same observation and action interface, I can reuse the weights from one phase to the next exactly, with nothing to reshape or throw away.

```
P1 --> P2 --> P3 --> P4 --> P5      (each phase warm-starts from the previous
                                     model.zip and carries over its
                                     VecNormalize statistics)
```

The wrapper stack is: `SubprocVecEnv(16)`, then `VecMonitor`, then `VecFrameStack(3)`, then `VecNormalize` (observations only).

### 4.1 Algorithm: intended versus actually used

**TQC** (truncated-quantile critics, from sb3-contrib) was meant to be the main algorithm. It tends to do well on contact-rich PyBullet manipulation, and its entropy tuning helps it discover rare events like a successful grasp. On my machine (a 22-core CPU) it only manages **120 to 330 environment steps per second**, and throughput drops if I use more than 8-10 torch threads. I benchmarked CPU against GPU torch directly: with a network this small, about 100k parameters, PyBullet's own physics stepping is the real bottleneck either way, so GPU gave no meaningful speedup (about 1450-1490 steps/s on CPU versus 1500-1550 on GPU). I used CPU (`.venv`) for phases 1-4's main training and GPU (`.venv_gpu_test`) for the later Phase 4 insertion investigation (R1-R6), since the choice barely affects throughput either way. TQC **solved Phase 1** (0.935, versus PPO's 1.00), but as the main algorithm for Phase 2, with a 2M-step budget, it **plateaued at 0.01**. Being off-policy and sample-efficient was not enough by itself to crack the gated grasp problem in the time I had.

**PPO** runs **5 to 10x faster** (roughly 1,200-2,000 steps/s with 16 parallel envs) and carries every phase in this project. Hyperparameters: `n_steps` 512, batch size 2048, 10 epochs/update, `gamma` 0.99, GAE `lambda` 0.95, clip range 0.2, entropy coefficient 0.004-0.01, learning rate 2e-4 to 3e-4, `[256, 256]` network. Configs for both algorithms are in `configs/` (`ppo_phase*.yaml` and `phase*.yaml`).

### 4.2 What actually made Phase 2 work

Phase 2 did not train successfully under the plan's original sequential warm-start. Full story in `EXPERIMENTS.md`; the essentials:

1. **Diagnose instead of guess** (`scripts/diagnose_phase2.py`). This script breaks an episode into approach, grasp, carry, and release, and tests each part on its own with an "always start already grasping" mode. Starting cold, the arm **never even approached the peg** (median end-effector distance 27 cm, only 7.5% got within the 9 cm trigger). Starting pre-grasped, it only **released correctly 38%** of the time.

2. **The Phase 1 warm-start was actively hurting the policy.** Phase 1's goal is "reach `goal_pos`", but in Phase 2 that same point is the mid-air place target, not the peg. So the warm-started policy kept driving toward the wrong point, and PPO's KL-limited updates could not unlearn that fast enough. I confirmed this directly: applying every fix below while still keeping the warm-start still collapsed to 0 success once the curriculum finished.

3. **The fix** (Phase 2 only, the success check itself was left alone): train **from scratch**, with no warm-start, so the same `-‖EE - peg‖` shaping that solves Phase 1 pulls the hand toward the peg with nothing fighting it. Reward shaping in `manipularl/rewards.py` (active only for `pick_place`): widened the grasp-approach kernel from 12 cm to 35 cm, raised `_GRASP_SHAPE_W` 0.04 to 0.08, added an in-range bonus and a one-time carry-progress bonus, raised `_GRASP_BONUS` 8 to 12, and fixed the release behavior so the per-step holding bonus stops once the target is reached instead of paying forever. I also added an **annealed grasp curriculum**: the fraction of episodes that start pre-grasped shrinks from 0.7 to 0 over the first 70% of training. This helps early on and forces cold-start grasping later.

   Result: **0.995** eval success, up from 0.00. Post-fix diagnostic: 99.3% cold-start success, 1.9 cm average end-effector-to-peg distance, 100% grasp rate, 37 steps/episode average.

I also built a demonstration and behavior-cloning backstop (`scripts/{scripted_expert,collect_demos,bc_pretrain}.py`: a scripted IK expert at about 80% success, 28,846 demo transitions, BC trained to a validation MSE of 0.038), but I did not end up needing it, since from-scratch RL solved Phase 2 first. BC alone scored 0% at evaluation, a classic case of covariate shift. That is exactly why the usual recipe is BC followed by RL fine-tuning, never BC by itself.

### 4.3 Per-phase budgets and warm-starting

| Phase | Steps | Warm-start | Notes |
|---|---|---|---|
| 1 Reaching | 1.5 M (PPO) / 0.6 M (TQC) | none | |
| 2 Pick & Place | 3.0 M | **none** (from scratch) | `configs/ppo_phase2_scratch.yaml` |
| 3 Obstacle-Aware | 2.5 M plus a 40k-step finetune | Phase 2 | grasp-curriculum anneal on. Adopted checkpoint (`runs/phase3_ppo_best/`) is the 40k-step point of an arm-jitter-penalty finetune, not the raw 2.5M baseline. `EXPERIMENTS.md` E13, E48 |
| 4 Peg-in-Hole | 4.0 M or more, across 61 experiments | Phase 3 | `EXPERIMENTS.md` E1-E61. A diagnosed environment ceiling (E55, E56), not a training-budget shortfall |
| 5/6 Generalization and Robustness | 0 (evaluation only) | Phase 4 (`runs/phase4_full_dream`, E51) | see sections 5.5-5.6. The problem statement's own wording, "re-evaluate the trained policy," does not require a converged Phase 4 checkpoint |

Phases 3 through 5 are driven by `scripts/chain_p3p4p5.sh`.

---

## 5. Results

All evaluation numbers below use 200 independently randomized episodes from the **evaluation** seed range (`[10^6, 2*10^6)`, unseen layouts), deterministic policy. See `evaluate.py` and `results/`.

### 5.1 What was trained (200 evaluation episodes each, deterministic policy)

| Phase | algo | steps | success | collision | mean steps | Outcome |
|---|---|---|---|---|---|---|
| **1 Reaching** | PPO | 1.5 M | **1.00** | 0.00 | 11 | **Solved.** `eval_phase1.csv` |
| 1 Reaching | TQC | 0.6 M | 0.935 | 0.00 | 24 | solved, off-policy comparison point. `eval_phase1_tqc.csv` |
| **2 Pick & Place** | PPO (scratch) | 3.0 M | **0.995** | 0.00 | 37 | **Solved.** Up from 0.00, see §4.2. `eval_phase2.csv` |
| **3 Obstacle-Aware** | PPO | 2.5 M plus 40k finetune | **0.86** | 0.08 | 77 | **Partial, but strong.** An arm-jitter penalty (built for Phase 4's Task 1, reused here) finetuned the 2.5M baseline; a checkpoint sweep found the 40k-step point ahead of both the baseline (0.765/0.195) and the rest of that run (flat 0.65-0.70/0.21-0.30). `runs/phase3_ppo_best/`, `EXPERIMENTS.md` E13/E48 |
| **4 Peg-in-Hole** | PPO | 4.0 M or more, 61 experiments | **0.015** | 0.57 | 226 | **Honest partial. Did not converge.** A privileged scripted oracle with perfect state also scores only ~2% (`EXPERIMENTS.md` E55), showing the ~1.5-2% ceiling comes from the environment (a persistent 20-40° grasp tilt too wide for the bore's 3mm clearance), not the algorithm. Four alternative approaches (environment-fix ablation, fixed-environment retrain, three demo sources, and a combination) all land on the same ceiling. E55-E61. `eval_phase4.csv` |
| 5 Dynamics Gen. | | | | | | See §5.5 |
| 6 Robustness | | | | | | See §5.6 |

Videos: `media/phase1_reaching.mp4` (5/5 successful), `media/phase2_pickplace.mp4` (5/5 successful), `media/phase3_obstacle_aware.mp4` (5/5 successful), `media/phase4_peg_in_hole.mp4` (5 episodes rolled honestly, 0 successful, matching the ~1.5-2% full-task ceiling documented above, plus one successful insertion appended at the end, found on attempt 19 of a separate `--successes-only` search, so both the typical outcome and a real success are visible). Chart: `plots/per_phase_summary.png`.

More ablations are logged in `EXPERIMENTS.md` and `results/eval_phase*_*.csv`. The TQC "gamble" for Phase 2 plateaued at 0.01. Keeping the Phase 1 warm-start while applying every reward fix still collapsed to 0.00, which isolates the warm-start as the actual cause. BC alone on scripted-expert demos scored 0.00 (covariate shift). A "soft warm-start" on Phase 3 (critic re-init, higher entropy, policy kept) reached 0.475, no better than the 0.515 baseline measured at the time (before later work moved Phase 3 to 0.86). The Phase 2 to 3 transfer is already mostly positive, so softening it did not help.

### 5.2 Honest account of where it fell short (as the problem statement's section 3.2 asks for)

- **Phases 1 and 2 are solved** on unseen layouts (1.00, 0.995). Phase 2 was the hard case: a rare gated event (the grasp) buried inside a long carry. The fix (from-scratch training, reward shaping, annealed curriculum, see §4.2) is a clean, repeatable result. The post-fix diagnostic shows a correct approach-grasp-carry-release sequence on 99% of cold-start episodes.

- **Phase 3 reaches 0.86/0.08** (`runs/phase3_ppo_best/`, the checkpoint every later phase warm-starts from), up from a 0.515/0.28 baseline. Two levers moved this number. First, dense obstacle-repulsion shaping (`_REPULSE_W`/`_REPULSE_MARGIN`, a per-step "steer away" gradient inside 12 cm, replacing what used to be only a terminal collision penalty), combined with a 0-1 to 2-5 obstacle-count curriculum, brought collision down to 0.19 in one pass (E13). Second, an arm-jitter penalty, reused from Phase 4's Task 1, was finetuned further to 0.86/0.08 through a checkpoint sweep (E48). Remaining failures are still mostly collisions and timeouts on cluttered layouts, just fewer of them now.

- **Phase 4 does not converge, and I investigated until I found the real cause instead of just calling it a budget limit.** A privileged scripted oracle (exact state, hand-written IK, none of a learner's exploration or credit-assignment difficulty) still only scores about 1.5-2%, the same ceiling every trained policy plateaus at (E55). That shows the learner was never the bottleneck. Tracing individual episodes found the actual cause: a grasped peg is often held 20-40° off vertical for the whole episode, far wider than the bore's 3mm clearance allows, which blocks entry no matter how good the horizontal alignment is (E56). I tried everything an earlier draft proposed as a fix: an environment-fix ablation (bore chamfer, more compliant grasp, slower actuator, joint-velocity cap, E56), a fixed-environment retrain with a stronger tilt-weighted reward (E57), three demonstration sources (a compliant/impedance controller with depth-stall search, E58; MPC over a learned dynamics model, E59; hindsight relabeling, E60), and a combination of two of these (E61). All five land on the same ~1.5-2% ceiling. At the time, I considered this a diagnosed, well-evidenced environment limitation. **This has since been partly updated, see §5.7**: the tilt was actually caused by a fixable bug in the code, not an unavoidable physical limit. §5.7 explains what I found, and what it does and does not change.

- **Not attempted:** a full off-policy (TQC/SAC) curriculum across all phases. The measured 120-330 steps/s made a full 5-phase run infeasible in the time available.

### 5.3 Environment throughput (`results/benchmark.txt`)

| Metric | This env | Scaffold-style |
|---|---|---|
| single-env `step()` | about 930 steps/s | |
| SubprocVecEnv, 8 workers, `step()` | about 2,700 steps/s | |
| `reset()` | about 770/s (persistent scene) | about 30/s (`resetSimulation` + 3 URDF reloads) |

Roughly **26x faster resets**, from building the scene once and teleporting bodies instead of reloading.

### 5.4 Algorithm note (TQC)

TQC configs: `configs/phase{1..5}.yaml`. Each gradient step takes about 13 ms for a `[128, 128]` network with 15-quantile critics, giving about **120 to 330 steps/s** (best with 10-12 torch threads). It trained Phase 1 to 0.935 in 0.6M steps but plateaued at 0.01 on the Phase 2 budget. That is why PPO (about 1,200-2,000 steps/s) carried every phase instead. I did not attempt a full off-policy curriculum.

### 5.5 Phase 5: Standard and Dynamics Generalization

This is not a training exercise. The Phase 5 environment support (`manipularl/configs.py`'s `PhaseConfig`, dynamics ranges, wider obstacle count) and `evaluate.py --phase 5` have existed since early in the project, but nobody had actually run them on any checkpoint before now. Earlier drafts said the blocker was "Phase 5 warm-starts from Phase 4, which never converged." That is true if you want to *train* a Phase-5-specific policy, but the problem statement only asks to "re-evaluate the trained policy" under new conditions, and that needs no Phase 5 training at all.

I evaluated **E51** (`runs/phase4_full_dream`, best full-task Phase 4 checkpoint) under `--phase 4` (in-distribution) and `--phase 5`: peg mass 0.03-0.12 kg, lateral friction 0.4-1.2, joint damping 0.5-3.0, restitution 0.0-0.2, bore clearance 0.002-0.006 m (trained values: 0.05 kg, 0.6, none, 0.003 m), and obstacle count 2-6 instead of 2-5.

| Condition | success | collision | insert depth | mean steps | final pos. error |
|---|---|---|---|---|---|
| in-distribution | 0.015 | 0.425 | 0.00136 m | 269 | 0.393 m |
| **unseen dynamics** | 0.015 | **0.555** | 0.0009 m | 228 | 0.359 m |

**Success rate stays flat**, matching the ~1.5% ceiling measured everywhere else, including the scripted oracle (§5.2, E55). The diagnosed root cause (E56) is a kinematic grasp-orientation problem, not a sensitivity to dynamics. **Collision rate is where the generalization gap actually shows up**: it rises 31% relative (0.425 to 0.555) under a wider obstacle count and shifted contact dynamics. That is exactly the kind of degradation the problem statement's Phase 5 section asks to make visible. The policy's willingness to attempt insertion generalizes fine, but its collision avoidance does not. Insertion depth falling back to the historic 0.0009 m wall under unseen (tighter, down to 2mm) bore clearance is expected: E51's own best of 0.00136 m was measured specifically at the trained 3mm clearance.

### 5.6 Phase 6: Robustness / Stress Test

This uses the same checkpoint as §5.5 (E51), running `evaluate.py --phase 4 --noise-sweep` across sigma 0.000/0.005/0.010/0.020/0.040/0.080. At each level, as Phase 6 defines, I apply additive Gaussian observation noise, additive Gaussian action noise, and a `sigma*300` N peg impulse every 25 steps, all at the same time. I found and fixed one real bug in the test harness first: the peg-perturbation force had a floor of `max(2.0, ...)`, so even `sigma=0.000` applied a 2N impulse, meaning there was never a clean baseline row. After the fix, `sigma=0.000` below exactly reproduces §5.5's in-distribution number, which confirms it worked.

| sigma | success | collision | insert depth | tilt (best-case) |
|---|---|---|---|---|
| 0.000 (clean baseline) | 0.015 | 0.425 | **0.00136 m** | 0.0105 deg |
| 0.005 | 0.015 | 0.545 | 0.0009 m | 0.105 deg |
| 0.010 | 0.015 | 0.555 | 0.0009 m | 0.106 deg |
| 0.020 | 0.015 | 0.625 | 0.0009 m | 0.107 deg |
| 0.040 | 0.015 | 0.685 | 0.0009 m | 0.109 deg |
| 0.080 | 0.015 | 0.690 | 0.0009 m | 0.107 deg |

**This is a genuinely mixed result, not simply "graceful" or "brittle."** Success rate stays flat at every noise level, but that is a floor effect, not real robustness: 0.015 is already close to the ~1.5-2% environment ceiling, so there is very little room left for it to get worse. Collision rate is where the real signal is, and it degrades gracefully: a smooth 62% relative rise from baseline to the highest noise level, leveling off between the top two. The rare, fine-precision behaviors are brittle, not graceful. E51's own best achievement, breaking the historic 0.0009 m insertion-depth wall that held across nearly every other Phase 4 run (E10, E16, E17, E28, E30, E52, E53), disappears at the **smallest** noise level tested (sigma=0.005) and never comes back at any higher level. Tilt control shows the same on/off pattern. This step-function failure suggests these specific behaviors are narrow, fragile results of one training run, not a skill the policy robustly learned.

**Overall:** common, frequently-practiced behaviors (attempting insertion, partially avoiding obstacles) degrade gracefully. The one rare, hard-won behavior (real insertion depth) does not survive any tested noise level. This matches §5.2's diagnosis: the policy's core skill ceiling was already the binding constraint, and noise mostly erodes whatever thin margin existed above it.

---

### 5.7 Addendum: the Phase 4 grasp-weld bug, and what training against the fix showed

Additional investigation after the results above were finalized. Full entry-by-entry account in `EXPERIMENTS.md` (E79, E80); this section is the summary. The headline numbers in §5.1-5.2 are **unchanged**.

**The bug.** `_form_grasp()` in `manipularl/env.py` welds the peg to the gripper with a PyBullet `JOINT_FIXED` constraint, and both of that constraint's frame orientations were left at the library default (identity rotation). This has a consequence nobody had noticed before: with identity frame orientations, `JOINT_FIXED` forces the child body's orientation to exactly match the parent's, **no matter what the position offset is**. So on every real grasp, the peg's orientation was silently overwritten to match whatever orientation the gripper happened to have. §5.2's "20-40° off vertical" finding was real and correctly measured, but the cause was this one piece of code, not the bore's geometry.

I fixed it by welding the peg using the real relative transform between the gripper and the peg at grasp time, computed from their actual poses instead of assumed. I also added a geometric check so the fix cannot reopen a separate, legitimate reason an offset existed in the first place (a zero offset would put the flange inside solid bore material at real insertion depth). I confirmed Phases 1-3 are unaffected by checking constraint parameters directly and deterministically, with no simulation stepping involved, not just by re-running evaluations. All 22 tests pass (`tests/test_env.py`, four of them new).

**Verified with zero additional training**, using the same scripted oracle as §5.2 (E55). With active reorientation during descent, the bore entry rate rose from a historical ceiling of about 18% to **62%**, the oracle's overall success **tripled** (2% to 6%), and measured grasp tilt collapsed from a ~21° median down to ~2.5°. Without active reorientation, the result barely moved. That means the fix is necessary but not sufficient by itself, and the earlier finding about KUKA's redundant 7th joint (from a UR5 comparison) is not overturned, just no longer the whole story.

**Training against the fix in four from-scratch configurations all stalled on the full task.** I tried a grasp curriculum (the same mechanism that solved Phase 2's identical grasp-collapse problem, see §4.2), a widened geometric admissibility check, the obstacle-count curriculum that helped Phase 3 (§5.1), and removing obstacles entirely. None of them acquired cold-start grasping: the ever-grasped rate stayed at 0.00-0.005 across the full 5M-step budget, every time. Tracing episodes explained why: with obstacles present, 85% collide before the arm even gets close enough to grasp. With obstacles removed, a controlled pre-grasped-versus-cold-start comparison at a fair checkpoint gave **identical 1.7% success either way**. A completely free grasp did not help at all. Grasp acquisition and obstacle avoidance are both real, partly-diagnosed problems, but neither explains the overall floor: **insertion precision itself is the actual bottleneck**, matching the oracle's own 2-6% ceiling.

**Isolating the insertion step directly produced a genuinely new result.** A policy trained from scratch on just the insertion sub-task (peg pre-grasped, no obstacles, using the existing `insert_only`/`insert_start_depth_curriculum` machinery) plateaus at **7.3% mean success from 750k steps onward** (std 0.024 across seven checkpoints, matching pure sampling noise at this episode count, so this is a real, stable plateau). This is roughly double the previous best on this task (4.0%, an earlier attempt that used obstacles and a warm-start). It is still only a sub-task result (pre-grasped, no obstacles, 80-step episode cap), and it does **not** move the full-task success rate, which stayed at the historical 1.5-2% floor throughout every configuration above.

**Net effect on this report: no change to the numbers, a real change to the explanation.** Phases 1-3 and the adopted Phase 4 checkpoint in §5.1-5.2 are unchanged. What changes is part of §5.2's explanation: the ~1.5-2% ceiling is not fully an unfixable physical constraint. One real bug was found and fixed inside it, but fixing it alone did not produce a working full-task policy in the time available. The identified next step, not yet attempted, is to train a reach-grasp-carry-align policy with obstacles (an existing config already supports this) under the corrected physics, and combine it with the insertion policy above through the existing Task 1 to Task 2 handoff (`scripts/evaluate_composed.py`), instead of asking one policy to learn the whole chain from scratch at once.

---

## 6. Reproducibility

```bash
uv venv --python 3.10 .venv && uv pip install -r requirements.txt
PYTHONPATH="" .venv/bin/python -m pytest -q tests/test_env.py          # 18 checks
PYTHONPATH=.  .venv/bin/python scripts/benchmark_env.py                # throughput table

# Phase 1 (solved):
PYTHONPATH=. .venv/bin/python train.py --config configs/ppo_phase1.yaml --out-dir runs/phase1_ppo

# Phase 2 (solved), from scratch, NOT warm-started (see section 4.2):
PYTHONPATH=. .venv/bin/python train.py --config configs/ppo_phase2_scratch.yaml --out-dir runs/phase2_ppo
PYTHONPATH=. .venv/bin/python scripts/diagnose_phase2.py --run runs/phase2_ppo --mode coldstart
PYTHONPATH=. .venv/bin/python scripts/diagnose_phase2.py --run runs/phase2_ppo --mode pregrasped

# Phase 3 (adopted checkpoint, runs/phase3_ppo_best/ = 0.86/0.08):
# scripts/chain_p3p4p5.sh's one-shot Phase 3-5 chain was superseded early.
# The adopted checkpoint is the 40k-step point of an arm-jitter-penalty
# finetune off the 2.5M-step baseline (EXPERIMENTS.md E13, E19-E48). It is
# not reproducible from a single script call; see EXPERIMENTS.md for the
# exact iterative recipe. To re-verify the adopted checkpoint directly:
PYTHONPATH=. .venv/bin/python evaluate.py --run runs/phase3_ppo_best --phase 3 --episodes 200 \
    --out results/eval_phase3_ppo_best_verify.csv

# Phase 4 (honest partial, 0.015/0.57, see EXPERIMENTS.md E1-E61):
PYTHONPATH=. .venv/bin/python evaluate.py --run runs/phase4_full_dream --phase 4 --episodes 200

# Phase 5/6 (generalization and robustness, evaluated on the Phase 4
# checkpoint above, see sections 5.5 and 5.6):
PYTHONPATH=. .venv/bin/python evaluate.py --run runs/phase4_full_dream --phase 5 --episodes 200
PYTHONPATH=. .venv/bin/python evaluate.py --run runs/phase4_full_dream --phase 4 --noise-sweep

# Eval / plots / video (any run):
PYTHONPATH=. .venv/bin/python evaluate.py --run runs/phase2_ppo --phase 2 --episodes 200
PYTHONPATH=. .venv/bin/python scripts/plot_eval.py --csv "results/eval_phase*.csv" --out plots/
PYTHONPATH=. .venv/bin/python scripts/record_rollout.py --run runs/phase2_ppo --phase 2 \
    --algo ppo --episodes 5 --out media/phase2_pickplace.mp4
```

`EXPERIMENTS.md` is the full run-by-run log: every attempt, its config, its result, and where its files live. Every randomization parameter used for training or evaluation is in `manipularl/configs.py` and `manipularl/randomization.py`.

---

## 7. References

- Ng, Harada & Russell, *Policy invariance under reward transformations*, ICML 1999 (potential-based shaping).
- Kuznetsov et al., *Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics* (TQC), ICML 2020.
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Haarnoja et al., *Soft Actor-Critic*, ICML 2018.
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019 (6-D rotation).
- Raffin et al., *Stable-Baselines3*, JMLR 2021; sb3-contrib.
- Coumans & Bai, *PyBullet*, 2016-2021.
- Stanford CS234; `stable-baselines3` docs; `benelot/pybullet-gym`;
  `leesweqq/ur5_reinforcement_learning_grasp_object` (referenced by the problem statement).
