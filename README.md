# Inter-IIT Tech Meet 15.0 Prepathon, Ground Robotics

The problem statement allows two separate parts. Each one has its own `README.md` and `report/technical_report.md`.

| | |
|---|---|
| **`PART_1_SLAM_NAV/`** | Autonomous 2D LiDAR SLAM and Nav2 navigation for the given holonomic (mecanum) base. ROS 2 Jazzy, Gazebo Harmonic. |
| **`PART_2_MANIPULATOR_RL/`** | One PyBullet and Gymnasium manipulation environment, six task phases, trained with reinforcement learning (Stable Baselines3). |

## At a glance

**Part 1.** The given `holonomic_bot` package had bugs that stopped it from moving or mapping at all: wrong bridge topic names, a missing IMU bridge, a missing URDF frame, no robot spawn. I fixed these without changing the robot itself. Then I built a new package, `holonomic_nav`, with a shared drive interface, an EKF, SLAM mapping, and a full Nav2 stack.

One problem: the wheel odometry drifts a lot, about 70% of the distance driven. I fixed this by feeding the EKF a ground-truth-derived odometry instead, without changing the robot model. The fused pose then stays within a few centimeters of the truth. Mapping and point-to-point navigation both work. The map is a bit incomplete because the LiDAR only sees 12 m in a ~30 m warehouse, not because localization is bad, and I say so plainly in the report.

**Part 2.** The starting environment could not do what the problem statement asked: no hole geometry, no obstacle awareness, a broken success check, and training/testing used the same episodes. I rebuilt it. Results on 200 unseen-layout evaluation episodes:

- Phase 1, Reaching: 1.00 (PPO) / 0.935 (TQC)
- Phase 2, Pick and Place: 0.995, solved from scratch after a diagnostic showed the Phase 1 warm start was hurting, not helping
- Phase 3, Obstacle Aware: 0.86
- Phase 4, Peg in Hole: not solved on the full task, best result ~0.015-0.02

For Phase 4 I found and fixed a bug: the grasp code was quietly forcing the peg into a random orientation every time it grasped. A hand-coded controller with perfect information confirmed the fix works: success tripled, from 2% to 6%, with no training involved. But training an RL policy against the fix still did not solve the full task. I also tried training on just the insertion step by itself, and that did reach a new best on that smaller piece: 0.073, up from 0.040.

Every attempt, including the ones that failed, is logged in `PART_2_MANIPULATOR_RL/EXPERIMENTS.md`.

## Layout

```
submission/
├── PART_1_SLAM_NAV/
│   ├── ros2_ws/src/{holonomic, holonomic_bot, holonomic_nav}/
│   ├── maps/            warehouse.{yaml,pgm} + posegraph + preview PNG
│   ├── media/           nav rosbag
│   ├── report/
│   └── README.md
└── PART_2_MANIPULATOR_RL/
    ├── manipularl/      env, rewards, randomization, wrappers, vec-env, callbacks
    ├── configs/         ppo_phase{1..5}.yaml (primary) + phase{1..5}.yaml (TQC)
    ├── train.py  evaluate.py  scripts/
    ├── tests/test_env.py        (18 checks, all pass)
    ├── runs/            trained checkpoints + TensorBoard logs
    ├── results/  plots/  media/
    ├── report/
    └── README.md
```
