# Asset provenance

`ur5_robotiq_85.urdf` and the `meshes/` folder in this directory are the UR5 +
Robotiq 2F-85 gripper model used to give the `ur5` robot option in
`manipularl_env.py` a look matching:

- https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object
  (the PyBullet + Gymnasium UR5 pick-and-place env these ManipulaRL stages
  were asked to visually match)

which in turn credits the original rig as sourced from:

- https://github.com/ElectronicElephant/pybullet_ur5_robotiq

Both are MIT-licensed. `LICENSE` in this directory is the MIT license text
as published in the `leesweqq/ur5_reinforcement_learning_grasp_object`
repository (Copyright (c) 2025 kai), included here for attribution.

No code from those repositories is used — only the URDF/mesh model files.
The RL logic, reward shaping, obstacle generation, and stage structure in
this project are original and unrelated to those repos' Gym env or
training code.
