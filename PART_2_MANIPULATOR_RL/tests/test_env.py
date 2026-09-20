"""
test_env.py -- correctness checks for the rebuilt ManipulaRLEnv.

Run before committing training time to any phase:

    .venv/bin/python -m pytest -q tests/test_env.py
    # or plain:
    .venv/bin/python tests/test_env.py

Layers:
  1. Gymnasium API compliance   (gymnasium.utils.env_checker.check_env)
  2. Random-action rollouts      (no crash, no NaN/Inf, shapes hold, all phases)
  3. Determinism                 (same seed -> identical trajectory)
  4. Train/eval seed disjointness (a policy can't be evaluated on a train layout)
  5. Hole solvability            (a scripted straight-down insert reaches success)
"""

import numpy as np
import pytest

from manipularl.env import ManipulaRLEnv
from manipularl.configs import PHASES, EVAL_SEED_LO, TRAIN_SEED_HI


ALL_PHASES = sorted(PHASES)


@pytest.mark.parametrize("phase", ALL_PHASES)
def test_gymnasium_api(phase):
    from gymnasium.utils.env_checker import check_env

    env = ManipulaRLEnv(phase=phase, split="train")
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.parametrize("phase", ALL_PHASES)
def test_random_rollout(phase):
    env = ManipulaRLEnv(phase=phase, split="train")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), (phase, obs.shape)
    for ep in range(2):
        obs, _ = env.reset(seed=ep)
        for t in range(60):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            assert np.isfinite(obs).all(), (phase, t, "obs")
            assert np.isfinite(r), (phase, t, "reward")
            assert isinstance(term, bool) and isinstance(trunc, bool)
            if term or trunc:
                break
    env.close()


@pytest.mark.parametrize("phase", [1, 4])
def test_determinism(phase):
    def traj(seed):
        env = ManipulaRLEnv(phase=phase, split="train")
        o, _ = env.reset(seed=seed)
        rng = np.random.default_rng(123)
        out = [o.copy()]
        for _ in range(40):
            o, r, term, trunc, _ = env.step(rng.uniform(-1, 1, size=env.action_space.shape))
            out.append(o.copy())
            if term or trunc:
                break
        env.close()
        return np.array(out)

    a = traj(7)
    b = traj(7)
    assert np.allclose(a, b, atol=1e-6), "same seed produced different trajectories"


def test_train_eval_seed_split():
    from manipularl.randomization import EpisodeSampler
    from manipularl.configs import get_phase

    cfg = get_phase(4)
    tr = EpisodeSampler(cfg, "train")
    ev = EpisodeSampler(cfg, "eval")
    train_seeds = {tr.sample(i).seed for i in range(500)}
    eval_seeds = {ev.sample(i).seed for i in range(500)}
    assert train_seeds.isdisjoint(eval_seeds)
    assert max(train_seeds) < TRAIN_SEED_HI <= min(eval_seeds) or min(eval_seeds) >= EVAL_SEED_LO


@pytest.mark.parametrize("phase", [3, 4])
def test_obstacle_count_matches_scene(phase):
    env = ManipulaRLEnv(phase=phase, split="train")
    for i in range(10):
        env.reset(options={"episode_index": i})
        want = len(env._cur.obstacles)
        assert env._obstacle_active == want
        assert PHASES[phase].obstacle_min <= want <= PHASES[phase].obstacle_max
    env.close()


def test_grasp_weld_preserves_transform():
    """_form_grasp's rigid (peg_in_hole, non-align_only) branch must weld
    preserving whatever relative pose exists at formation time -- no
    teleport, no forced re-orientation. This is the fix for the bug where
    the old hardcoded-offset weld left BOTH frame orientations at identity,
    which forces peg_orn == flange_orn regardless of childFramePosition (a
    JOINT_FIXED property, not specific to a nonzero offset) -- the root
    cause traced to E56's measured 20-40deg grasp tilt and the E31
    GRASP<->DROP flicker. See _form_grasp's docstring in manipularl/env.py.

    Deliberately zero-simulation-stepping: verified two ways, both exact.
    1. Algebraic: peg_pose ∘ rel == flange_pose to float precision.
    2. p.getConstraintInfo echoes back the exact childFramePosition/
       Orientation createConstraint was called with -- compare directly
       against the transform computed in (1), so the test checks what
       _form_grasp ACTUALLY passed to PyBullet, not just that the math is
       self-consistent.
    A dynamical (multi-step) version of this test was tried during
    development and abandoned: raw stepSimulation() loops outside
    env.step()'s own control loop let the KUKA arm drift under its own
    residual motor state by up to 0.1rad over 20 steps with NO peg, NO
    gravity, NO constraint at all involved, and env.step() itself (used to
    sidestep that) hit a large, separately-confounded drift attributable to
    the deliberately out-of-workspace test peg placement interacting with
    the environment's own termination/bounds handling -- neither confound
    has anything to do with grasp-weld correctness, which the
    zero-stepping checks below verify exactly and deterministically.
    """
    import pybullet as p
    from manipularl.env import _EE_LINK, _GRASP_OFFSET_Z

    env = ManipulaRLEnv(phase=4, split="eval", seed=EVAL_SEED_LO)
    env.reset()
    c = env.client

    fl_pos, fl_orn = p.getLinkState(env.robot_id, _EE_LINK, computeForwardKinematics=True,
                                    physicsClientId=c)[:2]
    tilt_quat = p.getQuaternionFromEuler([0.10, -0.06, 1.1])   # ~7deg tilt, arbitrary yaw
    peg_pos = (np.array(fl_pos) + np.array([0.02, -0.015, -0.03])).tolist()  # near, plausible
    p.resetBasePositionAndOrientation(env.peg_id, peg_pos, tilt_quat, physicsClientId=c)
    pg_pos, pg_orn = p.getBasePositionAndOrientation(env.peg_id, physicsClientId=c)

    inv_pos, inv_orn = p.invertTransform(pg_pos, pg_orn)
    rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, fl_pos, fl_orn)

    # 1. algebraic invariant
    recon_pos, recon_orn = p.multiplyTransforms(pg_pos, pg_orn, rel_pos, rel_orn)
    assert np.linalg.norm(np.array(recon_pos) - np.array(fl_pos)) < 1e-5
    assert abs(abs(float(np.dot(recon_orn, fl_orn))) - 1.0) < 1e-5

    # 2. _form_grasp actually passed this exact transform to PyBullet
    cid = env._form_grasp(_GRASP_OFFSET_Z)
    info = p.getConstraintInfo(cid, physicsClientId=c)
    got_pos, got_orn = info[7], info[9]
    env.close()
    assert np.linalg.norm(np.array(got_pos) - np.array(rel_pos)) < 1e-6, (
        "childFramePosition wasn't the preserved transform -- did _form_grasp "
        "fall back to the old hardcoded [0,0,child_off]?")
    assert abs(abs(float(np.dot(got_orn, rel_orn))) - 1.0) < 1e-6, (
        "childFrameOrientation wasn't the preserved transform -- did _form_grasp "
        "fall back to identity?")
    # and it must NOT be the old hardcoded value (this peg is deliberately
    # tilted/offset from where the old code would have assumed it was)
    assert np.linalg.norm(np.array(got_pos) - np.array([0, 0, _GRASP_OFFSET_Z])) > 0.01


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_grasp_weld_unaffected_for_pick_place(phase):
    """Phases 1-3 (pick_place/reach) must stay on the OLD zero-offset,
    identity-frame weld exactly -- see _form_grasp's `child_off <= 0.0`
    branch. Same getConstraintInfo technique as
    test_grasp_weld_preserves_transform: confirms the fix is genuinely
    gated by reading back what _form_grasp(0.0) actually told PyBullet,
    regardless of the peg's pose -- must be exactly [0,0,0] + identity,
    byte-for-byte the pre-fix behaviour, for every phase that doesn't use
    the peg_in_hole task.
    """
    import pybullet as p

    env = ManipulaRLEnv(phase=phase, split="eval", seed=EVAL_SEED_LO)
    if not env.cfg.use_peg:
        env.close()
        pytest.skip(f"phase {phase} has no peg")
    env.reset()
    c = env.client

    tilt_quat = p.getQuaternionFromEuler([0.10, -0.06, 1.1])
    peg_pos, _ = p.getBasePositionAndOrientation(env.peg_id, physicsClientId=c)
    p.resetBasePositionAndOrientation(env.peg_id, peg_pos, tilt_quat, physicsClientId=c)

    cid = env._form_grasp(0.0)
    info = p.getConstraintInfo(cid, physicsClientId=c)
    got_pos, got_orn = info[7], info[9]
    env.close()
    assert got_pos == (0.0, 0.0, 0.0), f"phase {phase}: childFramePosition changed: {got_pos}"
    assert got_orn == (0.0, 0.0, 0.0, 1.0), f"phase {phase}: childFrameOrientation changed: {got_orn}"


def test_hole_is_solvable():
    """Scripted: grasp the peg, carry it over the bore, push straight down.
    The scaffold's Stage 4 could never reach success (no hole existed)."""
    import pybullet as p

    env = ManipulaRLEnv(phase=4, split="train")
    env.reset(seed=2)
    # cheat the peg to just above the bore and freeze a grasp, then press down
    hole = env._hole_xy
    p.resetBasePositionAndOrientation(env.peg_id, [hole[0], hole[1], env._hole_mouth_z + 0.02],
                                      [0, 0, 0, 1], physicsClientId=env.client)
    env._grasp_cid = p.createConstraint(
        env.peg_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
        [hole[0], hole[1], env._hole_mouth_z + 0.02], physicsClientId=env.client)
    reached = False
    for k in range(120):
        # lower the constraint target
        z = env._hole_mouth_z + 0.02 - 0.001 * k
        p.changeConstraint(env._grasp_cid, [hole[0], hole[1], max(z, 0.03)],
                           physicsClientId=env.client)
        for _ in range(12):
            p.stepSimulation(physicsClientId=env.client)
        s = env._state_dict()
        if s["peg_depth"] >= PHASES[4].insert_success_depth and \
           np.linalg.norm(s["peg_pos"][:2] - hole) < 0.02:
            reached = True
            break
    env.close()
    assert reached, "scripted straight-down insertion never reached target depth"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
