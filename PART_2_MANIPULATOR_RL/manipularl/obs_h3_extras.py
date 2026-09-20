"""
obs_h3_extras.py -- the 8 observation dims/frame added by "H3" (see
EXPERIMENTS.md), pulled back out of env.py's live observation vector.

H3 widened every phase's observation from 123 to 131 dims/frame (369 to 393
stacked) by appending these 8 fields -- all quantities the reward already
computed internally but had never exposed to the policy: peg tilt in
radians, distance to the nearest point on the ring, a shaped depth signal,
the peg-tip-to-hole vector (3 dims), and ring contact (flag + clipped
force). It applied to every phase, not just Phase 4, because they all share
this one environment class.

That's a real observation-space change: any checkpoint saved before H3 no
longer has a first-layer shape matching env.py's observation_space, so the
report's adopted checkpoints (phase1_ppo, phase2_ppo, phase3_ppo_best,
phase4_full_dream -- all trained before H3) fail to load. Restoring exact
reproducibility for those checkpoints matters more than keeping H3 wired in
by default, so `_assemble_obs` no longer calls this -- env.py is back to
its original 123-dims/frame observation.

The fields themselves are still real and still potentially useful (they
were added for a genuine reason -- fine insertion control couldn't see
tilt/depth-shaped/ring-contact signals it needed). Kept here, callable,
for any future work that wants to deliberately reintroduce them (with a
matching `warm_start_pad`-based retrain, as several post-H3 phase 4
configs already did -- see train.py's warm_start_pad()).

Usage (not currently wired into env.py):
    from .obs_h3_extras import h3_extra_dims
    parts.append(h3_extra_dims(s))   # inserted after obstacles, before stage_onehot
"""

import numpy as np


def h3_extra_dims(s: dict) -> np.ndarray:
    """Returns the 8 H3 dims for state dict `s`: peg_tilt_rad (1),
    min_ring_dist clipped to 1.0 (1), peg_depth_shaped (1), peg_tip_to_hole
    (3), ring_contact (1), ring_contact_force clipped to 50.0 (1)."""
    return np.concatenate([
        np.asarray([s["peg_tilt_rad"]], dtype=np.float32),
        np.asarray([min(float(s["min_ring_dist"]), 1.0)], dtype=np.float32),
        np.asarray([s["peg_depth_shaped"]], dtype=np.float32),
        np.asarray(s["peg_tip_to_hole"], dtype=np.float32).reshape(-1),
        np.asarray([s["ring_contact"]], dtype=np.float32),
        np.asarray([min(float(s["ring_contact_force"]), 50.0)], dtype=np.float32),
    ]).astype(np.float32)
