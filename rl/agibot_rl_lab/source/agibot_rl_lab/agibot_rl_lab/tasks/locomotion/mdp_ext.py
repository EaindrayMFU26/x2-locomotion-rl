"""Local mdp extensions for agibot_rl_lab.

Currently only the training-time IMU pitch-bias observation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def projected_gravity_with_bias(
    env: ManagerBasedRLEnv,
    bias_range: tuple[float, float] = (-0.12, 0.12),
) -> torch.Tensor:
    """projected_gravity with per-episode persistent IMU pitch bias.

    Samples a fixed x-axis bias once per episode from Uniform(bias_range) and
    adds it to projected_gravity[x] for the entire episode.  This simulates a
    systematic IMU zero-point offset (e.g. +0.08 seen on certain robots), which
    per-step Unoise cannot cover because its expected value is zero.

    bias_range: (min, max) in projected_gravity units (~sin of tilt angle).
                ±0.12 covers ±6.9°, enough to include the observed +0.08 with margin.
    """
    from isaaclab.envs.mdp.observations import projected_gravity as _proj_gravity

    grav = _proj_gravity(env)

    if not hasattr(env, "_imu_pitch_bias"):
        env._imu_pitch_bias = torch.zeros(env.num_envs, device=env.device)

    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if reset_mask.any():
            lo, hi = bias_range
            new_bias = torch.empty(env.num_envs, device=env.device).uniform_(lo, hi)
            env._imu_pitch_bias = torch.where(reset_mask, new_bias, env._imu_pitch_bias)

    biased = grav.clone()
    biased[:, 0] = biased[:, 0] + env._imu_pitch_bias
    return biased
