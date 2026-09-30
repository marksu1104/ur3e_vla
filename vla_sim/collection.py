"""Small shared helpers for scripted H5 collection entry points.

The single-view and MultiView collectors intentionally keep their own recording
loops because their H5 payloads differ.  This module owns only the identical
episode setup and trajectory construction so those steps cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla_sim.actions import PoseTrajectoryPlayer
from vla_sim.config import TARGETS
from vla_sim.pick_place import build_pick_place_trajectory
from vla_sim.simulation import as_torch, pose_wxyz_from_sim


RECORD_EVERY_N_STEPS = 12  # 60 Hz simulation / 12 = 5 Hz policy data.
SETTLE_STEPS = 120


@dataclass(frozen=True)
class ScriptedEpisode:
    """Target state and trajectory shared by scripted collection workflows."""

    target: object
    target_resting: np.ndarray
    target_initial_z: float
    player: PoseTrajectoryPlayer


def prepare_scripted_episode(
    runtime, target_name: str, place_xy: tuple[float, float]
) -> ScriptedEpisode:
    """Reset the scene and build the established scripted pick/place path."""
    scene = runtime.scene
    controller = runtime.robot_controller
    if scene is None or controller is None:
        raise RuntimeError("collection runtime did not initialize")

    runtime.reset_targets()
    controller.reset_home()
    for _ in range(SETTLE_STEPS):
        runtime.step()

    target = scene[target_name]
    target_resting = as_torch(target.data.root_pos_w)[0].cpu().numpy()
    target_rot = pose_wxyz_from_sim(target.data.root_link_pose_w)[0, 3:7].cpu().numpy()
    trajectory = build_pick_place_trajectory(
        TARGETS[target_name], target_resting, target_rot, place_xy
    )
    return ScriptedEpisode(
        target=target,
        target_resting=target_resting,
        target_initial_z=float(target_resting[2]),
        player=PoseTrajectoryPlayer(trajectory, device=runtime.device),
    )
