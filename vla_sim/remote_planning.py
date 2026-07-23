"""Pure trajectory planning for the scripted remote pick-and-place task."""

from __future__ import annotations

import numpy as np

from vla_sim.config import (
    EE_ORIENT_DOWN,
    GRIPPER_OPEN,
    HOME_POS,
    WORKSPACE_Z,
)
from vla_sim.demo_planning import sample_grasp_quat
from vla_sim.geometry import quat_mul
from vla_sim.remote_config import REMOTE_GRIPPER_CLOSE


ARM_SPEED_MPS = 0.20
GRIPPER_SPEED_RAD_S = 0.75
GRIPPER_CLOSED = 1.0


def move_duration(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> float:
    """Return the duration for one Cartesian segment at the shared speed."""
    distance = float(
        np.linalg.norm(
            np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        )
    )
    return 0.0 if distance <= 1e-8 else distance / ARM_SPEED_MPS


def build_pick_place_trajectory(
    target_info: dict,
    target_resting: np.ndarray,
    target_rot: np.ndarray,
    place_xy: tuple[float, float],
) -> list[tuple]:
    """Build a no-dwell path with simultaneous translation/tool rotation.

    The two stationary segments are active gripper close/open operations, not
    idle pauses. Every Cartesian segment uses ``ARM_SPEED_MPS``.
    """
    tx, ty, tz = map(float, target_resting)
    gx = tx + float(target_info.get("x_nudge", 0.0))
    gy = ty + float(target_info.get("y_nudge", 0.0))
    hover_z = tz + float(target_info["hover_z"])
    grasp_z = tz + float(target_info["grasp_z"])
    carry_z = max(hover_z, float(target_info["min_carry_z"]))
    carry_z = min(
        carry_z + float(target_info.get("carry_extra_z", 0.0)), WORKSPACE_Z[1]
    )

    # Keep timing in physical radians while exposing logical commands to the
    # shared RobotController (0.0=open, 1.0=closed).
    close_position = float(target_info.get("gripper_close", REMOTE_GRIPPER_CLOSE))
    gripper_duration = (close_position - GRIPPER_OPEN) / GRIPPER_SPEED_RAD_S
    grasp_quat = sample_grasp_quat(
        target_info,
        tuple(map(float, target_rot)),
        np.random.default_rng(0),
    )
    tilt_x = float(target_info.get("grasp_tilt_x", 0.0))
    if abs(tilt_x) > 1e-8:
        tool_tilt = (
            float(np.cos(tilt_x / 2.0)),
            float(np.sin(tilt_x / 2.0)),
            0.0,
            0.0,
        )
        grasp_quat = quat_mul(grasp_quat, tool_tilt)

    hover = (gx, gy, hover_z)
    grasp = (gx, gy, grasp_z)
    pre_grasp = (gx, gy, min(grasp_z + 0.055, hover_z))
    place_down = (float(place_xy[0]), float(place_xy[1]), grasp_z)
    carry = (float(place_xy[0]), float(place_xy[1]), carry_z)

    return [
        (0.0, HOME_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (move_duration(HOME_POS, hover), hover, grasp_quat, GRIPPER_OPEN),
        (move_duration(hover, pre_grasp), pre_grasp, grasp_quat, GRIPPER_OPEN),
        (move_duration(pre_grasp, grasp), grasp, grasp_quat, GRIPPER_OPEN),
        (gripper_duration, grasp, grasp_quat, GRIPPER_CLOSED),
        (move_duration(grasp, hover), hover, grasp_quat, GRIPPER_CLOSED),
        (move_duration(hover, carry), carry, grasp_quat, GRIPPER_CLOSED),
        (
            move_duration(carry, place_down),
            place_down,
            grasp_quat,
            GRIPPER_CLOSED,
        ),
        (gripper_duration, place_down, grasp_quat, GRIPPER_OPEN),
        (move_duration(place_down, carry), carry, grasp_quat, GRIPPER_OPEN),
        (move_duration(carry, HOME_POS), HOME_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
    ]
