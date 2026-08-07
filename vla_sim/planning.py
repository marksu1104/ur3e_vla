"""Pick-place trajectory and success helpers."""

from __future__ import annotations

import numpy as np

from vla_sim.config import (
    EE_ORIENT_DOWN,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOME_POS,
    WORKSPACE_Z,
)


# Canonical scripted scene timing. The gripper command stays normalized; only
# this duration calculation needs the physical Robotiq joint range.
ARM_SPEED_MPS = 0.20
GRIPPER_SPEED_RAD_S = 0.75
GRIPPER_CLOSED = 1.0


def _quat_mul(q1, q2):
    """Multiply IsaacLab wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _yaw_from_quat(q) -> float:
    w, x, y, z = q
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(sin_yaw, cos_yaw))


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def sample_grasp_quat(target_info: dict, spawn_rot: tuple, rng: np.random.Generator) -> tuple:
    """Sample a handle-avoiding gripper yaw using the shortest useful turn."""
    if not target_info.get("align_gripper_to_yaw"):
        return EE_ORIENT_DOWN

    object_yaw = _yaw_from_quat(spawn_rot)
    current_yaw = _yaw_from_quat(EE_ORIENT_DOWN)
    offsets = np.asarray(
        target_info.get("grasp_yaw_offsets", (-np.pi / 2.0, np.pi / 2.0)),
        dtype=np.float32,
    )
    base_offset = float(target_info.get("gripper_yaw_offset", 0.0))
    candidates = [object_yaw + float(offset) + base_offset for offset in offsets]
    grasp_yaw = min(candidates, key=lambda yaw: abs(_angle_diff(yaw, current_yaw)))

    jitter_range = target_info.get("grasp_yaw_jitter", (-0.20, 0.20))
    grasp_yaw += float(rng.uniform(*jitter_range))
    yaw_quat = (np.cos(grasp_yaw / 2.0), 0.0, 0.0, np.sin(grasp_yaw / 2.0))
    return _quat_mul(yaw_quat, EE_ORIENT_DOWN)


def move_duration(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> float:
    """Return canonical Cartesian segment duration at the shared arm speed."""
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
    *,
    grasp_z_offset: float = 0.0,
    place_z_offset: float = 0.0,
    place_yaw_offset: float = 0.0,
    object_place_xy: tuple[float, float] | None = None,
    place_nudge_scale: float = 1.0,
) -> list[tuple]:
    """Build the no-dwell canonical path with logical 0..1 gripper commands.

    ``grasp_z_offset`` and ``place_z_offset`` raise the descent depth of the
    grasp and the release respectively, in metres. ``place_yaw_offset`` turns
    the held object while travelling from hover to carry, so it is aligned
    before the final descent. When ``object_place_xy`` is given, the release
    pose compensates for the grasp nudge (and its place rotation) so the object
    center, rather than the gripper, lands at that XY coordinate. Defaults
    preserve the canonical simulated trajectory and existing H5/RLDS data.
    ``place_nudge_scale`` accounts for partial slip after the physical grasp.

    They exist because the simulated depths are tuned against the simulated
    gripper and object meshes, while the real cell needs shallower descents:
    what is a clean grasp in simulation can bottom out against the real table.
    Keeping this as a caller-supplied offset means the real robot can be tuned
    without forking the path or perturbing training data.
    """
    tx, ty, tz = map(float, target_resting)
    gx = tx + float(target_info.get("x_nudge", 0.0))
    gy = ty + float(target_info.get("y_nudge", 0.0))
    hover_z = tz + float(target_info["hover_z"])
    grasp_z = tz + float(target_info["grasp_z"]) + float(grasp_z_offset)
    place_z = tz + float(target_info["grasp_z"]) + float(place_z_offset)
    carry_z = max(hover_z, float(target_info["min_carry_z"]))
    carry_z = min(
        carry_z + float(target_info.get("carry_extra_z", 0.0)), WORKSPACE_Z[1]
    )

    close_position = float(target_info.get("gripper_close", GRIPPER_CLOSE))
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
        grasp_quat = _quat_mul(grasp_quat, tool_tilt)

    place_quat = grasp_quat
    if abs(place_yaw_offset) > 1e-8:
        yaw_quat = (
            float(np.cos(place_yaw_offset / 2.0)),
            0.0,
            0.0,
            float(np.sin(place_yaw_offset / 2.0)),
        )
        place_quat = _quat_mul(yaw_quat, grasp_quat)

    hover = (gx, gy, hover_z)
    grasp = (gx, gy, grasp_z)
    pre_grasp = (gx, gy, min(grasp_z + 0.055, hover_z))
    place_ee_x, place_ee_y = map(float, place_xy)
    if object_place_xy is not None:
        nudge_x = float(target_info.get("x_nudge", 0.0))
        nudge_y = float(target_info.get("y_nudge", 0.0))
        cos_yaw = float(np.cos(place_yaw_offset))
        sin_yaw = float(np.sin(place_yaw_offset))
        rotated_nudge_x = place_nudge_scale * (
            cos_yaw * nudge_x - sin_yaw * nudge_y
        )
        rotated_nudge_y = place_nudge_scale * (
            sin_yaw * nudge_x + cos_yaw * nudge_y
        )
        place_ee_x = float(object_place_xy[0]) + rotated_nudge_x
        place_ee_y = float(object_place_xy[1]) + rotated_nudge_y
    place_down = (place_ee_x, place_ee_y, place_z)
    carry = (place_ee_x, place_ee_y, carry_z)

    return [
        (0.0, HOME_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (move_duration(HOME_POS, hover), hover, grasp_quat, GRIPPER_OPEN),
        (move_duration(hover, pre_grasp), pre_grasp, grasp_quat, GRIPPER_OPEN),
        (move_duration(pre_grasp, grasp), grasp, grasp_quat, GRIPPER_OPEN),
        (gripper_duration, grasp, grasp_quat, GRIPPER_CLOSED),
        (move_duration(grasp, hover), hover, grasp_quat, GRIPPER_CLOSED),
        (move_duration(hover, carry), carry, place_quat, GRIPPER_CLOSED),
        (move_duration(carry, place_down), place_down, place_quat, GRIPPER_CLOSED),
        (gripper_duration, place_down, place_quat, GRIPPER_OPEN),
        (move_duration(place_down, carry), carry, place_quat, GRIPPER_OPEN),
        (move_duration(carry, HOME_POS), HOME_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
    ]


def detect_success(
    target_obj,
    ee_pos: np.ndarray,
    target_initial_z: float,
    gripper_q: float,
    place_pos: tuple,
    best_lift_height: float,
    place_xy_threshold: float = 0.10,
) -> tuple[bool, dict]:
    """Check whether the object was lifted and delivered to the target place."""
    obj_pos = target_obj.data.root_pos_w[0].cpu().numpy()
    return evaluate_pick_place_success(
        obj_pos,
        ee_pos,
        target_initial_z,
        gripper_q,
        place_pos,
        best_lift_height,
        place_xy_threshold=place_xy_threshold,
    )


def evaluate_pick_place_success(
    obj_pos: np.ndarray,
    ee_pos: np.ndarray,
    target_initial_z: float,
    gripper_q: float,
    place_pos: tuple | np.ndarray,
    best_lift_height: float,
    place_xy_threshold: float = 0.10,
    ee_pos_for_safety: np.ndarray | None = None,
) -> tuple[bool, dict]:
    """Evaluate pick-and-place success from plain position arrays."""
    obj_pos = np.asarray(obj_pos, dtype=np.float32)
    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    safety_pos = ee_pos if ee_pos_for_safety is None else np.asarray(ee_pos_for_safety)
    place_pos_np = np.asarray(place_pos, dtype=np.float32)

    obj_lift_height = float(obj_pos[2] - target_initial_z)
    obj_place_xy_dist = float(np.linalg.norm(obj_pos[:2] - place_pos_np[:2]))
    obj_lifted = best_lift_height > 0.04
    obj_at_place = obj_place_xy_dist < place_xy_threshold
    ee_safe = bool((1.05 < safety_pos[2] < 1.7) and (-0.5 < safety_pos[0] < 0.7))
    gripper_closed = gripper_q > ((GRIPPER_OPEN + GRIPPER_CLOSE) * 0.5)
    obj_near_ee = bool(np.linalg.norm(obj_pos - ee_pos) < 0.25)

    success = obj_lifted and obj_at_place and ee_safe
    return success, {
        "obj_lifted": bool(obj_lifted),
        "obj_at_place": bool(obj_at_place),
        "gripper_closed": bool(gripper_closed),
        "ee_safe": ee_safe,
        "obj_near_ee": obj_near_ee,
        "obj_pos_final": obj_pos.tolist(),
        "ee_pos_final": ee_pos.tolist(),
        "obj_lift_height": obj_lift_height,
        "best_lift_height": float(best_lift_height),
        "obj_place_xy_dist": obj_place_xy_dist,
        "place_pos": place_pos_np.tolist(),
        "place_xy_threshold": float(place_xy_threshold),
    }
