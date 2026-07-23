"""Demonstration planning helpers for scene randomization and success checks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vla_sim.config import EE_ORIENT_DOWN, GRIPPER_CLOSE, GRIPPER_OPEN, HOME_POS, WORKSPACE_Z
from vla_sim.geometry import angle_diff, quat_mul, yaw_from_quat_wxyz
from vla_sim.remote_config import REMOTE_GRIPPER_CLOSE

MUG_TARGET_KEYS = ("red_mug", "blue_mug")
MIN_OBJECT_XY_DIST = 0.12
MIN_PLACE_XY_DIST = 0.12

# Canonical scripted scene timing. The gripper command stays normalized; only
# this duration calculation needs the physical Robotiq joint range.
ARM_SPEED_MPS = 0.20
GRIPPER_SPEED_RAD_S = 0.75
GRIPPER_CLOSED = 1.0


def sample_grasp_quat(target_info: dict, spawn_rot: tuple, rng: np.random.Generator) -> tuple:
    """Sample a handle-avoiding gripper yaw using the shortest useful turn."""
    if not target_info.get("align_gripper_to_yaw"):
        return EE_ORIENT_DOWN

    object_yaw = yaw_from_quat_wxyz(spawn_rot)
    current_yaw = yaw_from_quat_wxyz(EE_ORIENT_DOWN)
    offsets = np.asarray(
        target_info.get("grasp_yaw_offsets", (-np.pi / 2.0, np.pi / 2.0)),
        dtype=np.float32,
    )
    base_offset = float(target_info.get("gripper_yaw_offset", 0.0))
    candidates = [object_yaw + float(offset) + base_offset for offset in offsets]
    grasp_yaw = min(candidates, key=lambda yaw: abs(angle_diff(yaw, current_yaw)))

    jitter_range = target_info.get("grasp_yaw_jitter", (-0.20, 0.20))
    grasp_yaw += float(rng.uniform(*jitter_range))
    yaw_quat = (np.cos(grasp_yaw / 2.0), 0.0, 0.0, np.sin(grasp_yaw / 2.0))
    return quat_mul(yaw_quat, EE_ORIENT_DOWN)


def randomize_target_pose(
    target_info: dict,
    rng: np.random.Generator,
    randomize_pos: bool = True,
    randomize_rot: bool = True,
) -> tuple:
    """Return randomized target position and orientation."""
    base_x, base_y, base_z = target_info["spawn_pos"]
    base_rot = target_info.get("spawn_rot", (1.0, 0.0, 0.0, 0.0))

    if randomize_pos:
        pos_randomization = target_info.get("pos_randomization", {})
        x_range = pos_randomization.get("x", (-0.08, 0.08))
        y_range = pos_randomization.get("y", (-0.10, 0.10))
        rand_x = base_x + rng.uniform(*x_range)
        rand_y = base_y + rng.uniform(*y_range)
    else:
        rand_x, rand_y = base_x, base_y

    if randomize_rot:
        yaw = rng.uniform(-np.pi, np.pi)
        yaw_rot = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        spawn_rot = quat_mul(yaw_rot, base_rot)
    else:
        spawn_rot = base_rot

    return (rand_x, rand_y, base_z), spawn_rot


def _xy_dist(a: tuple, b: tuple) -> float:
    return float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))


def sample_scene_object_poses(
    targets: dict,
    rng: np.random.Generator,
    randomize_pos: bool = True,
    randomize_rot: bool = True,
    place_pos: tuple = HOME_POS,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Sample per-episode mug poses while avoiding overlaps and the place point."""
    sampled: dict[str, tuple[tuple, tuple]] = {}
    for key in MUG_TARGET_KEYS:
        if key not in targets:
            continue
        info = targets[key]
        for _ in range(100):
            pos, rot = randomize_target_pose(info, rng, randomize_pos, randomize_rot)
            if _xy_dist(pos, place_pos) < MIN_PLACE_XY_DIST:
                continue
            if any(_xy_dist(pos, other_pos) < MIN_OBJECT_XY_DIST for other_pos, _ in sampled.values()):
                continue
            sampled[key] = (pos, rot)
            break
        else:
            base_pos = info["spawn_pos"]
            base_rot = info.get("spawn_rot", (1.0, 0.0, 0.0, 0.0))
            sampled[key] = (base_pos, base_rot)
            if log_fn is not None:
                log_fn(f"  WARNING: using base pose for {key}; could not sample non-overlapping pose")
    return sampled


def sample_grasp_parameters(target_info: dict, rng: np.random.Generator) -> dict:
    """Sample per-episode grasp offsets/heights from target config."""
    params = {
        "grasp_z": float(target_info["grasp_z"]),
        "hover_z": float(target_info["hover_z"]),
        "x_nudge": float(target_info.get("x_nudge", 0.0)),
        "y_nudge": float(target_info.get("y_nudge", 0.0)),
    }
    cfg = target_info.get("grasp_randomization") or {}
    for key in ("grasp_z", "hover_z", "x_nudge", "y_nudge"):
        value = cfg.get(key)
        if value is None:
            continue
        if isinstance(value, (tuple, list)) and len(value) == 2:
            params[key] = float(rng.uniform(float(value[0]), float(value[1])))
        else:
            params[key] = float(value)
    return params


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


def build_canonical_pick_place_trajectory(
    target_info: dict,
    target_resting: np.ndarray,
    target_rot: np.ndarray,
    place_xy: tuple[float, float],
) -> list[tuple]:
    """Build the no-dwell canonical path with logical 0..1 gripper commands."""
    tx, ty, tz = map(float, target_resting)
    gx = tx + float(target_info.get("x_nudge", 0.0))
    gy = ty + float(target_info.get("y_nudge", 0.0))
    hover_z = tz + float(target_info["hover_z"])
    grasp_z = tz + float(target_info["grasp_z"])
    carry_z = max(hover_z, float(target_info["min_carry_z"]))
    carry_z = min(
        carry_z + float(target_info.get("carry_extra_z", 0.0)), WORKSPACE_Z[1]
    )

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
        (move_duration(carry, place_down), place_down, grasp_quat, GRIPPER_CLOSED),
        (gripper_duration, place_down, grasp_quat, GRIPPER_OPEN),
        (move_duration(place_down, carry), carry, grasp_quat, GRIPPER_OPEN),
        (move_duration(carry, HOME_POS), HOME_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
    ]


def build_pick_place_trajectory(
    target_info: dict,
    target_resting: np.ndarray,
    spawn_rot: tuple,
    rng: np.random.Generator,
    place_pos: tuple = HOME_POS,
) -> tuple[list[tuple], dict]:
    """Build the shared scripted mug trajectory and its episode metadata."""
    tx, ty, tz = np.asarray(target_resting, dtype=np.float32).tolist()
    grasp_params = sample_grasp_parameters(target_info, rng)
    gx = tx + grasp_params["x_nudge"]
    gy = ty + grasp_params["y_nudge"]
    hover_z = tz + grasp_params["hover_z"]
    grasp_z = tz + grasp_params["grasp_z"]

    hover_pos = (gx, gy, hover_z)
    pre_grasp_pos = (gx, gy, min(grasp_z + 0.055, hover_z))
    grasp_pos = (gx, gy, grasp_z)
    lift_start_pos = (gx, gy, min(grasp_z + 0.025, hover_z))
    grasp_quat = sample_grasp_quat(target_info, spawn_rot, rng)

    durations = {
        "hover": float(rng.uniform(3.2, 3.8)),
        "yaw": 0.0,
        "pre_grasp": float(rng.uniform(1.9, 2.4)),
        "descend": float(rng.uniform(1.0, 1.4)),
        "close": float(rng.uniform(0.35, 0.55)),
        "lift_start": float(rng.uniform(0.45, 0.70)),
        "lift": float(rng.uniform(2.1, 2.6)),
        "carry": float(rng.uniform(2.7, 3.3)),
        "release": float(rng.uniform(1.6, 2.1)),
    }
    trajectory = [
        (0.0, place_pos, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (durations["hover"], hover_pos, grasp_quat, GRIPPER_OPEN),
        (durations["pre_grasp"], pre_grasp_pos, grasp_quat, GRIPPER_OPEN),
        (durations["descend"], grasp_pos, grasp_quat, GRIPPER_OPEN),
        (durations["close"], grasp_pos, grasp_quat, GRIPPER_CLOSE),
        (durations["lift_start"], lift_start_pos, grasp_quat, GRIPPER_CLOSE),
        (durations["lift"], hover_pos, grasp_quat, GRIPPER_CLOSE),
        (durations["carry"], place_pos, grasp_quat, GRIPPER_CLOSE),
        (durations["release"], place_pos, grasp_quat, GRIPPER_OPEN),
    ]
    metadata = {
        "grasp_params": {key: float(value) for key, value in grasp_params.items()},
        "grasp_quat": list(map(float, grasp_quat)),
        "grasp_pos": list(map(float, grasp_pos)),
        "hover_pos": list(map(float, hover_pos)),
        "place_pos": list(map(float, place_pos)),
        "durations": durations,
    }
    return trajectory, metadata


def randomize_lighting(
    stage,
    rng: np.random.Generator,
    enabled: bool = True,
    log_fn: Callable[[str], None] | None = None,
):
    """Randomize dome light intensity and color."""
    if not enabled:
        return
    try:
        from pxr import UsdLux
        light_prim = stage.GetPrimAtPath("/World/light")
        if not light_prim.IsValid():
            return
        light = UsdLux.DomeLight(light_prim)
        intensity = 3000.0 * rng.uniform(0.7, 1.3)
        light.GetIntensityAttr().Set(float(intensity))
        color = (1.0, 0.95, 0.85) if rng.random() < 0.5 else (0.85, 0.95, 1.0)
        light.GetColorAttr().Set(color)
    except Exception as exc:
        if log_fn is not None:
            log_fn(f"randomize_lighting failed: {exc}")


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
