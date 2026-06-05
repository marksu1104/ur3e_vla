"""Demonstration planning helpers for scene randomization and success checks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vla_sim.config import EE_ORIENT_DOWN, HOME_POS
from vla_sim.geometry import angle_diff, quat_mul, yaw_from_quat_wxyz

MUG_TARGET_KEYS = ("red_mug", "blue_mug")
MIN_OBJECT_XY_DIST = 0.12
MIN_PLACE_XY_DIST = 0.12


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
    place_pos_np = np.asarray(place_pos, dtype=np.float32)

    obj_lift_height = float(obj_pos[2] - target_initial_z)
    obj_place_xy_dist = float(np.linalg.norm(obj_pos[:2] - place_pos_np[:2]))
    obj_lifted = best_lift_height > 0.04
    obj_at_place = obj_place_xy_dist < place_xy_threshold
    ee_safe = bool((1.05 < ee_pos[2] < 1.7) and (-0.5 < ee_pos[0] < 0.7))
    gripper_closed = gripper_q > 0.3
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


def detect_target_reached(
    ee_pos: np.ndarray,
    target_pos: tuple,
    target_initial_z: float,
    threshold: float = 0.08,
) -> tuple[bool, dict]:
    """Check whether the end effector reached the scripted target point."""
    target_pos_np = np.asarray(target_pos, dtype=np.float32)
    ee_target_dist = float(np.linalg.norm(ee_pos - target_pos_np))
    target_reached = bool(ee_target_dist < threshold)
    ee_safe = bool((1.05 < ee_pos[2] < 1.7) and (-0.5 < ee_pos[0] < 0.7))

    return target_reached, {
        "target_reached": target_reached,
        "ee_target_dist": ee_target_dist,
        "target_pos": target_pos_np.tolist(),
        "ee_pos_final": ee_pos.tolist(),
        "obj_lifted": False,
        "gripper_closed": True,
        "ee_safe": ee_safe,
        "obj_near_ee": False,
        "obj_lift_height": 0.0,
        "target_initial_z": float(target_initial_z),
    }
