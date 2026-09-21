"""Camera definitions for the shared base scene."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

from vla_sim.config import (
    CAMERA_HEIGHT,
    CAMERA_MAIN_FOCAL,
    CAMERA_MAIN_POS,
    CAMERA_MAIN_ROT,
    CAMERA_WIDTH,
    STREAM_HEIGHT,
    STREAM_WIDTH,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_WIDTH,
    YOLO_CAMERA_FOCAL,
    YOLO_CAMERA_POS,
    YOLO_CAMERA_ROT,
)
from .assets import quat_wxyz_to_isaac


def make_camera_cfg(
    prim_path: str,
    *,
    width: int,
    height: int,
    focal_length: float,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
) -> CameraCfg:
    """Build an RGB camera with the project's stable quaternion convention."""
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal_length),
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=quat_wxyz_to_isaac(rotation),
            convention="opengl",
        ),
    )


def make_yolo_camera_cfg(prim_path: str = "/World/CameraYolo") -> CameraCfg:
    return make_camera_cfg(
        prim_path,
        width=STREAM_WIDTH,
        height=STREAM_HEIGHT,
        focal_length=YOLO_CAMERA_FOCAL,
        position=YOLO_CAMERA_POS,
        rotation=YOLO_CAMERA_ROT,
    )


def make_policy_camera_cfg(prim_path: str = "/World/CameraPolicy") -> CameraCfg:
    return make_camera_cfg(
        prim_path,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        focal_length=CAMERA_MAIN_FOCAL,
        position=CAMERA_MAIN_POS,
        rotation=CAMERA_MAIN_ROT,
    )


def make_wrist_camera_cfg(
    prim_path: str = "/World/Robot/wrist_3_link/CameraWrist",
) -> CameraCfg:
    return make_camera_cfg(
        prim_path,
        width=WRIST_CAMERA_WIDTH,
        height=WRIST_CAMERA_HEIGHT,
        focal_length=18.0,
        position=(0.0, 0.0, 0.12),
        rotation=(0.0, 1.0, 0.0, 0.0),
    )
