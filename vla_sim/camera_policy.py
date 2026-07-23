"""Policy-only camera configuration for the canonical scene."""

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

from vla_sim.config import (
    CAMERA_HEIGHT,
    CAMERA_MAIN_FOCAL,
    CAMERA_MAIN_POS,
    CAMERA_MAIN_ROT,
    CAMERA_WIDTH,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_WIDTH,
)


POLICY_CAMERA_NAME = "camera_policy"


def make_policy_camera_cfg() -> CameraCfg:
    """Create the VLA observation camera; YOLO keeps its separate camera."""
    return CameraCfg(
        prim_path="/World/CameraPolicy",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA_MAIN_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_MAIN_POS,
            rot=CAMERA_MAIN_ROT,
            convention="opengl",
        ),
    )


def make_wrist_camera_cfg() -> CameraCfg:
    """Create the fixed wrist observation required by the existing H5 schema."""
    return CameraCfg(
        prim_path="/World/Robot/wrist_3_link/CameraWrist",
        update_period=0.0,
        height=WRIST_CAMERA_HEIGHT,
        width=WRIST_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.12),
            rot=(0.0, 1.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
