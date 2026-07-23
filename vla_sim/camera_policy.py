"""Policy-only camera configuration for the canonical scene."""

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

from vla_sim.config import (
    CAMERA_HEIGHT,
    CAMERA_MAIN_FOCAL,
    CAMERA_MAIN_POS,
    CAMERA_MAIN_ROT,
    CAMERA_WIDTH,
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
