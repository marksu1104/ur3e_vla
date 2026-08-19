"""Canonical scene extended with two extra third-person cameras.

Does not modify vla_sim/scene.py. Instead, subclasses its ``SceneCfg``
(an ``InteractiveSceneCfg``) and adds two new ``CameraCfg`` fields.
``InteractiveScene`` builds entities by walking the fields of whatever
scene-cfg dataclass it's given, so subclass fields are picked up the
same way the base class's fields are -- no change to the base file is
needed for the new cameras to become real, queryable entities
(``scene["camera_left"]`` / ``scene["camera_right"]``).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils.configclass import configclass

from vla_sim.config import CAMERA_HEIGHT, CAMERA_MAIN_FOCAL, CAMERA_WIDTH
from vla_sim.scene import SceneCfg, quat_wxyz_to_isaac

from vla_sim.multiview import config_multiview as mvcfg


@configclass
class MultiviewSceneCfg(SceneCfg):
    """Canonical scene plus two extra third-person cameras."""

    camera_top = CameraCfg(
        prim_path="/World/CameraTop",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA_MAIN_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=mvcfg.CAMERA_TOP_POS,
            rot=quat_wxyz_to_isaac(mvcfg.CAMERA_TOP_ROT),
            convention="opengl",
        ),
    )
    camera_left = CameraCfg(
        prim_path="/World/CameraLeft",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA_MAIN_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=mvcfg.CAMERA_LEFT_POS,
            rot=quat_wxyz_to_isaac(mvcfg.CAMERA_LEFT_ROT),
            convention="opengl",
        ),
    )
    camera_right = CameraCfg(
        prim_path="/World/CameraRight",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA_MAIN_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=mvcfg.CAMERA_RIGHT_POS,
            rot=quat_wxyz_to_isaac(mvcfg.CAMERA_RIGHT_ROT),
            convention="opengl",
        ),
    )


def make_multiview_scene_cfg(
    *,
    num_envs: int = 1,
    env_spacing: float = 2.0,
    stream_width: int | None = None,
    stream_height: int | None = None,
) -> MultiviewSceneCfg:
    """Multiview counterpart of ``vla_sim.scene.make_scene_cfg``.

    Kept as a thin, separate function (rather than editing the
    original) so ``runtime_multiview.py`` can build the extended scene
    the same way the base runtime builds ``SceneCfg``.
    """
    cfg = MultiviewSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    if stream_width is not None:
        cfg.camera_yolo.width = stream_width
    if stream_height is not None:
        cfg.camera_yolo.height = stream_height
    return cfg