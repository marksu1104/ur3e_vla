"""Base scene extended with three fixed recording cameras.

This module subclasses ``BaseSceneCfg`` and adds three ``CameraCfg`` fields.
``InteractiveScene`` builds entities by walking the fields of whatever
scene-cfg dataclass it's given, so subclass fields are picked up the
same way the base class's fields are -- no change to the base file is
needed for the new cameras to become real, queryable entities
(``scene["camera_left"]`` / ``scene["camera_right"]``).
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from vla_sim.config import CAMERA_HEIGHT, CAMERA_MAIN_FOCAL, CAMERA_WIDTH
from vla_sim.scene import BaseSceneCfg, make_camera_cfg

from vla_sim.multiview import config_multiview as mvcfg


@configclass
class MultiviewSceneCfg(BaseSceneCfg):
    """Canonical scene plus two extra third-person cameras."""

    camera_top = make_camera_cfg(
        "/World/CameraTop",
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        focal_length=CAMERA_MAIN_FOCAL,
        position=mvcfg.CAMERA_TOP_POS,
        rotation=mvcfg.CAMERA_TOP_ROT,
    )
    camera_left = make_camera_cfg(
        "/World/CameraLeft",
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        focal_length=CAMERA_MAIN_FOCAL,
        position=mvcfg.CAMERA_LEFT_POS,
        rotation=mvcfg.CAMERA_LEFT_ROT,
    )
    camera_right = make_camera_cfg(
        "/World/CameraRight",
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        focal_length=CAMERA_MAIN_FOCAL,
        position=mvcfg.CAMERA_RIGHT_POS,
        rotation=mvcfg.CAMERA_RIGHT_ROT,
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
    the same way the base runtime builds ``BaseSceneCfg``.
    """
    cfg = MultiviewSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    if stream_width is not None:
        cfg.camera_yolo.width = stream_width
    if stream_height is not None:
        cfg.camera_yolo.height = stream_height
    return cfg
