"""Public scene API for UR3e workflows.

Keep workflow code importing this package rather than individual implementation
modules.  The package is deliberately a compatibility boundary while the
scene internals evolve.
"""

from .base_scene import (
    BaseSceneCfg,
    SceneCfg,
    make_scene_cfg,
)
from .assets import (
    ASSET_DIR,
    ISAAC_QUAT_XYZW,
    enable_extensions,
    make_robot_cfg,
    make_static_cuboid_cfg,
    make_table_cfg,
    make_target_cfg,
    quat_isaac_to_wxyz,
    quat_wxyz_to_isaac,
    spawn_assembled_robot,
    spawn_raw_and_assemble,
)
from .cameras import (
    make_camera_cfg,
    make_policy_camera_cfg,
    make_wrist_camera_cfg,
    make_yolo_camera_cfg,
)
from .materials import (
    apply_target_colors,
    bind_gripper_pad_visuals,
    configure_gripper_pads,
    hide_markers,
    prepare_target_visuals,
    set_marker_material,
    set_plastic_material,
)
from .destinations import prepare_destination_fixtures
from .scene_options import (
    SCENE_OPTIONS,
    LightingSettings,
    SceneOptions,
    get_scene_options,
)

__all__ = [
    "ASSET_DIR",
    "BaseSceneCfg",
    "ISAAC_QUAT_XYZW",
    "LightingSettings",
    "SCENE_OPTIONS",
    "SceneCfg",
    "SceneOptions",
    "apply_target_colors",
    "bind_gripper_pad_visuals",
    "configure_gripper_pads",
    "enable_extensions",
    "hide_markers",
    "get_scene_options",
    "make_camera_cfg",
    "make_policy_camera_cfg",
    "make_robot_cfg",
    "make_scene_cfg",
    "make_static_cuboid_cfg",
    "make_table_cfg",
    "make_target_cfg",
    "make_wrist_camera_cfg",
    "make_yolo_camera_cfg",
    "prepare_destination_fixtures",
    "prepare_target_visuals",
    "quat_isaac_to_wxyz",
    "quat_wxyz_to_isaac",
    "set_marker_material",
    "set_plastic_material",
    "spawn_assembled_robot",
    "spawn_raw_and_assemble",
]
