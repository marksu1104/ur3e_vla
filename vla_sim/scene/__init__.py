"""Public scene API with lazy imports.

Workflow scripts import scene builders from this boundary.  Keeping the imports
lazy lets lightweight configuration code and unit tests run before Isaac Sim
has started, while preserving the existing public import paths.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "base_scene": {
        "BaseSceneCfg",
        "SceneCfg",
        "make_scene_cfg",
    },
    "assets": {
        "ASSET_DIR",
        "ISAAC_QUAT_XYZW",
        "enable_extensions",
        "make_robot_cfg",
        "make_static_cuboid_cfg",
        "make_table_cfg",
        "make_target_cfg",
        "quat_isaac_to_wxyz",
        "quat_wxyz_to_isaac",
        "spawn_assembled_robot",
        "spawn_raw_and_assemble",
    },
    "cameras": {
        "make_camera_cfg",
        "make_policy_camera_cfg",
        "make_wrist_camera_cfg",
        "make_yolo_camera_cfg",
    },
    "materials": {
        "apply_target_colors",
        "bind_gripper_pad_visuals",
        "configure_gripper_pads",
        "hide_markers",
        "prepare_target_visuals",
        "set_marker_material",
        "set_plastic_material",
    },
    "destinations": {"prepare_destination_fixtures"},
    "scene_options": {
        "LightingSettings",
        "SCENE_OPTIONS",
        "SceneOptions",
        "get_scene_options",
    },
}

__all__ = sorted(name for names in _EXPORTS.values() for name in names)


def __getattr__(name: str):
    """Load one implementation module only when its public name is requested."""
    for module_name, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f"{__name__}.{module_name}"), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
