"""Shared base scene for every single-environment UR3e workflow."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from vla_sim.config import (
    BACKDROP_BACK_POS,
    BACKDROP_BACK_SIZE,
    BACKDROP_SIDE_POS,
    BACKDROP_SIDE_SIZE,
    PLACE_MARKER_RADIUS,
    PLACE_MARKER_THICKNESS,
    PLACE_POSITIONS,
    TABLE_A_POS,
    TABLE_B_POS,
    TABLE_MAT_A_POS,
    TABLE_MAT_B_POS,
    TABLE_MAT_SIZE,
)
from .assets import (
    ASSET_DIR,
    ISAAC_QUAT_XYZW,
    enable_extensions,
    make_base_target_cfg,
    make_robot_cfg,
    make_static_cuboid_cfg,
    make_table_cfg,
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
from .scene_options import SceneOptions, get_scene_options


def _marker_cfg(index: int) -> AssetBaseCfg:
    """Create a collision-free visual disk flush with the table mat."""
    x, y = PLACE_POSITIONS[index]
    table_surface_z = TABLE_MAT_A_POS[2] + 0.5 * TABLE_MAT_SIZE[2]
    marker_center_z = table_surface_z + 0.5 * PLACE_MARKER_THICKNESS
    return AssetBaseCfg(
        prim_path=f"/World/MarkerP{index}",
        spawn=sim_utils.MeshCylinderCfg(
            radius=PLACE_MARKER_RADIUS,
            height=PLACE_MARKER_THICKNESS,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, marker_center_z)),
    )


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """The canonical single-arm scene for all runtimes."""

    if ISAAC_QUAT_XYZW:
        fill_light = AssetBaseCfg(
            prim_path="/World/FillLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=650.0, color=(0.75, 0.75, 0.75)
            ),
        )
        camera_fill_light = AssetBaseCfg(
            prim_path="/World/CameraFillLight",
            spawn=sim_utils.SphereLightCfg(
                intensity=6500.0,
                color=(0.75, 0.75, 0.75),
                normalize=True,
                radius=0.75,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.13, 0.84, 1.80)),
        )
    robot = make_robot_cfg()
    table_a = make_table_cfg("/World/TableA", TABLE_A_POS)
    table_b = make_table_cfg("/World/TableB", TABLE_B_POS)
    mat_a = make_static_cuboid_cfg("/World/MatA", TABLE_MAT_SIZE, TABLE_MAT_A_POS)
    mat_b = make_static_cuboid_cfg("/World/MatB", TABLE_MAT_SIZE, TABLE_MAT_B_POS)
    backdrop_back = make_static_cuboid_cfg(
        "/World/BackdropBack", BACKDROP_BACK_SIZE, BACKDROP_BACK_POS
    )
    backdrop_side = make_static_cuboid_cfg(
        "/World/BackdropSide", BACKDROP_SIDE_SIZE, BACKDROP_SIDE_POS
    )
    red_mug = make_base_target_cfg("red_mug")
    spoon = make_base_target_cfg("spoon")
    bowl = make_base_target_cfg("bowl")
    marker_p0 = _marker_cfg(0)
    marker_p1 = _marker_cfg(1)
    marker_p2 = _marker_cfg(2)
    camera_yolo = make_yolo_camera_cfg()
    camera_policy = make_policy_camera_cfg()
    camera_wrist = make_wrist_camera_cfg()


def make_scene_cfg(
    *,
    num_envs: int = 1,
    env_spacing: float = 2.0,
    stream_width: int | None = None,
    stream_height: int | None = None,
    profile: str | SceneOptions = "canonical",
) -> BaseSceneCfg:
    """Compose the canonical physics scene for one named workflow."""
    resolved = get_scene_options(profile)
    cfg = BaseSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    for camera_name in ("camera_yolo", "camera_policy", "camera_wrist"):
        if camera_name not in resolved.cameras:
            setattr(cfg, camera_name, None)
    if stream_width is not None and cfg.camera_yolo is not None:
        cfg.camera_yolo.width = stream_width
    if stream_height is not None and cfg.camera_yolo is not None:
        cfg.camera_yolo.height = stream_height
    if hasattr(cfg, "fill_light"):
        cfg.fill_light.spawn.intensity = resolved.lighting.dome_intensity
        cfg.fill_light.spawn.color = resolved.lighting.dome_color
    if hasattr(cfg, "camera_fill_light"):
        cfg.camera_fill_light.spawn.intensity = resolved.lighting.fill_intensity
        cfg.camera_fill_light.spawn.color = resolved.lighting.fill_color
        cfg.camera_fill_light.spawn.radius = resolved.lighting.fill_radius
        cfg.camera_fill_light.init_state.pos = resolved.lighting.fill_position
    return cfg


# Temporary compatibility alias for external or in-progress MultiView code.
SceneCfg = BaseSceneCfg


# Compatibility export required by the active MultiView workstream. Canonical
# code imports presentation helpers directly from vla_sim.scene.materials.
from vla_sim.scene.materials import set_plastic_material  # noqa: E402,F401
