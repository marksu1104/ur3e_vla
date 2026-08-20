"""Canonical three-object scene and its USD presentation helpers."""

from __future__ import annotations

from pathlib import Path

import omni.kit.app
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from vla_sim.isaac_app import log
from vla_sim.config import (
    BACKDROP_BACK_POS,
    BACKDROP_BACK_SIZE,
    BACKDROP_SIDE_POS,
    BACKDROP_SIDE_SIZE,
    ASSEMBLY_NAMESPACE,
    GRIPPER_MOUNT_REL_CANDIDATES,
    GRIPPER_PRIM_PATH,
    GRIPPER_USD_RELATIVE,
    CAMERA_HEIGHT,
    CAMERA_MAIN_FOCAL,
    CAMERA_MAIN_POS,
    CAMERA_MAIN_ROT,
    CAMERA_WIDTH,
    PLACE_MARKER_RADIUS,
    PLACE_MARKER_THICKNESS,
    PLACE_POSITIONS,
    ROBOT_BASE_POS,
    ROBOT_BASE_ROT,
    ROBOT_PRIM_PATH,
    SceneProfile,
    STREAM_HEIGHT,
    STREAM_WIDTH,
    TABLE_A_POS,
    TABLE_B_POS,
    TABLE_MAT_A_POS,
    TABLE_MAT_B_POS,
    TABLE_MAT_SIZE,
    TABLE_ROT,
    TABLE_SCALE,
    TABLE_USD_RELATIVE,
    TARGETS,
    UR3E_MOUNT_ABS,
    UR3E_USD_RELATIVE,
    VARIANT_NAME,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_WIDTH,
    YOLO_CAMERA_FOCAL,
    YOLO_CAMERA_POS,
    YOLO_CAMERA_ROT,
    get_scene_profile,
)

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
ISAAC_QUAT_XYZW = AssetBaseCfg.InitialStateCfg().rot == (0.0, 0.0, 0.0, 1.0)


def quat_wxyz_to_isaac(quaternion):
    """Convert the project's wxyz convention at the Isaac API boundary."""
    if not ISAAC_QUAT_XYZW:
        return quaternion
    if isinstance(quaternion, (tuple, list)):
        w, x, y, z = quaternion
        return (x, y, z, w)
    return quaternion[..., [1, 2, 3, 0]]


def quat_isaac_to_wxyz(quaternion):
    """Convert an Isaac quaternion to the project's stable wxyz convention."""
    if not ISAAC_QUAT_XYZW:
        return quaternion
    if isinstance(quaternion, (tuple, list)):
        x, y, z, w = quaternion
        return (w, x, y, z)
    return quaternion[..., [3, 0, 1, 2]]


def enable_extensions() -> None:
    manager = omni.kit.app.get_app().get_extension_manager()
    enabled = manager.set_extension_enabled_immediate(
        "isaacsim.robot_setup.assembler", True
    )
    log(f"robot_setup.assembler enabled = {enabled}")


def _find_gripper_mount(stage) -> str:
    from pxr import Usd

    for relative_path in GRIPPER_MOUNT_REL_CANDIDATES:
        path = f"{GRIPPER_PRIM_PATH}/{relative_path}"
        if stage.GetPrimAtPath(path).IsValid():
            return path
    root = stage.GetPrimAtPath(GRIPPER_PRIM_PATH)
    for prim in Usd.PrimRange(root):
        if "base_link" in prim.GetName().lower():
            return str(prim.GetPath())
    raise RuntimeError("could not locate the Robotiq assembly mount")


def spawn_raw_and_assemble(gripper_usd_relative: str | None = None) -> None:
    """Load and assemble the canonical UR3e and official Robotiq physics USD."""
    from isaaclab.sim.utils.prims import add_usd_reference
    from isaacsim.robot_setup.assembler import RobotAssembler
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, "/World")
    asset_root = str(ISAAC_NUCLEUS_DIR)
    gripper_asset = gripper_usd_relative or GRIPPER_USD_RELATIVE
    add_usd_reference(
        usd_path=f"{asset_root}/{UR3E_USD_RELATIVE}", prim_path=ROBOT_PRIM_PATH
    )
    add_usd_reference(
        usd_path=f"{asset_root}/{gripper_asset}", prim_path=GRIPPER_PRIM_PATH
    )
    kit = omni.kit.app.get_app()
    for _ in range(120):
        kit.update()

    gripper_mount = _find_gripper_mount(stage)
    if not stage.GetPrimAtPath(UR3E_MOUNT_ABS).IsValid():
        raise RuntimeError(f"robot mount not found: {UR3E_MOUNT_ABS}")
    assembler = RobotAssembler()
    assembler.begin_assembly(
        stage,
        ROBOT_PRIM_PATH,
        UR3E_MOUNT_ABS,
        GRIPPER_PRIM_PATH,
        gripper_mount,
        ASSEMBLY_NAMESPACE,
        VARIANT_NAME,
    )
    assembler.assemble()
    assembler.finish_assemble()
    for _ in range(60):
        kit.update()


def spawn_assembled_robot() -> None:
    """Load the exported UR3e + Robotiq assembly without the GUI extension."""
    from isaaclab.sim.utils.prims import add_usd_reference
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("SimulationContext must exist before loading the robot")
    UsdGeom.Xform.Define(stage, "/World")
    asset_path = ASSET_DIR / "ur3e_robotiq_2f140_assembled_stage.usda"
    add_usd_reference(prim_path="/World", usd_path=str(asset_path), stage=stage)

    required_paths = (
        ROBOT_PRIM_PATH,
        GRIPPER_PRIM_PATH,
        f"{GRIPPER_PRIM_PATH}/finger_joint",
        f"{GRIPPER_PRIM_PATH}/robotiq_base_link/AssemblerFixedJoint",
    )
    missing = [path for path in required_paths if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Assembled robot asset is missing prims: {missing}")

def make_static_cuboid_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def make_table_cfg(prim_path: str, pos: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/{TABLE_USD_RELATIVE}", scale=TABLE_SCALE
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=pos, rot=quat_wxyz_to_isaac(TABLE_ROT)
        ),
    )


def make_target_cfg(name: str, info: dict, prim_path: str | None = None):
    rigid = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=2,
        max_linear_velocity=100.0,
        max_angular_velocity=100.0,
        max_depenetration_velocity=5.0,
        linear_damping=0.2,
        angular_damping=0.2,
    )
    scale = info.get("scale")
    scale_option = {} if scale is None else {"scale": (float(scale),) * 3}
    spawn = sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / info["collision_usd"]),
        **scale_option,
        rigid_props=rigid,
        mass_props=sim_utils.MassPropertiesCfg(mass=info["mass"]),
        collision_props=sim_utils.CollisionPropertiesCfg(
            torsional_patch_radius=0.05, min_torsional_patch_radius=0.05
        ),
    )
    return RigidObjectCfg(
        prim_path=prim_path or f"/World/{name.capitalize()}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=info["spawn_pos"],
            rot=quat_wxyz_to_isaac(info["spawn_rot"]),
        ),
    )


def make_robot_cfg(prim_path: str = ROBOT_PRIM_PATH) -> ArticulationCfg:
    """Build the one authoritative UR3e + Robotiq articulation config."""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS,
            rot=quat_wxyz_to_isaac(ROBOT_BASE_ROT),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                stiffness=10000.0,
                damping=500.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
            ),
            # Only finger_joint is actively commanded; the closed linkage
            # joints remain compliant and passive in the official physics USD.
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=11.25,
                damping=0.1,
                effort_limit_sim=10.0,
                velocity_limit_sim=1.0,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                stiffness=0.2,
                damping=0.001,
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_inner_finger_pad_joint",
                    ".*_outer_finger_joint",
                    "right_outer_knuckle_joint",
                ],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
            ),
        },
    )


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


def _target_cfg(name: str):
    return make_target_cfg(name, TARGETS[name])


@configclass
class SceneCfg(InteractiveSceneCfg):
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
    red_mug = _target_cfg("red_mug")
    spoon = _target_cfg("spoon")
    bowl = _target_cfg("bowl")
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
    profile: str | SceneProfile = "canonical",
) -> SceneCfg:
    """Compose the canonical physics scene for one named workflow."""
    resolved = get_scene_profile(profile)
    cfg = SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
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


# Compatibility export required by the active MultiView workstream. Canonical
# code imports presentation helpers directly from vla_sim.visuals.
from vla_sim.visuals import set_plastic_material  # noqa: E402,F401
