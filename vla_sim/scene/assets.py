"""Robot, furniture, and movable-object definitions for the base scene."""

from __future__ import annotations

from pathlib import Path

import omni.kit.app
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from vla_sim.config import (
    ASSEMBLY_NAMESPACE,
    GRIPPER_MOUNT_REL_CANDIDATES,
    GRIPPER_PRIM_PATH,
    GRIPPER_USD_RELATIVE,
    ROBOT_BASE_POS,
    ROBOT_BASE_ROT,
    ROBOT_PRIM_PATH,
    TABLE_ROT,
    TABLE_SCALE,
    TABLE_USD_RELATIVE,
    TARGETS,
    UR3E_MOUNT_ABS,
    UR3E_USD_RELATIVE,
    VARIANT_NAME,
)
from vla_sim.isaac_app import log


ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"
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
    """Enable the one Isaac extension needed to assemble a raw robot asset."""
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


def make_target_cfg(name: str, info: dict, prim_path: str | None = None) -> RigidObjectCfg:
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


def make_base_target_cfg(name: str) -> RigidObjectCfg:
    """Return the configured movable object by its stable task name."""
    return make_target_cfg(name, TARGETS[name])
