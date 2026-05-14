"""Isaac Lab scene configuration and asset assembly helpers."""

import omni.usd
import omni.kit.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg

from vla_sim.isaac_app import log
from vla_sim.config import (
    # USD paths
    UR3E_USD_RELATIVE,
    GRIPPER_USD_RELATIVE,
    # Stage prims
    ROBOT_PRIM_PATH,
    GRIPPER_PRIM_PATH,
    UR3E_MOUNT_ABS,
    GRIPPER_MOUNT_REL_CANDIDATES,
    ASSEMBLY_NAMESPACE,
    VARIANT_NAME,
    # Robot
    ROBOT_BASE_POS,
    ROBOT_BASE_ROT,
    # Camera resolutions
    CAMERA_WIDTH, CAMERA_HEIGHT,
    WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT,
    ORBIT_CAMERA_WIDTH, ORBIT_CAMERA_HEIGHT,
    # Targets / gripper
    TARGETS,
    GRIPPER_MIMIC_MAP,
    CAMERA_MAIN_POS, CAMERA_MAIN_ROT, CAMERA_MAIN_FOCAL,
)

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"

# Resolved asset URLs (after carb settings hack in boot.py)
UR3E_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/{UR3E_USD_RELATIVE}"
GRIPPER_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/{GRIPPER_USD_RELATIVE}"


def enable_extensions():
    mgr = omni.kit.app.get_app().get_extension_manager()
    ok = mgr.set_extension_enabled_immediate("isaacsim.robot_setup.assembler", True)
    log(f"robot_setup.assembler enabled = {ok}")


def find_gripper_mount_abs(stage):
    """Find the gripper base-link prim used as the assembly mount."""
    from pxr import Usd

    for rel in GRIPPER_MOUNT_REL_CANDIDATES:
        abs_path = f"{GRIPPER_PRIM_PATH}/{rel}"
        if stage.GetPrimAtPath(abs_path).IsValid():
            log(f"Found gripper mount at: {abs_path}")
            return abs_path

    gripper_prim = stage.GetPrimAtPath(GRIPPER_PRIM_PATH)
    for p in Usd.PrimRange(gripper_prim):
        if "base_link" in p.GetName().lower():
            log(f"Fallback mount found: {p.GetPath()}")
            return str(p.GetPath())

    raise RuntimeError("Could not locate gripper base link for assembly")


def spawn_raw_and_assemble():
    """Load UR3e and Robotiq USD assets, then assemble them into one robot."""
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot_setup.assembler import RobotAssembler
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, "/World")

    import omni.client as _client
    log(f"DEBUG: ISAAC_NUCLEUS_DIR = {ISAAC_NUCLEUS_DIR}")
    log(f"DEBUG: UR3E_USD_PATH    = {UR3E_USD_PATH}")
    log(f"DEBUG: GRIPPER_USD_PATH = {GRIPPER_USD_PATH}")
    _r, _ = _client.stat(UR3E_USD_PATH)
    log(f"DEBUG: UR3e stat result    = {_r}")
    _r, _ = _client.stat(GRIPPER_USD_PATH)
    log(f"DEBUG: Gripper stat result = {_r}")

    log(f"Loading UR3e at {ROBOT_PRIM_PATH}...")
    add_reference_to_stage(usd_path=UR3E_USD_PATH, prim_path=ROBOT_PRIM_PATH)
    log("UR3e add_reference_to_stage returned OK")

    log(f"Loading gripper at {GRIPPER_PRIM_PATH}...")
    add_reference_to_stage(usd_path=GRIPPER_USD_PATH, prim_path=GRIPPER_PRIM_PATH)
    log("Gripper add_reference_to_stage returned OK")

    kit = omni.kit.app.get_app()
    log("Calling kit.update() x120...")
    for i in range(120):
        kit.update()
        if i % 30 == 0:
            log(f"  kit update {i}/120")
    log("kit.update() x120 done")

    log("Looking up mount frames...")
    try:
        log("=== Stage prims under /World ===")
        world_prim = stage.GetPrimAtPath("/World")
        log(f"World prim valid: {world_prim.IsValid()}")
        if world_prim.IsValid():
            for p in world_prim.GetAllChildren():
                log(f"  {p.GetPath()} (type={p.GetTypeName()})")
                try:
                    children = list(p.GetAllChildren())
                    log(f"    has {len(children)} children")
                    for c in children[:5]:
                        log(f"    -> {c.GetName()} ({c.GetTypeName()})")
                except Exception as e:
                    log(f"    GetAllChildren error: {e}")
    except Exception as e:
        log(f"Stage dump error: {e}")
        import traceback
        log(traceback.format_exc())

    log(f"=== Looking for {UR3E_MOUNT_ABS} ===")
    try:
        ur_mount = stage.GetPrimAtPath(UR3E_MOUNT_ABS)
        log(f"UR3e mount: valid={ur_mount.IsValid()}, path={UR3E_MOUNT_ABS}")
    except Exception as e:
        log(f"UR3e mount lookup error: {e}")
        raise

    log("=== Looking for gripper mount ===")
    try:
        gripper_mount_abs = find_gripper_mount_abs(stage)
        log(f"Gripper mount found: {gripper_mount_abs}")
    except Exception as e:
        log(f"Gripper mount lookup error: {e}")
        raise

    gr_mount = stage.GetPrimAtPath(gripper_mount_abs)
    if not ur_mount.IsValid() or not gr_mount.IsValid():
        raise RuntimeError("Mount prims not valid.")

    assembler = RobotAssembler()
    assembler.begin_assembly(
        stage,
        ROBOT_PRIM_PATH, UR3E_MOUNT_ABS,
        GRIPPER_PRIM_PATH, gripper_mount_abs,
        ASSEMBLY_NAMESPACE, VARIANT_NAME,
    )
    assembler.assemble()
    assembler.finish_assemble()

    for _ in range(60):
        kit.update()


def make_target_cfg(name: str, info: dict) -> RigidObjectCfg:
    """Create the physical cuboid proxy for a YCB target object."""
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=2,
        max_linear_velocity=100.0,
        max_angular_velocity=100.0,
        max_depenetration_velocity=5.0,
        linear_damping=0.2,
        angular_damping=0.2,
    )
    spawn_cfg = sim_utils.CuboidCfg(
        size=info["size"],
        rigid_props=rigid_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=info["mass"]),
        collision_props=sim_utils.CollisionPropertiesCfg(
            torsional_patch_radius=0.05,
            min_torsional_patch_radius=0.05,
        ),
    )
    return RigidObjectCfg(
        prim_path=f"/World/{name.capitalize()}",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=info["spawn_pos"],
            rot=info.get("spawn_rot", (1.0, 0.0, 0.0, 0.0)),
        ),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )

    robot = ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS, rot=ROBOT_BASE_ROT,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
                ],
                stiffness=10000.0,
                damping=500.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=list(GRIPPER_MIMIC_MAP.keys()),
                stiffness=100.0,
                damping=12.0,
                effort_limit_sim=20.0,
            ),
        },
    )

    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            scale=(1.8, 0.8, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.17, 0.17, 1.05),
            rot=(0.0, 0.0, 0.0, 1.0)
            ),
    )
    mat = AssetBaseCfg(
        prim_path="/World/Mat",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.6, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.46, 0.17, 1.05),
            rot=(0.0, 0.0, 0.0, 1.0)
        ),
    )
    banana = make_target_cfg("banana", TARGETS["banana"])
    mug = make_target_cfg("mug", TARGETS["mug"])

    camera_main = CameraCfg(
        prim_path="/World/CameraMain",
        update_period=0.0,
        height=CAMERA_HEIGHT, width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA_MAIN_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_MAIN_POS,
            rot=CAMERA_MAIN_ROT,
            convention="opengl",
        ),
    )   

    camera_wrist = CameraCfg(
        prim_path="/World/Robot/wrist_3_link/CameraWrist",
        update_period=0.0,
        height=WRIST_CAMERA_HEIGHT, width=WRIST_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.12),
            rot=(0.0, 1.0, 0.0, 0.0),
            convention="opengl",
        ),
    )

    camera_orbit = CameraCfg(
        prim_path="/World/CameraOrbit",
        update_period=0.0,
        height=ORBIT_CAMERA_HEIGHT, width=ORBIT_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
        offset=CameraCfg.OffsetCfg(
            pos=(1.3, 0.15, 1.55),
            rot=(0.354, 0.146, 0.354, 0.854),
            convention="opengl",
        ),
    )
    ee_frame = FrameTransformerCfg(
        prim_path=f"{ROBOT_PRIM_PATH}/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{ROBOT_PRIM_PATH}/wrist_3_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.18),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )


def hide_proxy_meshes(stage, target_keys):
    """Hide cuboid proxy meshes after visual YCB meshes are attached."""
    from pxr import UsdGeom

    for target_key in target_keys:
        for sub in ("Visuals/geometry/mesh", "geometry/mesh"):
            mesh_path = f"/World/{target_key.capitalize()}/{sub}"
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if mesh_prim.IsValid():
                UsdGeom.Imageable(mesh_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                log(f"Hid proxy mesh: {mesh_path}")
                break
