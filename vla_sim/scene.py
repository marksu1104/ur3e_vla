"""Canonical three-object scene and its USD presentation helpers."""

from __future__ import annotations

from pathlib import Path

import omni.kit.app
import omni.usd
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
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
)

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


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
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot_setup.assembler import RobotAssembler
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, "/World")
    asset_root = str(ISAAC_NUCLEUS_DIR)
    gripper_asset = gripper_usd_relative or GRIPPER_USD_RELATIVE
    add_reference_to_stage(
        usd_path=f"{asset_root}/{UR3E_USD_RELATIVE}", prim_path=ROBOT_PRIM_PATH
    )
    add_reference_to_stage(
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=TABLE_ROT),
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
            pos=info["spawn_pos"], rot=info["spawn_rot"]
        ),
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

    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS,
            rot=ROBOT_BASE_ROT,
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
    camera_yolo = CameraCfg(
        prim_path="/World/CameraYolo",
        update_period=0.0,
        height=STREAM_HEIGHT,
        width=STREAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=YOLO_CAMERA_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=YOLO_CAMERA_POS, rot=YOLO_CAMERA_ROT, convention="opengl"
        ),
    )
    camera_policy = CameraCfg(
        prim_path="/World/CameraPolicy",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA_MAIN_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_MAIN_POS, rot=CAMERA_MAIN_ROT, convention="opengl"
        ),
    )
    camera_wrist = CameraCfg(
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


def make_scene_cfg(
    *,
    num_envs: int = 1,
    env_spacing: float = 2.0,
    stream_width: int | None = None,
    stream_height: int | None = None,
) -> SceneCfg:
    """Instantiate the canonical scene with optional bridge stream dimensions."""
    cfg = SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    if stream_width is not None:
        cfg.camera_yolo.width = stream_width
    if stream_height is not None:
        cfg.camera_yolo.height = stream_height
    return cfg


def hide_markers(stage) -> None:
    """Keep placement markers hidden in every normal bridge state."""
    from pxr import UsdGeom

    for index in range(len(PLACE_POSITIONS)):
        prim = stage.GetPrimAtPath(f"/World/MarkerP{index}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def set_plastic_material(stage, prim_path: str, color: tuple[float, float, float]) -> None:
    """Apply one matte 3D-print-style material below ``prim_path``."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        log(f"Material skipped, prim not found: {prim_path}")
        return

    material_path = f"/World/Looks/{prim_path.strip('/').replace('/', '_')}_Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.62)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([color])
            UsdShade.MaterialBindingAPI(prim).Bind(material)


def set_marker_material(stage, prim_path: str, color: tuple[float, float, float]) -> None:
    """Apply a flat emissive material distinct from the object materials."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        log(f"Marker material skipped, prim not found: {prim_path}")
        return

    material_path = f"/World/Looks/{prim_path.strip('/').replace('/', '_')}_Marker"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(0.15 * channel for channel in color))
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([color])
            UsdShade.MaterialBindingAPI(prim).Bind(material)


def _smooth_meshes(stage, prim_path: str, iterations: int = 3) -> None:
    """Smooth a visual-only mesh without modifying collision geometry."""
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    root = stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not len(points) or not len(counts) or int(counts.sum()) != len(indices):
            continue

        faces = []
        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + int(count)]
            cursor += int(count)
            if len(face) >= 3:
                faces.append(face)
        if not faces:
            continue

        edges = np.concatenate(
            [np.column_stack((face, np.roll(face, -1))) for face in faces], axis=0
        )
        _, unique_indices, inverse = np.unique(
            np.round(points, decimals=6),
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        welded = points[unique_indices].copy()
        welded_edges = inverse[edges]
        welded_edges = welded_edges[welded_edges[:, 0] != welded_edges[:, 1]]
        src = np.concatenate((welded_edges[:, 0], welded_edges[:, 1]))
        dst = np.concatenate((welded_edges[:, 1], welded_edges[:, 0]))
        degree = np.bincount(src, minlength=len(welded)).astype(np.float64)
        movable = degree > 0
        for factor in (0.35, -0.36) * max(0, int(iterations)):
            neighbor_sum = np.zeros_like(welded)
            np.add.at(neighbor_sum, src, welded[dst])
            averages = np.zeros_like(welded)
            averages[movable] = neighbor_sum[movable] / degree[movable, None]
            welded[movable] += factor * (averages[movable] - welded[movable])
        points = welded[inverse]
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))

        normals = np.zeros_like(points)
        for face in faces:
            for index in range(1, len(face) - 1):
                tri = face[[0, index, index + 1]]
                normal = np.cross(
                    points[tri[1]] - points[tri[0]],
                    points[tri[2]] - points[tri[0]],
                )
                normals[tri] += normal
        _, normal_inverse = np.unique(
            np.round(points, decimals=6), axis=0, return_inverse=True
        )
        welded_normals = np.zeros((int(normal_inverse.max()) + 1, 3))
        np.add.at(welded_normals, normal_inverse, normals)
        normals = welded_normals[normal_inverse]
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals[valid] /= lengths[valid, None]
        normals[~valid] = (0.0, 0.0, 1.0)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if np.all(counts == 3):
            mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.loop)
            mesh.GetInterpolateBoundaryAttr().Set(UsdGeom.Tokens.edgeAndCorner)


def prepare_target_visuals(stage) -> None:
    """Build calibrated render visuals while retaining local collision meshes."""
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    for name, target in TARGETS.items():
        root_path = f"/World/{name.capitalize()}"
        if name == "spoon":
            set_plastic_material(stage, root_path, target["color"])
            continue

        visual_path = f"{root_path}/SmoothedVisual"
        add_reference_to_stage(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/{target['usd_relative']}",
            prim_path=visual_path,
        )
        root = stage.GetPrimAtPath(root_path)
        for prim in Usd.PrimRange(root):
            if not str(prim.GetPath()).startswith(visual_path):
                if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdGeom.Imageable(prim).MakeInvisible()
        set_plastic_material(stage, visual_path, target["color"])
        _smooth_meshes(stage, visual_path)
def apply_target_colors(stage, target_keys, root_prefix: str = "/World") -> None:
    """Apply canonical object materials under one scene or cloned env."""
    for name in target_keys:
        set_plastic_material(stage, f"{root_prefix}/{name.capitalize()}", TARGETS[name]["color"])

def bind_gripper_pad_visuals(stage) -> list[str]:
    """Make both inner-finger render meshes concrete and visible."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    material_path = "/World/Looks/RobotiqFingerPadsVisual"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.72, 0.72, 0.72)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    bound_paths = []
    for side in ("left", "right"):
        link_path = f"/World/Gripper/{side}_inner_finger"
        link = stage.GetPrimAtPath(link_path)
        if not link.IsValid():
            continue
        for part_name in ("Fingertip_01", "Finger4_01"):
            part = stage.GetPrimAtPath(f"{link_path}/{part_name}")
            if part.IsValid() and part.IsInstance():
                part.SetInstanceable(False)
        for prim in Usd.PrimRange(link):
            if prim == link or prim.IsA(UsdGeom.Gprim):
                api = (
                    UsdShade.MaterialBindingAPI(prim)
                    if prim.HasAPI(UsdShade.MaterialBindingAPI)
                    else UsdShade.MaterialBindingAPI.Apply(prim)
                )
                api.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose=UsdShade.Tokens.allPurpose,
                )
        bound_paths.append(link_path)
    return bound_paths


def configure_gripper_pads(stage) -> list[str]:
    """Apply high-friction physics and the missing pad render material."""
    from pxr import UsdShade

    material_path = "/World/PhysicsMaterials/RobotiqFingerPads"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    material_cfg.func(material_path, material_cfg)
    physics_material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    bound_paths = bind_gripper_pad_visuals(stage)
    for link_path in bound_paths:
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(link_path)).Bind(
            physics_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
    if len(bound_paths) != 2:
        raise RuntimeError(f"Expected two Robotiq inner-finger links, found {bound_paths}")
    return bound_paths
