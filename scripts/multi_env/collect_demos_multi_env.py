"""Collect scripted UR3e grasp demonstrations with multiple IsaacLab envs.

This collector uses the exported assembled UR3e+Robotiq USD so IsaacLab can
instantiate one vectorized articulation per environment. It intentionally keeps
feature scope smaller than collect_demos.py: no runtime RobotAssembler.
The H5 schema is the same as the single-env collector. Optional video recording
is intended for short test previews only.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_parser.add_argument("--episodes", type=int, default=8, help="Number of successful episodes to collect.")
_parser.add_argument("--num-envs", type=int, default=4, help="Number of IsaacLab envs to run in parallel.")
_parser.add_argument("--output-dir", type=str, default="outputs/test/h5_multi_env")
_parser.add_argument("--max-episodes-tried", type=int, default=0)
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--overwrite", action="store_true")
_parser.add_argument("--no-save-h5", action="store_true")
_parser.add_argument("--randomize-pos", action=argparse.BooleanOptionalAction, default=True)
_parser.add_argument("--randomize-rot", action=argparse.BooleanOptionalAction, default=True)
_parser.add_argument("--show-gui", action="store_true")
_parser.add_argument("--record-video", action="store_true", help="Record a short test preview MP4 from one env camera.")
_parser.add_argument("--video-path", type=str, default="", help="MP4 path for --record-video. Defaults under output dir.")
_parser.add_argument("--video-camera", type=str, default="camera_main", choices=["camera_main", "camera_wrist"])
_parser.add_argument("--video-env", type=int, default=-1, help="Env index to record; -1 tiles all env camera frames side by side.")
_parser.add_argument("--video-fps", type=float, default=30.0)
_parser.add_argument("--video-every-n-steps", type=int, default=2)
_parser.add_argument(
    "--asset",
    type=Path,
    default=PROJECT_ROOT / "assets" / "ur3e_robotiq_2f140_assembled_stage.usda",
    help="Assembled UR3e+Robotiq stage USD.",
)
_extra_args, _ = _parser.parse_known_args()

if not _extra_args.show_gui and "--headless" not in sys.argv:
    sys.argv.append("--headless")

from vla_sim.isaac_app import boot_app, args_cli, log

app = boot_app()
_app_closed = False


def close_app_once():
    global _app_closed
    if _app_closed:
        return
    _app_closed = True
    try:
        app.close(wait_for_replicator=False)
    except TypeError:
        app.close()


import random

import omni.usd
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from vla_sim.actions import PoseTrajectoryPlayer, compute_action_from_ee_poses
from vla_sim.config import (
    BACKDROP_BACK_POS,
    BACKDROP_BACK_SIZE,
    BACKDROP_SIDE_POS,
    BACKDROP_SIDE_SIZE,
    CAMERA_HEIGHT,
    CAMERA_MAIN_FOCAL,
    CAMERA_MAIN_POS,
    CAMERA_MAIN_ROT,
    CAMERA_WIDTH,
    EE_BODY_NAME,
    EE_ORIENT_DOWN,
    GRIPPER_CLOSE,
    GRIPPER_MIMIC_MAP,
    GRIPPER_OPEN,
    HOME_POS,
    HOME_Q,
    PHYSICS_DT,
    ROBOT_BASE_POS,
    ROBOT_BASE_ROT,
    TABLE_A_POS,
    TABLE_B_POS,
    TABLE_MAT_A_POS,
    TABLE_MAT_B_POS,
    TABLE_MAT_SIZE,
    TABLE_ROT,
    TABLE_SCALE,
    TABLE_USD_RELATIVE,
    TARGETS,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_WIDTH,
)
from vla_sim.data_collector import EpisodeBuffer, append_episode_h5
from vla_sim.video import VideoRecorder
from vla_sim.demo_planning import (
    detect_success,
    randomize_target_pose,
    sample_grasp_parameters,
    sample_grasp_quat,
    sample_scene_object_poses,
)

random.seed(_extra_args.seed)
np.random.seed(_extra_args.seed)

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ASSEMBLED_PRIM = "{ENV_REGEX_NS}/Assembled"
ROBOT_PRIM = f"{ASSEMBLED_PRIM}/Robot"


def _asset_path() -> str:
    return str(_extra_args.asset.expanduser().resolve())


def _make_static_cuboid_cfg(prim_path: str, size: tuple, pos: tuple) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _make_table_cfg(prim_path: str, pos: tuple) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/{TABLE_USD_RELATIVE}",
            scale=TABLE_SCALE,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=TABLE_ROT),
    )



def _set_display_color_recursive(stage, prim_path: str, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return

    safe_name = prim_path.strip("/").replace("/", "_")
    material = UsdShade.Material.Define(stage, f"/World/Looks/{safe_name}_Material")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{safe_name}_Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    for prim in Usd.PrimRange(root):
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDisplayColorAttr([color])
            UsdShade.MaterialBindingAPI(prim).Bind(material)


def _attach_target_visuals_for_envs(stage, num_envs: int) -> None:
    """Match the single-env visual setup for every cloned env target."""
    from isaacsim.core.utils.stage import add_reference_to_stage

    log("Attaching target visuals for multi-env scene...")
    for env_id in range(num_envs):
        env_prefix = f"/World/envs/env_{env_id}"
        for target_key, info in TARGETS.items():
            target_root = f"{env_prefix}/{target_key.capitalize()}"
            if info.get("collision_usd"):
                log(f"  env{env_id} {target_key}: using local collision USD, skip visual attach")
                continue
            usd_abs = f"{ISAAC_NUCLEUS_DIR}/{info['usd_relative']}"
            visual_path = f"{target_root}/Visuals"
            try:
                if not stage.GetPrimAtPath(visual_path).IsValid():
                    add_reference_to_stage(usd_path=usd_abs, prim_path=visual_path)
            except Exception as exc:
                log(f"  attach failed (env{env_id} {target_key}): {exc}")


def _apply_target_colors_for_envs(stage, num_envs: int) -> None:
    """Apply the same red/blue mug colors and hide banana proxy meshes per env."""
    from pxr import UsdGeom

    for env_id in range(num_envs):
        env_prefix = f"/World/envs/env_{env_id}"
        for target_key, info in TARGETS.items():
            target_root = f"{env_prefix}/{target_key.capitalize()}"
            if "color" in info:
                _set_display_color_recursive(stage, target_root, info["color"])
            if not info.get("collision_usd"):
                for sub in ("Visuals/geometry/mesh", "geometry/mesh"):
                    mesh_path = f"{target_root}/{sub}"
                    mesh_prim = stage.GetPrimAtPath(mesh_path)
                    if mesh_prim.IsValid():
                        UsdGeom.Imageable(mesh_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                        log(f"Hid proxy mesh: {mesh_path}")
                        break


def _tile_env_rgb(rgb_all: np.ndarray) -> np.ndarray:
    frames = np.asarray(rgb_all)[..., :3].astype(np.uint8)
    if frames.ndim == 3:
        return frames
    if frames.ndim != 4:
        raise ValueError(f"expected camera rgb shape (N,H,W,C), got {frames.shape}")
    return np.concatenate([frames[i] for i in range(frames.shape[0])], axis=1)


def _make_target_cfg(name: str, info: dict) -> RigidObjectCfg:
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
    mass_props = sim_utils.MassPropertiesCfg(mass=info["mass"])
    collision_props = sim_utils.CollisionPropertiesCfg(
        torsional_patch_radius=0.05,
        min_torsional_patch_radius=0.05,
    )
    if info.get("collision_usd"):
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(PROJECT_ROOT / "assets" / info["collision_usd"]),
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=collision_props,
        )
    else:
        spawn_cfg = sim_utils.CuboidCfg(
            size=info["size"],
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=collision_props,
        )
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=info["spawn_pos"],
            rot=info.get("spawn_rot", (1.0, 0.0, 0.0, 0.0)),
        ),
    )


def _make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=ROBOT_BASE_POS, rot=ROBOT_BASE_ROT),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                stiffness=10000.0,
                damping=500.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=list(GRIPPER_MIMIC_MAP.keys()),
                stiffness=60.0,
                damping=8.0,
                effort_limit_sim=8.0,
            ),
        },
    )


@configclass
class MultiEnvSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )
    assembled = AssetBaseCfg(
        prim_path=ASSEMBLED_PRIM,
        spawn=sim_utils.UsdFileCfg(usd_path=_asset_path()),
    )
    robot = _make_robot_cfg()

    table_a = _make_table_cfg("{ENV_REGEX_NS}/TableA", TABLE_A_POS)
    table_b = _make_table_cfg("{ENV_REGEX_NS}/TableB", TABLE_B_POS)
    mat_a = _make_static_cuboid_cfg("{ENV_REGEX_NS}/MatA", TABLE_MAT_SIZE, TABLE_MAT_A_POS)
    mat_b = _make_static_cuboid_cfg("{ENV_REGEX_NS}/MatB", TABLE_MAT_SIZE, TABLE_MAT_B_POS)
    backdrop_back = _make_static_cuboid_cfg("{ENV_REGEX_NS}/BackdropBack", BACKDROP_BACK_SIZE, BACKDROP_BACK_POS)
    backdrop_side = _make_static_cuboid_cfg("{ENV_REGEX_NS}/BackdropSide", BACKDROP_SIDE_SIZE, BACKDROP_SIDE_POS)

    banana = _make_target_cfg("banana", TARGETS["banana"])
    red_mug = _make_target_cfg("red_mug", TARGETS["red_mug"])
    blue_mug = _make_target_cfg("blue_mug", TARGETS["blue_mug"])

    camera_main = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraMain",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA_MAIN_FOCAL),
        offset=CameraCfg.OffsetCfg(pos=CAMERA_MAIN_POS, rot=CAMERA_MAIN_ROT, convention="opengl"),
    )
    camera_wrist = CameraCfg(
        prim_path=f"{ROBOT_PRIM}/wrist_3_link/CameraWrist",
        update_period=0.0,
        height=WRIST_CAMERA_HEIGHT,
        width=WRIST_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.12), rot=(0.0, 1.0, 0.0, 0.0), convention="opengl"),
    )


def _shape_of(value):
    try:
        return tuple(value.shape)
    except Exception:
        return None


def _make_trajectory(target_info: dict, target_resting: np.ndarray, spawn_rot: tuple, rng: np.random.Generator, env_origin: np.ndarray):
    tx, ty, tz = target_resting.tolist()
    home_pos_w = tuple((np.asarray(HOME_POS, dtype=np.float32) + env_origin.astype(np.float32)).tolist())
    grasp_params = sample_grasp_parameters(target_info, rng)
    gx = tx + grasp_params["x_nudge"]
    gy = ty + grasp_params["y_nudge"]
    hover_z = tz + grasp_params["hover_z"]
    grasp_z = tz + grasp_params["grasp_z"]

    hover_pos = (gx, gy, hover_z)
    pre_grasp_pos = (gx, gy, min(grasp_z + 0.055, hover_z))
    grasp_pos = (gx, gy, grasp_z)
    lift_start_pos = (gx, gy, min(grasp_z + 0.025, hover_z))
    grasp_quat = sample_grasp_quat(target_info, spawn_rot, rng)

    hover_dt = float(rng.uniform(3.2, 3.8))
    pre_grasp_dt = float(rng.uniform(1.9, 2.4))
    descend_dt = float(rng.uniform(1.0, 1.4))
    close_dt = float(rng.uniform(0.35, 0.55))
    lift_start_dt = float(rng.uniform(0.45, 0.70))
    lift_dt = float(rng.uniform(2.1, 2.6))
    carry_dt = float(rng.uniform(2.7, 3.3))
    release_dt = float(rng.uniform(1.6, 2.1))

    trajectory = [
        (0.0, home_pos_w, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (hover_dt, hover_pos, grasp_quat, GRIPPER_OPEN),
        (pre_grasp_dt, pre_grasp_pos, grasp_quat, GRIPPER_OPEN),
        (descend_dt, grasp_pos, grasp_quat, GRIPPER_OPEN),
        (close_dt, grasp_pos, grasp_quat, GRIPPER_CLOSE),
        (lift_start_dt, lift_start_pos, grasp_quat, GRIPPER_CLOSE),
        (lift_dt, hover_pos, grasp_quat, GRIPPER_CLOSE),
        (carry_dt, home_pos_w, grasp_quat, GRIPPER_CLOSE),
        (release_dt, home_pos_w, grasp_quat, GRIPPER_OPEN),
    ]
    meta = {
        "grasp_params": {k: float(v) for k, v in grasp_params.items()},
        "grasp_quat": list(map(float, grasp_quat)),
        "place_pos": list(map(float, home_pos_w)),
        "durations": {
            "hover": hover_dt,
            "pre_grasp": pre_grasp_dt,
            "descend": descend_dt,
            "close": close_dt,
            "lift_start": lift_start_dt,
            "lift": lift_dt,
            "carry": carry_dt,
            "release": release_dt,
        },
    }
    return trajectory, meta


def _success_for_env(target_obj, env_id: int, ee_pos: np.ndarray, target_initial_z: float, gripper_q: float, best_lift: float, place: np.ndarray, env_origin: np.ndarray):
    obj_pos = target_obj.data.root_pos_w[env_id].detach().cpu().numpy()
    ee_local = ee_pos - env_origin
    obj_lift_height = float(obj_pos[2] - target_initial_z)
    place_dist = float(np.linalg.norm(obj_pos[:2] - place[:2]))
    obj_lifted = best_lift > 0.04
    obj_at_place = place_dist < 0.10
    ee_safe = bool((1.05 < ee_local[2] < 1.7) and (-0.5 < ee_local[0] < 0.7))
    return obj_lifted and obj_at_place, {
        "obj_lifted": bool(obj_lifted),
        "obj_at_place": bool(obj_at_place),
        "ee_safe": ee_safe,
        "obj_pos_final": obj_pos.tolist(),
        "ee_pos_final": ee_pos.tolist(),
        "obj_lift_height": obj_lift_height,
        "best_lift_height": float(best_lift),
        "obj_place_xy_dist": place_dist,
        "place_pos": place.tolist(),
        "place_xy_threshold": 0.10,
        "gripper_closed": bool(gripper_q > 0.3),
    }


def _setup_batch(scene, sim, robot, sim_dt, rngs, target_name, target_info, arm_ids_t, finger_ids_t, gripper_signs, gripper_lows, gripper_highs):
    device = str(sim.device)
    num_envs = scene.num_envs
    origins = scene.env_origins.to(device=device, dtype=torch.float32)

    root_pose = torch.zeros((num_envs, 7), device=device, dtype=torch.float32)
    root_pose[:, :3] = torch.tensor(ROBOT_BASE_POS, device=device, dtype=torch.float32).unsqueeze(0) + origins
    root_pose[:, 3:] = torch.tensor(ROBOT_BASE_ROT, device=device, dtype=torch.float32).unsqueeze(0)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=device))

    home_q = torch.tensor(HOME_Q, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    robot.write_joint_state_to_sim(home_q, torch.zeros_like(home_q), joint_ids=arm_ids_t)

    open_cmd = torch.clamp(
        (gripper_signs * GRIPPER_OPEN).unsqueeze(0).repeat(num_envs, 1),
        gripper_lows.unsqueeze(0),
        gripper_highs.unsqueeze(0),
    )
    robot.set_joint_position_target(home_q, joint_ids=arm_ids_t)
    robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)

    origins_np = origins.detach().cpu().numpy()
    sampled_by_env = []
    for env_id in range(num_envs):
        poses = sample_scene_object_poses(
            TARGETS,
            rngs[env_id],
            randomize_pos=_extra_args.randomize_pos,
            randomize_rot=_extra_args.randomize_rot,
            place_pos=HOME_POS,
            log_fn=None,
        )
        if target_name not in poses:
            poses[target_name] = randomize_target_pose(
                target_info,
                rngs[env_id],
                randomize_pos=_extra_args.randomize_pos,
                randomize_rot=_extra_args.randomize_rot,
            )
        sampled_by_env.append(poses)

    for object_name, object_info in TARGETS.items():
        obj = scene[object_name]
        pose = torch.zeros((num_envs, 7), device=device, dtype=torch.float32)
        vel = torch.zeros((num_envs, 6), device=device, dtype=torch.float32)
        for env_id in range(num_envs):
            if object_name not in sampled_by_env[env_id]:
                sampled_by_env[env_id][object_name] = randomize_target_pose(
                    object_info,
                    rngs[env_id],
                    randomize_pos=_extra_args.randomize_pos,
                    randomize_rot=_extra_args.randomize_rot,
                )
            pos, rot = sampled_by_env[env_id][object_name]
            pose[env_id, :3] = torch.tensor(pos, device=device, dtype=torch.float32) + origins[env_id]
            pose[env_id, 3:] = torch.tensor(rot, device=device, dtype=torch.float32)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(vel)

    for _ in range(60):
        robot.set_joint_position_target(home_q, joint_ids=arm_ids_t)
        robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    target_obj = scene[target_name]
    target_resting = target_obj.data.root_pos_w.detach().cpu().numpy()
    players = []
    traj_meta = []
    for env_id in range(num_envs):
        _pos, spawn_rot = sampled_by_env[env_id][target_name]
        trajectory, meta = _make_trajectory(target_info, target_resting[env_id], spawn_rot, rngs[env_id], origins_np[env_id])
        players.append(PoseTrajectoryPlayer(trajectory, device=device))
        meta["grasp_pos"] = list(map(float, trajectory[3][1]))
        meta["hover_pos"] = list(map(float, trajectory[1][1]))
        meta["scene_object_poses"] = {
            name: {"pos": list(map(float, pose[0])), "rot": list(map(float, pose[1]))}
            for name, pose in sampled_by_env[env_id].items()
        }
        traj_meta.append(meta)
    place_positions = np.asarray([meta['place_pos'] for meta in traj_meta], dtype=np.float32)
    return players, traj_meta, target_resting[:, 2].copy(), place_positions, origins_np.copy()


def _run_batch(sim, scene, robot, ik, sim_dt, arm_ids_t, finger_ids_t, gripper_signs, gripper_lows, gripper_highs, ee_body_idx, ee_jac_idx, target_name, target_info, rngs, instruction, video_recorder=None, video_camera="camera_main", video_env=0, video_every_n_steps=2):
    device = str(sim.device)
    num_envs = scene.num_envs
    target_obj = scene[target_name]
    players, traj_meta, target_initial_z, place_positions, origins_np = _setup_batch(
        scene, sim, robot, sim_dt, rngs, target_name, target_info,
        arm_ids_t, finger_ids_t, gripper_signs, gripper_lows, gripper_highs,
    )

    buffers = [EpisodeBuffer([], [], [], [], [], [], []) for _ in range(num_envs)]
    prev_ee_pos = [None] * num_envs
    prev_ee_quat = [None] * num_envs
    last_grip = np.full(num_envs, GRIPPER_OPEN, dtype=np.float32)
    finished_at = np.full(num_envs, -1, dtype=np.int32)
    best_lift = np.zeros(num_envs, dtype=np.float32)

    t_sim = 0.0
    step = 0
    record_every = 12
    max_steps = 1800

    while step < max_steps:
        samples = [player.sample(t_sim) for player in players]
        tgt_pos_w = torch.stack([s[0] for s in samples], dim=0)
        tgt_quat_w = torch.stack([s[1] for s in samples], dim=0)
        grip_target = np.asarray([s[2] for s in samples], dtype=np.float32)
        finished = np.asarray([s[3] for s in samples], dtype=bool)

        for env_id, done in enumerate(finished):
            if done and finished_at[env_id] < 0:
                finished_at[env_id] = step
        if np.all(finished_at >= 0) and np.all(step - finished_at > 60):
            break

        root_pos = robot.data.root_state_w[:, :3]
        ik.set_command(torch.cat([tgt_pos_w - root_pos, tgt_quat_w], dim=-1))

        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
        ee_pos_b = ee_pose_w[:, :3] - root_pos
        ee_quat_b = ee_pose_w[:, 3:]
        q_current = robot.data.joint_pos[:, arm_ids_t]
        jac = robot.root_physx_view.get_jacobians()[:, ee_jac_idx, :, :][:, :, arm_ids_t]
        q_target = ik.compute(ee_pos_b, ee_quat_b, jac, q_current)
        robot.set_joint_position_target(q_target, joint_ids=arm_ids_t)

        delta = np.clip(grip_target - last_grip, -0.008, 0.008)
        last_grip = np.clip(last_grip + delta, GRIPPER_OPEN, GRIPPER_CLOSE)
        grip_tensor = torch.tensor(last_grip, dtype=torch.float32, device=device).unsqueeze(1)
        finger_cmd = torch.clamp(
            gripper_signs.unsqueeze(0) * grip_tensor,
            gripper_lows.unsqueeze(0),
            gripper_highs.unsqueeze(0),
        )
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids_t)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        obj_pos = target_obj.data.root_pos_w.detach().cpu().numpy()
        best_lift = np.maximum(best_lift, obj_pos[:, 2] - target_initial_z)

        if video_recorder is not None and step % max(1, video_every_n_steps) == 0:
            try:
                rgb_all = scene[video_camera].data.output["rgb"].detach().cpu().numpy().astype(np.uint8)
                if video_env < 0:
                    video_rgb = _tile_env_rgb(rgb_all)
                else:
                    video_rgb = rgb_all[video_env, ..., :3]
                video_recorder.write_rgb(video_rgb[..., :3])
            except Exception as exc:
                log(f"video frame failed: {exc}")

        if step % record_every == 0:
            ee_pose_now_all = robot.data.body_state_w[:, ee_body_idx, :7].detach().cpu().numpy()
            joint_now_all = robot.data.joint_pos[:, arm_ids_t].detach().cpu().numpy()
            try:
                main_rgb_all = scene["camera_main"].data.output["rgb"].detach().cpu().numpy().astype(np.uint8)
            except Exception as exc:
                log(f"camera_main read failed: {exc}")
                main_rgb_all = np.zeros((num_envs, CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            try:
                wrist_rgb_all = scene["camera_wrist"].data.output["rgb"].detach().cpu().numpy().astype(np.uint8)
            except Exception:
                wrist_rgb_all = np.zeros((num_envs, WRIST_CAMERA_HEIGHT, WRIST_CAMERA_WIDTH, 3), dtype=np.uint8)

            for env_id in range(num_envs):
                ee_pose_now = ee_pose_now_all[env_id]
                grip_binary = 1.0 if last_grip[env_id] > ((GRIPPER_OPEN + GRIPPER_CLOSE) * 0.5) else 0.0
                if prev_ee_pos[env_id] is None:
                    action_7d = np.zeros(7, dtype=np.float32)
                    action_7d[6] = grip_binary
                else:
                    action_7d = compute_action_from_ee_poses(
                        prev_ee_pos[env_id], prev_ee_quat[env_id],
                        ee_pose_now[:3], ee_pose_now[3:],
                        gripper_target=grip_binary,
                    )
                buffers[env_id].main_images.append(main_rgb_all[env_id, ..., :3])
                buffers[env_id].wrist_images.append(wrist_rgb_all[env_id, ..., :3])
                buffers[env_id].ee_poses.append([float(x) for x in ee_pose_now.tolist()])
                buffers[env_id].joint_positions.append(joint_now_all[env_id].tolist())
                buffers[env_id].gripper_states.append(grip_binary)
                buffers[env_id].actions_7d.append(action_7d.tolist())
                buffers[env_id].timestamps.append(t_sim)
                prev_ee_pos[env_id] = ee_pose_now[:3].copy()
                prev_ee_quat[env_id] = ee_pose_now[3:].copy()

        t_sim += sim_dt
        step += 1

    results = []
    ee_pos_final_all = robot.data.body_state_w[:, ee_body_idx, :3].detach().cpu().numpy()
    joint_final_all = robot.data.joint_pos[:, finger_ids_t].detach().cpu().numpy()
    for env_id in range(num_envs):
        success, diag = _success_for_env(
            target_obj,
            env_id,
            ee_pos_final_all[env_id],
            float(target_initial_z[env_id]),
            float(last_grip[env_id]),
            float(best_lift[env_id]),
            place_positions[env_id],
            origins_np[env_id],
        )
        grasp_pos = np.asarray(traj_meta[env_id]["grasp_pos"], dtype=np.float32)
        final_obj = np.asarray(diag["obj_pos_final"], dtype=np.float32)
        diag["num_steps_recorded"] = len(buffers[env_id].main_images)
        diag["env_id"] = int(env_id)
        diag["ee_to_grasp_dist_final"] = float(np.linalg.norm(ee_pos_final_all[env_id] - grasp_pos))
        diag["ee_to_obj_xy_final"] = float(np.linalg.norm(ee_pos_final_all[env_id][:2] - final_obj[:2]))
        diag["gripper_cmd_final"] = float(last_grip[env_id])
        diag["gripper_joint_pos_final"] = joint_final_all[env_id].tolist()
        diag.update(traj_meta[env_id])
        results.append((success, buffers[env_id] if success else None, diag))
    return results


def main():
    target_name = args_cli.target
    if target_name not in ("red_mug", "blue_mug"):
        raise ValueError("collect_demos_multi_env.py currently supports red_mug and blue_mug only")
    target_info = TARGETS[target_name]
    instruction = f"pick up the {target_name.replace('_', ' ')}"
    asset_path = _extra_args.asset.expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Assembled asset not found: {asset_path}")

    out_dir_rel = Path(_extra_args.output_dir)
    out_dir = out_dir_rel if out_dir_rel.is_absolute() else PROJECT_ROOT / out_dir_rel
    if out_dir.name != target_name:
        out_dir = out_dir / target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "demos.h5"
    if _extra_args.overwrite and h5_path.exists() and not _extra_args.no_save_h5:
        h5_path.unlink()
        log(f"Removed existing dataset: {h5_path}")

    num_envs = max(1, int(_extra_args.num_envs))
    n_target = max(1, int(_extra_args.episodes))
    max_tried = _extra_args.max_episodes_tried or max(n_target * 3, n_target + 10)
    rngs = [np.random.default_rng(_extra_args.seed + env_id * 1009) for env_id in range(num_envs)]

    log("Multi-env scripted collection")
    log(f"Target object   : {target_name}")
    log(f"Instruction     : '{instruction}'")
    log(f"Asset           : {asset_path}")
    log(f"num_envs        : {num_envs}")
    log(f"Target episodes : {n_target}")
    log(f"Max tried       : {max_tried}")
    log(f"Output          : {out_dir}")
    log(f"Save H5         : {not _extra_args.no_save_h5}")

    video_recorder = None
    video_path = None
    if _extra_args.record_video:
        video_env = int(_extra_args.video_env)
        if video_env >= num_envs:
            raise ValueError(f"--video-env must be -1 or in [0, {num_envs - 1}], got {video_env}")
        if _extra_args.video_path:
            video_path = Path(_extra_args.video_path).expanduser()
            if not video_path.is_absolute():
                video_path = PROJECT_ROOT / video_path
        else:
            video_path = out_dir / f"{target_name}_multi_env_preview_env{video_env}.mp4"
        video_recorder = VideoRecorder(video_path, fps=float(_extra_args.video_fps))
        log(f"Video output    : {video_path}")
        video_label = "all-envs tiled" if video_env < 0 else f"env{video_env}"
        log(f"Video camera/env: {_extra_args.video_camera} / {video_label}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT))
    scene = InteractiveScene(MultiEnvSceneCfg(num_envs=num_envs, env_spacing=2.0))
    stage = omni.usd.get_context().get_stage()
    _attach_target_visuals_for_envs(stage, num_envs)
    log("Phase: sim.reset() + sim.play()")
    sim.reset()
    sim.play()
    sim_dt = sim.get_physics_dt()
    _apply_target_colors_for_envs(stage, num_envs)

    robot = scene["robot"]
    device = str(sim.device)
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    arm_ids_t = torch.tensor(arm_ids, dtype=torch.long, device=device)
    gripper_joint_ids = []
    gripper_joint_cfg = []
    for joint_name, cfg in GRIPPER_MIMIC_MAP.items():
        ids, _ = robot.find_joints([joint_name])
        if ids:
            gripper_joint_ids.append(ids[0])
            gripper_joint_cfg.append(cfg)
    finger_ids_t = torch.tensor(gripper_joint_ids, dtype=torch.long, device=device)
    gripper_signs = torch.tensor([c[0] for c in gripper_joint_cfg], dtype=torch.float32, device=device)
    gripper_lows = torch.tensor([c[1] for c in gripper_joint_cfg], dtype=torch.float32, device=device)
    gripper_highs = torch.tensor([c[2] for c in gripper_joint_cfg], dtype=torch.float32, device=device)

    ee_ids, _ = robot.find_bodies([EE_BODY_NAME])
    ee_body_idx = ee_ids[0]
    ee_jac_idx = robot.body_names.index(EE_BODY_NAME) - 1
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=num_envs,
        device=device,
    )

    log(f"robot root shape  : {_shape_of(robot.data.root_pos_w)}")
    log(f"robot joint names : {robot.data.joint_names}")
    log(f"arm ids           : {arm_ids}")
    log(f"gripper ids       : {gripper_joint_ids}")

    n_success = 0
    n_tried = 0
    t_start = time.monotonic()
    while n_success < n_target and n_tried < max_tried:
        batch_attempts = min(num_envs, max_tried - n_tried)
        n_tried += batch_attempts
        log(f"\n[Batch | success={n_success}/{n_target} tried={n_tried}/{max_tried}]")
        try:
            results = _run_batch(
                sim, scene, robot, ik, sim_dt,
                arm_ids_t, finger_ids_t, gripper_signs, gripper_lows, gripper_highs,
                ee_body_idx, ee_jac_idx,
                target_name, target_info, rngs, instruction,
                video_recorder=video_recorder,
                video_camera=_extra_args.video_camera,
                video_env=int(_extra_args.video_env),
                video_every_n_steps=int(_extra_args.video_every_n_steps),
            )
        except Exception as exc:
            log(f"  BATCH EXCEPTION: {type(exc).__name__}: {exc}")
            log(traceback.format_exc())
            continue

        for env_id, (success, buffer, diag) in enumerate(results[:batch_attempts]):
            if success:
                prefix = "SUCCESS" if n_success < n_target else "EXTRA SUCCESS ignored"
                log(f"  env{env_id}: {prefix} best_lift={diag['best_lift_height']:.3f} place={diag['obj_place_xy_dist']:.3f} steps={diag['num_steps_recorded']}")
                if n_success < n_target:
                    meta = {
                        "episode_id": n_success,
                        "target": target_name,
                        "instruction": instruction,
                        "success": True,
                        "num_steps": diag["num_steps_recorded"],
                        "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "tried_index": n_tried - batch_attempts + env_id + 1,
                        "env_id": int(env_id),
                    }
                    if _extra_args.no_save_h5:
                        log("    SAVE SKIPPED (--no-save-h5)")
                    else:
                        append_episode_h5(h5_path, n_success, buffer, meta)
                    n_success += 1
            else:
                log(
                    f"  env{env_id}: FAILED  best_lift={diag.get('best_lift_height', 0):.3f} "
                    f"place={diag.get('obj_place_xy_dist', 0):.3f} "
                    f"ee_safe={diag.get('ee_safe')} "
                    f"ee_grasp={diag.get('ee_to_grasp_dist_final', 0):.3f} "
                    f"ee_obj_xy={diag.get('ee_to_obj_xy_final', 0):.3f} "
                    f"grip={diag.get('gripper_cmd_final', 0):.3f} "
                    f"joints={[round(x, 3) for x in diag.get('gripper_joint_pos_final', [])]} "
                    f"lifted={diag.get('obj_lifted')} at_place={diag.get('obj_at_place')}"
                )

    elapsed = time.monotonic() - t_start
    log("=" * 60)
    log(f"Collection done. {n_success}/{n_target} successes from {n_tried} attempts in {elapsed/60:.1f} min")
    log(f"Success rate: {n_success}/{max(1, n_tried)} = {100*n_success/max(1,n_tried):.1f}%")
    log(f"Output: {out_dir}")
    if video_recorder is not None:
        video_recorder.close()
        log(f"Video saved: {video_recorder.path} ({video_recorder.frame_count} frames)")
    log("Exiting process.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"EXCEPTION: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise
    finally:
        close_app_once()
