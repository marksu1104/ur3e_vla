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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
_parser.add_argument("--show-gui", action="store_true")
_parser.add_argument("--record-video", action="store_true", help="Record a short test preview MP4 from one env camera.")
_parser.add_argument("--video-path", type=str, default="", help="MP4 path for --record-video. Defaults under output dir.")
_parser.add_argument("--video-camera", type=str, default="camera_policy", choices=["camera_policy", "camera_wrist"])
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

from vla_sim.isaac_app import boot_app, close_app, args_cli, log

app = boot_app()



import omni.usd
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass

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
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOME_Q,
    PHYSICS_DT,
    PLACE_POSITIONS,
    ROBOT_BASE_POS,
    ROBOT_BASE_ROT,
    TABLE_A_POS,
    TABLE_B_POS,
    TABLE_MAT_A_POS,
    TABLE_MAT_B_POS,
    TABLE_MAT_SIZE,
    TARGETS,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_WIDTH,
)
from vla_sim.data_collector import EpisodeBuffer, VideoRecorder, append_episode_h5
from vla_sim.planning import (
    GRIPPER_SPEED_RAD_S,
    build_pick_place_trajectory,
    evaluate_pick_place_success,
)
from vla_sim.scene import (
    apply_target_colors,
    make_static_cuboid_cfg,
    make_table_cfg,
    make_target_cfg,
)

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


def _tile_env_rgb(rgb_all: np.ndarray) -> np.ndarray:
    frames = np.asarray(rgb_all)[..., :3].astype(np.uint8)
    if frames.ndim == 3:
        return frames
    if frames.ndim != 4:
        raise ValueError(f"expected camera rgb shape (N,H,W,C), got {frames.shape}")
    return np.concatenate([frames[i] for i in range(frames.shape[0])], axis=1)


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

    table_a = make_table_cfg("{ENV_REGEX_NS}/TableA", TABLE_A_POS)
    table_b = make_table_cfg("{ENV_REGEX_NS}/TableB", TABLE_B_POS)
    mat_a = make_static_cuboid_cfg("{ENV_REGEX_NS}/MatA", TABLE_MAT_SIZE, TABLE_MAT_A_POS)
    mat_b = make_static_cuboid_cfg("{ENV_REGEX_NS}/MatB", TABLE_MAT_SIZE, TABLE_MAT_B_POS)
    backdrop_back = make_static_cuboid_cfg("{ENV_REGEX_NS}/BackdropBack", BACKDROP_BACK_SIZE, BACKDROP_BACK_POS)
    backdrop_side = make_static_cuboid_cfg("{ENV_REGEX_NS}/BackdropSide", BACKDROP_SIDE_SIZE, BACKDROP_SIDE_POS)

    spoon = make_target_cfg("spoon", TARGETS["spoon"], "{ENV_REGEX_NS}/Spoon")
    red_mug = make_target_cfg("red_mug", TARGETS["red_mug"], "{ENV_REGEX_NS}/Red_mug")
    bowl = make_target_cfg("bowl", TARGETS["bowl"], "{ENV_REGEX_NS}/Bowl")

    camera_policy = CameraCfg(
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


def _setup_batch(scene, sim, robot, sim_dt, target_name, target_info, arm_ids_t, finger_ids_t):
    device = str(sim.device)
    num_envs = scene.num_envs
    origins = scene.env_origins.to(device=device, dtype=torch.float32)
    origins_np = origins.detach().cpu().numpy()

    root_pose = torch.zeros((num_envs, 7), device=device, dtype=torch.float32)
    root_pose[:, :3] = torch.tensor(ROBOT_BASE_POS, device=device).unsqueeze(0) + origins
    root_pose[:, 3:] = torch.tensor(ROBOT_BASE_ROT, device=device).unsqueeze(0)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=device))

    home_q = torch.tensor(HOME_Q, device=device).unsqueeze(0).repeat(num_envs, 1)
    open_cmd = torch.full((num_envs, 1), GRIPPER_OPEN, device=device)
    robot.write_joint_state_to_sim(home_q, torch.zeros_like(home_q), joint_ids=arm_ids_t)
    robot.set_joint_position_target(home_q, joint_ids=arm_ids_t)
    robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)

    scene_poses = {}
    for object_name, object_info in TARGETS.items():
        pose = torch.zeros((num_envs, 7), device=device)
        local_pos = torch.tensor(object_info["spawn_pos"], device=device)
        pose[:, :3] = local_pos.unsqueeze(0) + origins
        pose[:, 3:] = torch.tensor(object_info["spawn_rot"], device=device).unsqueeze(0)
        scene[object_name].write_root_pose_to_sim(pose)
        scene[object_name].write_root_velocity_to_sim(
            torch.zeros((num_envs, 6), device=device)
        )
        scene_poses[object_name] = (
            tuple(object_info["spawn_pos"]), tuple(object_info["spawn_rot"])
        )

    for _ in range(60):
        robot.set_joint_position_target(home_q, joint_ids=arm_ids_t)
        robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    target_obj = scene[target_name]
    target_resting = target_obj.data.root_pos_w.detach().cpu().numpy()
    target_rotations = target_obj.data.root_state_w[:, 3:7].detach().cpu().numpy()
    players = []
    traj_meta = []
    place_positions = []
    for env_id in range(num_envs):
        place_local = PLACE_POSITIONS[env_id % len(PLACE_POSITIONS)]
        place_world = (
            float(place_local[0] + origins_np[env_id, 0]),
            float(place_local[1] + origins_np[env_id, 1]),
        )
        trajectory = build_pick_place_trajectory(
            target_info, target_resting[env_id], target_rotations[env_id], place_world
        )
        players.append(PoseTrajectoryPlayer(trajectory, device=device))
        traj_meta.append(
            {
                "grasp_pos": list(map(float, trajectory[3][1])),
                "place_pos": [*place_world, 0.0],
                "scene_object_poses": {
                    name: {"pos": list(pos), "rot": list(rot)}
                    for name, (pos, rot) in scene_poses.items()
                },
            }
        )
        place_positions.append((*place_world, 0.0))
    return (
        players,
        traj_meta,
        target_resting[:, 2].copy(),
        np.asarray(place_positions, dtype=np.float32),
        origins_np.copy(),
    )


def _run_batch(sim, scene, robot, ik, sim_dt, arm_ids_t, finger_ids_t, ee_body_idx, ee_jac_idx, target_name, target_info, video_recorder=None, video_camera="camera_policy", video_env=0, video_every_n_steps=2):
    device = str(sim.device)
    num_envs = scene.num_envs
    target_obj = scene[target_name]
    players, traj_meta, target_initial_z, place_positions, origins_np = _setup_batch(
        scene, sim, robot, sim_dt, target_name, target_info,
        arm_ids_t, finger_ids_t,
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

        physical_target = grip_target * GRIPPER_CLOSE
        step_limit = GRIPPER_SPEED_RAD_S * sim_dt
        last_grip += np.clip(physical_target - last_grip, -step_limit, step_limit)
        finger_cmd = torch.tensor(
            last_grip, dtype=torch.float32, device=device
        ).unsqueeze(1)
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
                main_rgb_all = scene["camera_policy"].data.output["rgb"].detach().cpu().numpy().astype(np.uint8)
            except Exception as exc:
                log(f"camera_policy read failed: {exc}")
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
    obj_pos_final_all = target_obj.data.root_pos_w.detach().cpu().numpy()
    joint_final_all = robot.data.joint_pos[:, finger_ids_t].detach().cpu().numpy()
    for env_id in range(num_envs):
        success, diag = evaluate_pick_place_success(
            obj_pos_final_all[env_id],
            ee_pos_final_all[env_id],
            float(target_initial_z[env_id]),
            float(last_grip[env_id]),
            place_positions[env_id],
            float(best_lift[env_id]),
            place_xy_threshold=0.12 if target_name == "bowl" else 0.10,
            ee_pos_for_safety=ee_pos_final_all[env_id] - origins_np[env_id],
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
    if target_name not in TARGETS:
        raise ValueError(f"target must be one of {tuple(TARGETS)}")
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
    log("Phase: sim.reset() + sim.play()")
    sim.reset()
    sim.play()
    sim_dt = sim.get_physics_dt()
    for env_id in range(num_envs):
        env_prefix = f"/World/envs/env_{env_id}"
        apply_target_colors(stage, TARGETS.keys(), root_prefix=env_prefix)

    robot = scene["robot"]
    device = str(sim.device)
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    arm_ids_t = torch.tensor(arm_ids, dtype=torch.long, device=device)
    gripper_joint_ids, _ = robot.find_joints(["finger_joint"])
    if len(gripper_joint_ids) != 1:
        raise RuntimeError(f"expected one finger_joint, found {gripper_joint_ids}")
    finger_ids_t = torch.tensor(gripper_joint_ids, dtype=torch.long, device=device)

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
                arm_ids_t, finger_ids_t,
                ee_body_idx, ee_jac_idx,
                target_name, target_info,
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
                        "scene_profile": "canonical_scene_v1",
                        "gripper_action_encoding": "logical_binary_0_open_1_closed",
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
        close_app()
