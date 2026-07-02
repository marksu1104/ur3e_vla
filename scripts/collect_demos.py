"""Collect scripted UR3e grasp demonstrations for VLA fine-tuning."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument("--episodes", type=int, default=5,
                    help="Number of successful episodes to collect.")
_extra.add_argument("--output-dir", type=str, default="outputs/test/h5",
                    help="Base output directory relative to this project.")
_extra.add_argument("--max-episodes-tried", type=int, default=0,
                    help="Maximum attempts. Use 0 for an automatic limit.")
_extra.add_argument("--randomize-pos", action=argparse.BooleanOptionalAction, default=True,
                    help="Randomize target position.")
_extra.add_argument("--randomize-rot", action=argparse.BooleanOptionalAction, default=True,
                    help="Randomize target yaw while preserving base orientation.")
_extra.add_argument("--randomize-light", action=argparse.BooleanOptionalAction, default=True,
                    help="Randomize dome light intensity and color.")
_extra.add_argument("--randomize-camera-view", action=argparse.BooleanOptionalAction, default=True,
                    help="Randomly choose a small camera_main view preset per episode.")
_extra.add_argument("--keep-sim-alive", action=argparse.BooleanOptionalAction, default=False,
                    help="Keep simulation open after collection.")
_extra.add_argument("--show-gui", action="store_true",
                    help="Show the Isaac Sim GUI while collecting demos.")
_extra.add_argument("--seed", type=int, default=42)
_extra.add_argument(
    "--overwrite",
    action="store_true",
    help="Remove the existing target demos.h5 before collecting new episodes.",
)
_extra.add_argument(
    "--no-save-h5",
    action="store_true",
    help="Run episodes without writing demos.h5. Useful for trajectory inspection.",
)
_extra.add_argument(
    "--record-video",
    action="store_true",
    help="Record camera_main frames to an MP4 while running episodes.",
)
_extra.add_argument(
    "--video-path",
    type=str,
    default="",
    help="Output MP4 path. Defaults to <output-dir>/<target>/trajectory.mp4.",
)
_extra.add_argument(
    "--video-camera",
    choices=("camera_main", "camera_wrist"),
    default="camera_main",
    help="Camera used for MP4 recording.",
)
_extra.add_argument(
    "--video-fps",
    type=float,
    default=30.0,
    help="Output MP4 frame rate.",
)
_extra.add_argument(
    "--video-width",
    type=int,
    default=2560,
    help="Camera width used only for --record-video runs.",
)
_extra.add_argument(
    "--video-height",
    type=int,
    default=1440,
    help="Camera height used only for --record-video runs.",
)
_extra.add_argument(
    "--video-every-n-steps",
    type=int,
    default=2,
    help="Write one video frame every N sim steps. At 60 Hz, N=2 gives 30 FPS.",
)
_extra_args, _ = _extra.parse_known_args()

if _extra_args.record_video:
    os.environ["VLA_CAMERA_MAIN_WIDTH"] = str(_extra_args.video_width)
    os.environ["VLA_CAMERA_MAIN_HEIGHT"] = str(_extra_args.video_height)

if not _extra_args.show_gui and "--headless" not in sys.argv:
    sys.argv.append("--headless")

from vla_sim.isaac_app import boot_app, args_cli, log

app = boot_app()
_app_closed = False


def close_app_once():
    """Close Isaac Sim once; some Kit versions are unhappy with duplicate close calls."""
    global _app_closed
    if _app_closed:
        return
    _app_closed = True
    try:
        app.close(wait_for_replicator=False)
    except TypeError:
        app.close()


import time
import random
import traceback

import numpy as np
import torch

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from vla_sim.config import (
    PHYSICS_DT,
    ROBOT_BASE_POS, ROBOT_BASE_ROT,
    EE_BODY_NAME, EE_ORIENT_DOWN,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    HOME_POS, HOME_Q,
    TARGETS, CAMERA_MAIN_VIEW_PRESETS,
    GRIPPER_MIMIC_MAP,
)
from vla_sim.scene import (
    enable_extensions,
    spawn_raw_and_assemble,
    SceneCfg,
    apply_scene_colors,
    hide_proxy_meshes,
)
from vla_sim.actions import PoseTrajectoryPlayer, compute_action_from_ee_poses
from vla_sim.data_collector import EpisodeBuffer, append_episode_h5
from vla_sim.demo_planning import (
    randomize_lighting,
    randomize_target_pose,
    sample_grasp_parameters,
    sample_grasp_quat,
    sample_scene_object_poses,
    detect_success,
)
from vla_sim.video import VideoRecorder


random.seed(_extra_args.seed)
np.random.seed(_extra_args.seed)


def apply_camera_main_view(stage, view: dict):
    """Apply a discrete camera_main pose preset to the USD camera prim."""
    try:
        from pxr import Gf, UsdGeom

        prim = stage.GetPrimAtPath("/World/CameraMain")
        if not prim.IsValid():
            return False

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

        pos = view["pos"]
        rot = view["rot"]
        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        orient_op.Set(Gf.Quatd(float(rot[0]), Gf.Vec3d(float(rot[1]), float(rot[2]), float(rot[3]))))
        return True
    except Exception as exc:
        log(f"  camera view randomization failed: {exc}")
        return False


def run_one_episode(
    sim, scene, robot, ik, sim_dt,
    arm_ids_t, finger_ids_t,
    gripper_signs, gripper_lows, gripper_highs,
    ee_body_idx, ee_jac_idx,
    target_name: str,
    target_info: dict,
    target_obj,
    rng: np.random.Generator,
    instruction: str,
    record_every_n_steps: int = 12,    # 12 sim steps @ 60Hz = 5Hz logging
    video_recorder: VideoRecorder | None = None,
    video_camera: str = "camera_main",
    video_every_n_steps: int = 2,
) -> tuple[bool, EpisodeBuffer | None, dict]:
    """Run one scripted episode and return success, data, and diagnostics."""
    device = str(sim.device)

    scene_object_poses = sample_scene_object_poses(
        TARGETS,
        rng,
        randomize_pos=_extra_args.randomize_pos,
        randomize_rot=_extra_args.randomize_rot,
        place_pos=HOME_POS,
        log_fn=log,
    )
    if target_name not in scene_object_poses:
        scene_object_poses[target_name] = randomize_target_pose(
            target_info,
            rng,
            randomize_pos=_extra_args.randomize_pos,
            randomize_rot=_extra_args.randomize_rot,
        )

    for object_name, (object_pos, object_rot) in scene_object_poses.items():
        try:
            object_obj = scene[object_name]
        except Exception:
            continue
        object_state = torch.tensor(
            [[*object_pos, *object_rot, 0, 0, 0, 0, 0, 0]],
            device=device, dtype=torch.float32,
        )
        object_obj.write_root_pose_to_sim(object_state[:, :7])
        object_obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

    spawn_pos, spawn_rot = scene_object_poses[target_name]

    # 把手臂送回 home pose
    home_q_t = torch.tensor([HOME_Q], device=device, dtype=torch.float32)
    robot.write_joint_state_to_sim(
        position=home_q_t, velocity=torch.zeros_like(home_q_t), joint_ids=arm_ids_t,
    )
    open_cmd = torch.clamp(
        (gripper_signs * GRIPPER_OPEN).unsqueeze(0),
        gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
    )
    robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)

    stage = omni.usd.get_context().get_stage()
    randomize_lighting(stage, rng, enabled=_extra_args.randomize_light, log_fn=log)

    if _extra_args.randomize_camera_view:
        view_idx = int(rng.integers(0, len(CAMERA_MAIN_VIEW_PRESETS)))
        camera_view = CAMERA_MAIN_VIEW_PRESETS[view_idx]
    else:
        camera_view = CAMERA_MAIN_VIEW_PRESETS[0]
    apply_camera_main_view(stage, camera_view)

    for _ in range(60):
        robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
        robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    target_resting = target_obj.data.root_pos_w[0].cpu().numpy()
    target_initial_z = float(target_resting[2])
    tx, ty, tz = target_resting.tolist()
    grasp_params = sample_grasp_parameters(target_info, rng)
    gx = tx + grasp_params["x_nudge"]
    gy = ty + grasp_params["y_nudge"]
    hover_z = tz + grasp_params["hover_z"]
    grasp_z = tz + grasp_params["grasp_z"]

    HOVER_POS = (gx, gy, hover_z)
    PRE_GRASP_POS = (gx, gy, min(grasp_z + 0.055, hover_z))
    GRASP_POS = (gx, gy, grasp_z)
    LIFT_START_POS = (gx, gy, min(grasp_z + 0.025, hover_z))
    grasp_quat = sample_grasp_quat(target_info, spawn_rot, rng)

    # Add small timing variation so the policy cannot overfit to a fixed close
    # step. The visual/robot state should determine when to close, not episode
    # clock time. Keep close itself short, then immediately lift.
    hover_dt = float(rng.uniform(3.2, 3.8))
    yaw_dt = 0.0
    pre_grasp_dt = float(rng.uniform(1.9, 2.4))
    descend_dt = float(rng.uniform(1.0, 1.4))
    close_dt = float(rng.uniform(0.35, 0.55))
    lift_start_dt = float(rng.uniform(0.45, 0.70))
    lift_dt = float(rng.uniform(2.1, 2.6))
    carry_dt = float(rng.uniform(2.7, 3.3))
    release_dt = float(rng.uniform(1.6, 2.1))

    trajectory = [
        (0.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_OPEN),
        # Move toward the hover pose while yaw-aligning. A separate stationary
        # yaw segment made larger datasets overfit to hovering/yawing instead
        # of descending once aligned.
        (hover_dt, HOVER_POS, grasp_quat, GRIPPER_OPEN),
        (pre_grasp_dt, PRE_GRASP_POS, grasp_quat, GRIPPER_OPEN),
        (descend_dt, GRASP_POS, grasp_quat, GRIPPER_OPEN),
        # Close briefly while stationary for a stable grasp, then lift right
        # away to make the close -> lift transition unambiguous.
        (close_dt, GRASP_POS, grasp_quat, GRIPPER_CLOSE),
        (lift_start_dt, LIFT_START_POS, grasp_quat, GRIPPER_CLOSE),
        (lift_dt, HOVER_POS, grasp_quat, GRIPPER_CLOSE),
        (carry_dt, HOME_POS,  grasp_quat, GRIPPER_CLOSE),
        (release_dt, HOME_POS,  grasp_quat, GRIPPER_OPEN),
    ]
    player = PoseTrajectoryPlayer(trajectory, device=device)

    buffer = EpisodeBuffer(
        main_images=[], wrist_images=[],
        ee_poses=[], joint_positions=[],
        gripper_states=[], actions_7d=[], timestamps=[],
    )

    last_grip_target = GRIPPER_OPEN
    t_sim = 0.0
    step = 0
    prev_ee_pos = None
    prev_ee_quat = None
    finished_at = None
    success_seen = False
    success_diag = None
    best_lift_height = 0.0

    while True:
        tgt_pos_w, tgt_quat_w, grip_target, finished = player.sample(t_sim)

        # Let the object settle briefly after the scripted motion ends.
        if finished and finished_at is None:
            finished_at = step
        if finished_at is not None and step - finished_at > 60:
            break

        root = robot.data.root_state_w[:, :7]
        root_pos = root[:, :3]
        tgt_pos_b = tgt_pos_w.unsqueeze(0) - root_pos
        tgt_quat_b = tgt_quat_w.unsqueeze(0)
        ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))

        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
        ee_pos_b = ee_pose_w[:, :3] - root_pos
        ee_quat_b = ee_pose_w[:, 3:]
        q_current = robot.data.joint_pos[:, arm_ids_t]

        jac_full = robot.root_physx_view.get_jacobians()
        jac = jac_full[:, ee_jac_idx, :, :][:, :, arm_ids_t]

        q_target = ik.compute(ee_pos_b, ee_quat_b, jac, q_current)
        robot.set_joint_position_target(q_target, joint_ids=arm_ids_t)

        max_step = 0.008
        delta = grip_target - last_grip_target
        if delta > max_step:
            grip_target = last_grip_target + max_step
        elif delta < -max_step:
            grip_target = last_grip_target - max_step
        grip_target = float(np.clip(grip_target, GRIPPER_OPEN, GRIPPER_CLOSE))
        last_grip_target = grip_target

        finger_cmd = torch.clamp(
            (gripper_signs * grip_target).unsqueeze(0),
            gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
        )
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids_t)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if video_recorder is not None and step % max(1, video_every_n_steps) == 0:
            try:
                video_rgb = (
                    scene[video_camera].data.output["rgb"][0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                video_recorder.write_rgb(video_rgb)
            except Exception as e:
                log(f"  video frame failed: {e}")

        obj_pos_check = target_obj.data.root_pos_w[0].cpu().numpy()
        best_lift_height = max(best_lift_height, float(obj_pos_check[2] - target_initial_z))

        if step % record_every_n_steps == 0:
            # Read current state
            ee_pose_now = robot.data.body_state_w[0, ee_body_idx, :7].cpu().numpy()
            ee_pos_now = ee_pose_now[:3]
            ee_quat_now = ee_pose_now[3:]  # [w, x, y, z]
            joint_now = robot.data.joint_pos[0, arm_ids_t].cpu().numpy()
            # Store the commanded logical gripper state instead of a mimic
            # joint position. The physical joint can lag or use a sign/range
            # that does not map cleanly to open/closed, while the policy needs
            # to know whether the scripted controller currently intends open or
            # closed.
            grip_binary = 1.0 if grip_target > ((GRIPPER_OPEN + GRIPPER_CLOSE) * 0.5) else 0.0

            # Compute action_7d from (prev -> current) EE pose
            if prev_ee_pos is None:
                action_7d = np.zeros(7, dtype=np.float32)
                action_7d[6] = grip_binary
            else:
                action_7d = compute_action_from_ee_poses(
                    prev_ee_pos, prev_ee_quat,
                    ee_pos_now,  ee_quat_now,
                    gripper_target=grip_binary,
                )

            try:
                main_rgb = scene["camera_main"].data.output["rgb"][0].cpu().numpy().astype(np.uint8)
                wrist_rgb = (
                    scene["camera_wrist"].data.output["rgb"][0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
            except Exception as e:
                log(f"  camera read failed: {e}")
                main_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
                wrist_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

            buffer.main_images.append(main_rgb)
            buffer.wrist_images.append(wrist_rgb)
            buffer.ee_poses.append([
                float(ee_pos_now[0]), float(ee_pos_now[1]), float(ee_pos_now[2]),
                float(ee_quat_now[0]), float(ee_quat_now[1]),
                float(ee_quat_now[2]), float(ee_quat_now[3]),
            ])
            buffer.joint_positions.append(joint_now.tolist())
            buffer.gripper_states.append(grip_binary)
            buffer.actions_7d.append(action_7d.tolist())
            buffer.timestamps.append(t_sim)

            prev_ee_pos = ee_pos_now
            prev_ee_quat = ee_quat_now

        t_sim += sim_dt
        step += 1

    ee_pos_final = robot.data.body_state_w[0, ee_body_idx, :3].cpu().numpy()
    success, diag = detect_success(
        target_obj,
        ee_pos_final,
        target_initial_z,
        last_grip_target,
        HOME_POS,
        best_lift_height,
    )

    diag["target_initial_pos"]  = list(map(float, target_resting.tolist()))
    diag["num_steps_recorded"]  = len(buffer.main_images)
    diag["spawn_pos"]           = list(map(float, spawn_pos))
    diag["spawn_rot"]           = list(map(float, spawn_rot))
    diag["grasp_params"]        = {k: float(v) for k, v in grasp_params.items()}
    diag["grasp_quat"]          = list(map(float, grasp_quat))
    diag["camera_main_view"]    = {
        "name": camera_view.get("name", "unknown"),
        "pos": list(map(float, camera_view["pos"])),
        "rot": list(map(float, camera_view["rot"])),
    }
    diag["trajectory_durations"] = {
        "hover": hover_dt,
        "yaw": yaw_dt,
        "pre_grasp": pre_grasp_dt,
        "descend": descend_dt,
        "close": close_dt,
        "lift_start": lift_start_dt,
        "lift": lift_dt,
        "carry": carry_dt,
        "release": release_dt,
    }
    diag["scene_object_poses"]  = {
        name: {
            "pos": list(map(float, pose[0])),
            "rot": list(map(float, pose[1])),
        }
        for name, pose in scene_object_poses.items()
    }

    return success, (buffer if success else None), diag


def main():
    target_name = args_cli.target
    target_info = TARGETS[target_name]
    instruction = f"pick up the {target_name.replace('_', ' ')}"

    out_dir_rel = Path(_extra_args.output_dir)
    out_dir = out_dir_rel if out_dir_rel.is_absolute() else PROJECT_ROOT / out_dir_rel
    if out_dir.name != target_name:
        out_dir = out_dir / target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "demos.h5"
    if _extra_args.no_save_h5:
        log("H5 saving       : disabled (--no-save-h5)")
    elif _extra_args.overwrite and h5_path.exists():
        h5_path.unlink()
        log(f"Removed existing dataset: {h5_path}")
    elif h5_path.exists():
        try:
            import h5py
            with h5py.File(h5_path, "r") as h5_file:
                existing_count = len(h5_file.get("data", {}))
        except Exception:
            existing_count = "unknown"
        log(
            f"WARNING: appending to existing dataset: {h5_path} "
            f"(existing episodes: {existing_count}). Use --overwrite to start fresh."
        )

    video_recorder = None
    video_path = None
    if _extra_args.record_video:
        if _extra_args.video_path:
            video_path = Path(_extra_args.video_path)
            if not video_path.is_absolute():
                video_path = PROJECT_ROOT / video_path
        else:
            video_path = out_dir / "trajectory.mp4"
        video_recorder = VideoRecorder(video_path, fps=float(_extra_args.video_fps))

    n_target_episodes = _extra_args.episodes
    max_tried = _extra_args.max_episodes_tried or max(n_target_episodes * 3, n_target_episodes + 10)

    log(f"Target object   : {target_name}")
    log(f"Instruction     : '{instruction}'")
    log(f"Target episodes : {n_target_episodes}")
    log(f"Max tried       : {max_tried}")
    log(f"argv            : {sys.argv}")
    log(f"Output dir      : {out_dir}")
    if video_path is not None:
        log(f"Video output    : {video_path}")
        log(f"Video camera/fps: {_extra_args.video_camera} / {_extra_args.video_fps}")
        log(f"Video resolution: {_extra_args.video_width}x{_extra_args.video_height}")
    log(f"Random seed     : {_extra_args.seed}")
    log(f"Randomize pos / rot / light: {_extra_args.randomize_pos} / "
        f"{_extra_args.randomize_rot} / {_extra_args.randomize_light}")
    log(f"Randomize camera view: {_extra_args.randomize_camera_view}")

    rng = np.random.default_rng(_extra_args.seed)

    enable_extensions()
    log("Phase 1: spawn + Robot Assembler")
    spawn_raw_and_assemble()

    log("Phase 2: SimulationContext + InteractiveScene")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT))
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    log("Attaching YCB visual meshes...")
    from isaacsim.core.utils.stage import add_reference_to_stage
    for target_key, info in TARGETS.items():
        if info.get("collision_usd"):
            log(f"  {target_key}: using local collision USD, skip visual attach")
            continue
        usd_abs = f"{ISAAC_NUCLEUS_DIR}/{info['usd_relative']}"
        visual_path = f"/World/{target_key.capitalize()}/Visuals"
        try:
            add_reference_to_stage(usd_path=usd_abs, prim_path=visual_path)
        except Exception as e:
            log(f"  attach failed ({target_key}): {e}")

    log("Phase 3: sim.reset() + sim.play()")
    sim.reset()
    sim.play()
    sim_dt = sim.get_physics_dt()

    stage = omni.usd.get_context().get_stage()
    apply_scene_colors(stage)
    hide_proxy_meshes(stage, TARGETS.keys())

    robot = scene["robot"]
    device = str(sim.device)

    root_pose = torch.tensor(
        [[*ROBOT_BASE_POS, *ROBOT_BASE_ROT]],
        device=device, dtype=torch.float32,
    )
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    arm_ids, _ = robot.find_joints(arm_joint_names)
    arm_ids_t = torch.tensor(arm_ids, dtype=torch.long, device=device)

    gripper_joint_ids = []
    gripper_joint_cfg = []
    for joint_name, (sign, lo, hi) in GRIPPER_MIMIC_MAP.items():
        ids, _ = robot.find_joints([joint_name])
        if ids:
            gripper_joint_ids.append(ids[0])
            gripper_joint_cfg.append((sign, lo, hi))
    finger_ids_t = torch.tensor(gripper_joint_ids, dtype=torch.long, device=device)
    gripper_signs = torch.tensor(
        [c[0] for c in gripper_joint_cfg], dtype=torch.float32, device=device
    )
    gripper_lows = torch.tensor(
        [c[1] for c in gripper_joint_cfg], dtype=torch.float32, device=device
    )
    gripper_highs = torch.tensor(
        [c[2] for c in gripper_joint_cfg], dtype=torch.float32, device=device
    )

    ee_ids, _ = robot.find_bodies([EE_BODY_NAME])
    ee_body_idx = ee_ids[0]
    body_names = robot.body_names
    ee_jac_idx = body_names.index(EE_BODY_NAME) - 1

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls",
    )
    ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    target_obj = scene[target_name]

    log("=" * 60)
    log("Starting episode collection...")
    log("=" * 60)

    n_success = 0
    n_tried = 0
    t_start = time.monotonic()

    while n_success < n_target_episodes and n_tried < max_tried:
        n_tried += 1
        ep_id = n_success
        log(f"\n[Ep {n_tried:04d} | success={n_success}/{n_target_episodes}]")

        try:
            success, buffer, diag = run_one_episode(
                sim, scene, robot, ik, sim_dt,
                arm_ids_t, finger_ids_t,
                gripper_signs, gripper_lows, gripper_highs,
                ee_body_idx, ee_jac_idx,
                target_name, target_info, target_obj,
                rng, instruction,
                video_recorder=video_recorder,
                video_camera=_extra_args.video_camera,
                video_every_n_steps=_extra_args.video_every_n_steps,
            )
        except Exception as e:
            log(f"  EXCEPTION: {e}")
            log(traceback.format_exc())
            continue

        if success:
            log(f"  SUCCESS  lift={diag['obj_lift_height']:.3f}m  "
                f"best_lift={diag.get('best_lift_height', 0):.3f}m  "
                f"place_dist={diag.get('obj_place_xy_dist', 0):.3f}m  "
                f"steps={diag['num_steps_recorded']}")
            meta = {
                "episode_id": ep_id,
                "target": target_name,
                "instruction": instruction,
                "success": True,
                "diag": diag,
                "num_steps": diag["num_steps_recorded"],
                "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tried_index": n_tried,
            }
            if _extra_args.no_save_h5:
                log("  SAVE SKIPPED (--no-save-h5)")
            else:
                try:
                    append_episode_h5(h5_path, ep_id, buffer, meta)
                except Exception as e:
                    log(f"  SAVE FAILED: {type(e).__name__}: {e}")
                    log(traceback.format_exc())
                    continue
            n_success += 1
            log(f"  Progress: success={n_success}/{n_target_episodes}, "
                f"tried={n_tried}/{max_tried}")
        else:
            log(f"  FAILED   lift={diag.get('obj_lift_height', 0):.3f}m  "
                f"best_lift={diag.get('best_lift_height', 0):.3f}m  "
                f"lifted={diag.get('obj_lifted')}  "
                f"at_place={diag.get('obj_at_place')}  "
                f"place_dist={diag.get('obj_place_xy_dist', 0):.3f}m  "
                f"ee_safe={diag.get('ee_safe')}")
            log(f"  Progress: success={n_success}/{n_target_episodes}, "
                f"tried={n_tried}/{max_tried}")

    elapsed = time.monotonic() - t_start
    if video_recorder is not None:
        video_recorder.close()
        log(f"Video saved: {video_recorder.path} ({video_recorder.frame_count} frames)")
    log("=" * 60)
    if n_success >= n_target_episodes:
        log("Stop reason: reached target successful episodes")
    elif n_tried >= max_tried:
        log("Stop reason: reached max episode attempts")
    else:
        log("Stop reason: episode loop exited")
    log(f"Collection done. {n_success}/{n_target_episodes} episodes in {elapsed/60:.1f} min")
    log(f"Success rate: {n_success}/{n_tried} = {100*n_success/max(1,n_tried):.1f}%")
    log(f"Average per success episode: {elapsed/max(1,n_success):.1f} sec")
    log(f"Output: {out_dir}")

    if _extra_args.keep_sim_alive:
        log("Collection complete. Keeping simulation alive. Press Ctrl+C to close.")
        try:
            while app.is_running():
                sim.step()
                scene.update(sim_dt)
        except KeyboardInterrupt:
            log("Ctrl+C received. Closing simulation.")
    else:
        log("Collection complete. Exiting process.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        close_app_once()
