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
_extra_args, _ = _extra.parse_known_args()

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
    TARGETS,
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


random.seed(_extra_args.seed)
np.random.seed(_extra_args.seed)



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

    trajectory = [
        (0.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_OPEN),
        # Stage the demonstration so each segment has one main intent:
        # move to hover, yaw-align, descend, close, lift, carry, release.
        (3.5, HOVER_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (0.8, HOVER_POS, grasp_quat, GRIPPER_OPEN),
        (2.2, PRE_GRASP_POS, grasp_quat, GRIPPER_OPEN),
        (1.2, GRASP_POS, grasp_quat, GRIPPER_OPEN),
        # Close briefly while stationary for a more stable physical grasp, then
        # lift immediately to avoid teaching a long no-op at the grasp pose.
        (0.6, GRASP_POS, grasp_quat, GRIPPER_CLOSE),
        (0.6, LIFT_START_POS, grasp_quat, GRIPPER_CLOSE),
        (2.4, HOVER_POS, grasp_quat, GRIPPER_CLOSE),
        (3.0, HOME_POS,  grasp_quat, GRIPPER_CLOSE),
        (2.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_OPEN),
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

        obj_pos_check = target_obj.data.root_pos_w[0].cpu().numpy()
        best_lift_height = max(best_lift_height, float(obj_pos_check[2] - target_initial_z))

        if step % record_every_n_steps == 0:
            # Read current state
            ee_pose_now = robot.data.body_state_w[0, ee_body_idx, :7].cpu().numpy()
            ee_pos_now = ee_pose_now[:3]
            ee_quat_now = ee_pose_now[3:]  # [w, x, y, z]
            joint_now = robot.data.joint_pos[0, arm_ids_t].cpu().numpy()
            grip_now = float(robot.data.joint_pos[0, finger_ids_t[0]].cpu().item())

            # Compute action_7d from (prev -> current) EE pose
            if prev_ee_pos is None:
                action_7d = np.zeros(7, dtype=np.float32)
                action_7d[6] = 1.0 if grip_target > 0.2 else 0.0
            else:
                action_7d = compute_action_from_ee_poses(
                    prev_ee_pos, prev_ee_quat,
                    ee_pos_now,  ee_quat_now,
                    gripper_target=1.0 if grip_target > 0.2 else 0.0,
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
            buffer.gripper_states.append(grip_now)
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
    if _extra_args.overwrite and h5_path.exists():
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

    n_target_episodes = _extra_args.episodes
    max_tried = _extra_args.max_episodes_tried or max(n_target_episodes * 3, n_target_episodes + 10)

    log(f"Target object   : {target_name}")
    log(f"Instruction     : '{instruction}'")
    log(f"Target episodes : {n_target_episodes}")
    log(f"Max tried       : {max_tried}")
    log(f"argv            : {sys.argv}")
    log(f"Output dir      : {out_dir}")
    log(f"Random seed     : {_extra_args.seed}")
    log(f"Randomize pos / rot / light: {_extra_args.randomize_pos} / "
        f"{_extra_args.randomize_rot} / {_extra_args.randomize_light}")

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
