"""Run UR3e simulation with actions from a remote OpenVLA server."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument("--instruction", type=str, default="pick up the object")
_extra.add_argument("--vla-server", type=str, default="http://localhost:8000")
_extra.add_argument(
    "--vla-step-interval",
    type=int,
    default=12,
    help="Number of sim steps between VLA calls. 12 is 5 Hz at 60 Hz sim.",
)
_extra.add_argument(
    "--action-scale",
    type=float,
    default=0.5,
    help="Scale applied to VLA delta actions.",
)
_extra.add_argument(
    "--vla-timeout",
    type=float,
    default=10.0,
    help="HTTP timeout for one VLA prediction request.",
)
_extra.add_argument(
    "--max-steps",
    type=int,
    default=6000,
    help="Maximum simulation steps. Use 0 to run until interrupted.",
)
_extra.add_argument(
    "--camera",
    type=str,
    default="camera_main",
    choices=("camera_main", "camera_wrist"),
    help="Camera observation sent to the VLA policy.",
)
_extra.add_argument(
    "--unnorm-key",
    type=str,
    default="ur3e_vla_dataset",
    help="OpenVLA action unnormalization key.",
)
_extra.add_argument(
    "--lock-orientation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep the end-effector orientation fixed during VLA rollout.",
)
_extra_args, _ = _extra.parse_known_args()

from vla_sim.isaac_app import boot_app, args_cli, log

app = boot_app()

import time
import threading
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
    EE_BODY_NAME,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    HOME_Q,
    TARGETS,
    GRIPPER_MIMIC_MAP,
    WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z,
)
from vla_sim.scene import (
    enable_extensions,
    spawn_raw_and_assemble,
    SceneCfg,
    apply_scene_colors,
    hide_proxy_meshes,
)
from vla_sim.actions import apply_delta_action, clamp_action
from vla_sim.vla_client import VLAClient


def main():
    target_name = args_cli.target
    instruction = _extra_args.instruction
    vla_server = _extra_args.vla_server
    step_interval = _extra_args.vla_step_interval
    action_scale = _extra_args.action_scale
    vla_timeout = _extra_args.vla_timeout
    max_steps = _extra_args.max_steps
    camera_name = _extra_args.camera
    unnorm_key = _extra_args.unnorm_key
    lock_orientation = _extra_args.lock_orientation

    log(f"Target: {target_name}")
    log(f"Instruction: '{instruction}'")
    log(f"VLA server: {vla_server}")
    log(f"VLA call every {step_interval} sim steps "
        f"(= {60/step_interval:.1f} Hz at 60Hz sim)")
    log(f"Action scale: {action_scale}")
    log(f"VLA timeout: {vla_timeout}s")
    log(f"VLA camera: {camera_name}")
    log(f"Unnorm key: {unnorm_key}")
    log(f"Lock orientation: {lock_orientation}")

    vla = VLAClient(vla_server, timeout=vla_timeout)
    log("[VLA] Checking server health...")
    if not vla.health_check(log):
        log("[VLA] WARNING: server is not ready. VLA calls will reuse the last action.")
        log("[VLA] Start it with: python scripts/vla_server.py --model-path ~/models/openvla-7b")
    else:
        log("[VLA] Server OK, model loaded.")

    enable_extensions()

    log("Phase 1: spawn + Robot Assembler")
    spawn_raw_and_assemble()

    log("Phase 2: SimulationContext + InteractiveScene")
    log("Phase 2.1: Creating SimulationContext...")
    try:
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT))
        log("Phase 2.1: SimulationContext OK")
    except Exception as e:
        log(f"Phase 2.1 FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        raise

    log("Phase 2.2: Creating SceneCfg...")
    try:
        scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
        log("Phase 2.2: SceneCfg OK")
    except Exception as e:
        log(f"Phase 2.2 FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        raise

    log("Phase 2.3: Creating InteractiveScene...")
    try:
        scene = InteractiveScene(scene_cfg)
        log("Phase 2.3: InteractiveScene OK")
    except Exception as e:
        log(f"Phase 2.3 FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        raise

    # Attach YCB visual mesh
    log("Attaching YCB visual meshes...")
    from isaacsim.core.utils.stage import add_reference_to_stage
    import omni.client as _client
    for target_key, info in TARGETS.items():
        if info.get("collision_usd"):
            log(f"  {target_key}: using local collision USD, skip visual attach")
            continue
        usd_abs = f"{ISAAC_NUCLEUS_DIR}/{info['usd_relative']}"
        visual_path = f"/World/{target_key.capitalize()}/Visuals"
        result, _ = _client.stat(usd_abs)
        log(f"  {target_key}: stat={result}")
        try:
            add_reference_to_stage(usd_path=usd_abs, prim_path=visual_path)
        except Exception as e:
            log(f"  attach FAILED ({target_key}): {e}")

    log("Phase 3.1: sim.reset()...")
    try:
        sim.reset()
        log("Phase 3.1: sim.reset() OK")
    except Exception as e:
        log(f"Phase 3.1 FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        raise

    log("Phase 3.2: sim.play()...")
    try:
        sim.play()
        log("Phase 3.2: sim.play() OK")
    except Exception as e:
        log(f"Phase 3.2 FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        raise

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

    home_q_t = torch.tensor([HOME_Q], device=device, dtype=torch.float32)
    robot.write_joint_state_to_sim(
        position=home_q_t, velocity=torch.zeros_like(home_q_t), joint_ids=arm_ids_t,
    )

    open_cmd = torch.clamp(
        (gripper_signs * GRIPPER_OPEN).unsqueeze(0),
        gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
    )
    robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
    robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)

    log("Settling 180 steps...")
    for _ in range(180):
        robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
        robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # IK controller
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls",
    )
    ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    ws_x = WORKSPACE_X
    ws_y = WORKSPACE_Y
    ws_z = WORKSPACE_Z

    # Start from the current end-effector pose.
    ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
    ee_target_pos = ee_pose_w[0, :3].clone()
    ee_target_quat = ee_pose_w[0, 3:7].clone()
    gripper_target = GRIPPER_OPEN
    gripper_latched_closed = False

    log("Entering main loop. Ctrl+C to stop.")
    step = 0
    last_log_t = time.monotonic()
    last_vla_action = np.zeros(7, dtype=np.float32)
    last_action_id = 0
    applied_action_id = 0
    vla_pending = False
    action_lock = threading.Lock()

    try:
        while app.is_running():
            if step % step_interval == 0:
                cam_data = scene[camera_name].data.output.get("rgb")
                if cam_data is not None:
                    rgb_np = cam_data[0].cpu().numpy().astype(np.uint8)

                    def _async_predict():
                        nonlocal last_vla_action, last_action_id, vla_pending
                        try:
                            action = clamp_action(
                                vla.predict(rgb_np, instruction, unnorm_key=unnorm_key, log=log)
                            )
                            with action_lock:
                                last_vla_action = action
                                last_action_id += 1
                        finally:
                            with action_lock:
                                vla_pending = False

                    should_start = False
                    with action_lock:
                        if not vla_pending:
                            vla_pending = True
                            should_start = True
                    if should_start:
                        t = threading.Thread(target=_async_predict, daemon=True)
                        t.start()

            # Apply each predicted action once. Re-applying a stale action can
            # push the IK target out of distribution while async inference is slow.
            action_to_apply = None
            with action_lock:
                if last_action_id > applied_action_id:
                    action_to_apply = last_vla_action.copy()
                    applied_action_id = last_action_id

            if action_to_apply is not None and not np.all(action_to_apply == 0):
                new_pos, new_quat, new_grip = apply_delta_action(
                    ee_target_pos, ee_target_quat,
                    action_to_apply, action_scale,
                    ws_x, ws_y, ws_z,
                )
                ee_target_pos = new_pos
                if not lock_orientation:
                    ee_target_quat = new_quat
                if new_grip > 0.75:
                    gripper_latched_closed = True
                elif (not gripper_latched_closed) and new_grip < 0.25:
                    gripper_target = GRIPPER_OPEN
                if gripper_latched_closed:
                    gripper_target = 1.0
                else:
                    gripper_target = new_grip
                log(f"[VLA] apply action#{applied_action_id}: {action_to_apply.round(4).tolist()}")

            root_pos = robot.data.root_state_w[:, :3]
            ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
            ee_pos_b = ee_pose_w[:, :3] - root_pos
            ee_quat_b = ee_pose_w[:, 3:]

            tgt_pos_b = ee_target_pos.unsqueeze(0) - root_pos
            tgt_quat_b = ee_target_quat.unsqueeze(0)
            ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))

            q_current = robot.data.joint_pos[:, arm_ids_t]
            jac_full = robot.root_physx_view.get_jacobians()
            jac = jac_full[:, ee_jac_idx, :, :][:, :, arm_ids_t]

            q_target = ik.compute(ee_pos_b, ee_quat_b, jac, q_current)
            robot.set_joint_position_target(q_target, joint_ids=arm_ids_t)

            GRIP_SMOOTH = 0.12
            if gripper_latched_closed:
                raw_grip_binary = 1.0
            else:
                raw_grip_binary = 1.0 if gripper_target > 0.75 else 0.0
            gripper_target = gripper_target + GRIP_SMOOTH * (raw_grip_binary - gripper_target)
            grip_scaled = gripper_target * (GRIPPER_CLOSE - GRIPPER_OPEN) + GRIPPER_OPEN
            finger_cmd = torch.clamp(
                (gripper_signs * grip_scaled).unsqueeze(0),
                gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
            )
            robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids_t)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            now = time.monotonic()
            if now - last_log_t >= 1.0:
                ee_now = ee_pose_w[0, :3].cpu().numpy().round(3).tolist()
                tgt_now = ee_target_pos.cpu().numpy().round(3).tolist()
                obj = scene[target_name].data.root_pos_w[0].cpu().numpy().round(3).tolist()
                with action_lock:
                    pending_now = vla_pending
                    action_now = last_action_id
                log(f"step={step} ee={ee_now} tgt={tgt_now} "
                    f"grip={gripper_target:.2f} latched={gripper_latched_closed} {target_name}={obj} "
                    f"vla_req={vla.stats['requests']} err={vla.stats['errors']} "
                    f"pending={pending_now} action={applied_action_id}/{action_now}")
                last_log_t = now

            step += 1
            if max_steps > 0 and step >= max_steps:
                log(f"Reached max_steps={max_steps}, stopping.")
                break

    except KeyboardInterrupt:
        log("Ctrl+C received.")
    finally:
        log(f"VLA stats: {vla.stats}")
        log("Simulation done. Press Ctrl+C to close window.")
        while app.is_running():
            sim.step() 


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
