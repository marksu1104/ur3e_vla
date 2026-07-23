"""Serve a scripted three-object UR3e pick-and-place simulation over HTTP/WS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument("--bridge-port", type=int, default=None)
_extra.add_argument("--stream-width", type=int, default=None)
_extra.add_argument("--stream-height", type=int, default=None)
_extra.add_argument("--stream-quality", type=int, default=None)
_extra.add_argument("--stream-every-n-steps", type=int, default=None)
_extra.add_argument("--max-trial-seconds", type=float, default=None)
_extra.add_argument("--show-markers", action="store_true")
_extra.add_argument("--seed", type=int, default=42)
_extra_args, _ = _extra.parse_known_args()

from vla_sim.isaac_app import boot_app, close_app, log

app = boot_app()

import numpy as np
from vla_sim.actions import PoseTrajectoryPlayer
from vla_sim.config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    EE_ORIENT_DOWN,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOME_POS,
    MAX_TRIAL_SECONDS,
    PLACE_POSITIONS,
    STREAM_EVERY_N_STEPS,
    STREAM_JPEG_QUALITY,
    TARGET_KEYS,
    TARGETS,
    TASK_INDEX_MAP,
)
from vla_sim.bridge import BridgeServer
from vla_sim.planning import build_pick_place_trajectory, detect_success
from vla_sim.visibility import goal_visibility_report, object_visibility_report
from vla_sim.runtime import RuntimeOptions, SimulationRuntime


def main() -> None:
    port = BRIDGE_PORT if _extra_args.bridge_port is None else _extra_args.bridge_port
    stream_every = (
        STREAM_EVERY_N_STEPS
        if _extra_args.stream_every_n_steps is None
        else _extra_args.stream_every_n_steps
    )
    quality = (
        STREAM_JPEG_QUALITY
        if _extra_args.stream_quality is None
        else _extra_args.stream_quality
    )
    max_trial_seconds = (
        MAX_TRIAL_SECONDS
        if _extra_args.max_trial_seconds is None
        else _extra_args.max_trial_seconds
    )
    if stream_every < 1:
        raise ValueError("--stream-every-n-steps must be at least 1")

    bridge = BridgeServer(
        TASK_INDEX_MAP,
        len(PLACE_POSITIONS),
        host=BRIDGE_HOST,
        port=port,
        jpeg_quality=quality,
    )
    bridge.start()
    bridge.set_state("starting")
    try:
        runtime = SimulationRuntime(
            RuntimeOptions(
                stream_width=_extra_args.stream_width,
                stream_height=_extra_args.stream_height,
                show_markers=_extra_args.show_markers,
            )
        ).start()
        scene = runtime.scene
        controller = runtime.robot_controller
        if scene is None or controller is None:
            raise RuntimeError("canonical runtime did not initialize")
        sim_dt = runtime.sim_dt
        device = runtime.device
        finger_limits = controller.finger_limits

        state = "resetting"
        paused_from_state = "running"
        seed = _extra_args.seed
        runtime.reset_targets()
        controller.reset_home()
        settle_steps = 120
        trial = None
        player = None
        run_t = 0.0
        target_initial_z = 0.0
        best_lift = 0.0
        max_gripper_command = GRIPPER_OPEN
        max_finger_joint = GRIPPER_OPEN
        frozen_pos = HOME_POS
        frozen_quat = EE_ORIENT_DOWN
        logical_grip = 0.0
        step = 0
        bridge.set_state("resetting", seed=seed)

        while app.is_running():
            command = bridge.poll_command()
            if command is not None:
                bridge.command_applied(command)
                if command["type"] == "task" and state == "waiting":
                    trial = command
                    target = scene[command["object"]]
                    resting = target.data.root_pos_w[0].cpu().numpy()
                    rotation = target.data.root_state_w[0, 3:7].cpu().numpy()
                    target_initial_z = float(resting[2])
                    best_lift = 0.0
                    max_gripper_command = float(GRIPPER_OPEN)
                    max_finger_joint = float(GRIPPER_OPEN)
                    run_t = 0.0
                    target_info = TARGETS[command["object"]]
                    trajectory = build_pick_place_trajectory(
                        target_info,
                        resting,
                        rotation,
                        PLACE_POSITIONS[command["position_index"]],
                    )
                    player = PoseTrajectoryPlayer(trajectory, device=device)

                    frozen_pos, frozen_quat, logical_grip, _ = player.sample(0.0)
                    state = "running"
                    paused_from_state = "running"
                    bridge.set_state(
                        "running",
                        trial_id=command["trial_id"],
                        object=command["object"],
                        task_index=command["task_index"],
                        position_index=command["position_index"],
                        progress={"t": 0.0, "total": player.total_time},
                        seed=seed,
                    )
                elif command["type"] == "control":
                    action = command["action"]
                    if action == "pause" and state in {"running", "settling"}:
                        paused_from_state = state
                        state = "paused"
                        bridge.set_state(
                            "paused",
                            progress={"t": run_t, "total": player.total_time},
                        )
                    elif action == "resume" and state == "paused":
                        state = paused_from_state
                        bridge.set_state(
                            "running",
                            progress={"t": run_t, "total": player.total_time},
                        )
                    elif action == "reset":
                        seed = seed if command["seed"] is None else command["seed"]
                        runtime.reset_targets()
                        controller.reset_home()
                        state = "resetting"
                        paused_from_state = "running"
                        trial = None
                        player = None
                        settle_steps = 120
                        frozen_pos = HOME_POS
                        frozen_quat = EE_ORIENT_DOWN
                        logical_grip = 0.0
                        bridge.set_state("resetting", seed=seed)

            if state == "running" and player is not None:
                run_t += sim_dt
                frozen_pos, frozen_quat, desired_grip, finished = player.sample(run_t)
                logical_grip = float(desired_grip)
                controller.set_gripper_command(logical_grip, sim_dt)
                raw_grip = controller.physical_gripper_command
                target = scene[trial["object"]]
                lift_height = (
                    float(target.data.root_pos_w[0, 2].item()) - target_initial_z
                )
                best_lift = max(best_lift, lift_height)
                finger_q = controller.finger_position
                max_gripper_command = max(max_gripper_command, float(raw_grip))
                max_finger_joint = max(max_finger_joint, finger_q)
                bridge.update_status(
                    progress={
                        "t": min(run_t, player.total_time),
                        "total": player.total_time,
                        "gripper_command": float(raw_grip),
                        "finger_joint": finger_q,
                    }
                )
                if run_t > max_trial_seconds:
                    finished = True
                if finished:
                    settle_steps = 40
                    state = "settling"
            elif state == "settling":
                settle_steps -= 1
                if settle_steps <= 0 and trial is not None:
                    target = scene[trial["object"]]
                    ee_pos = controller.ee_position
                    gripper_q = controller.finger_position
                    threshold = 0.12 if trial["object"] == "bowl" else 0.10
                    place_pos = (
                        *PLACE_POSITIONS[trial["position_index"]],
                        0.0,
                    )
                    success, detail = detect_success(
                        target,
                        ee_pos,
                        target_initial_z,
                        gripper_q,
                        place_pos,
                        best_lift,
                        place_xy_threshold=threshold,
                    )
                    detail.update(
                        {
                            "gripper_close_target": float(GRIPPER_CLOSE),
                            "max_gripper_command": float(max_gripper_command),
                            "max_finger_joint": float(max_finger_joint),
                        }
                    )
                    result = {
                        "success": bool(success),
                        "detail": detail,
                        "reason": (
                            "completed" if success else "success_checks_failed"
                        ),
                    }
                    state = "done"
                    bridge.finish_trial(
                        trial_id=trial["trial_id"],
                        result=result,
                        progress={"t": player.total_time, "total": player.total_time},
                    )

            if state == "resetting":
                settle_steps -= 1
                if settle_steps <= 0:
                    poses = {
                        name: scene[name]
                        .data.root_pos_w[0]
                        .cpu()
                        .numpy()
                        .round(5)
                        .tolist()
                        for name in TARGET_KEYS
                    }
                    waiting_finger_q = controller.finger_position
                    yolo_visibility = object_visibility_report(scene["camera_yolo"])
                    goal_visibility = (
                        goal_visibility_report(scene["camera_yolo"])
                        if _extra_args.show_markers
                        else None
                    )
                    log(f"YOLO visibility: {yolo_visibility}")
                    if goal_visibility is not None:
                        log(f"YOLO goal visibility: {goal_visibility}")
                    state = "waiting"
                    bridge.set_state(
                        "waiting",
                        object_poses=poses,
                        seed=seed,
                        gripper={
                            "command": controller.physical_gripper_command,
                            "finger_joint": waiting_finger_q,
                            "soft_limits": finger_limits,
                        },
                        yolo_visibility=yolo_visibility,
                        goal_visibility=goal_visibility,
                    )
            controller.set_pose_target(frozen_pos, frozen_quat)
            runtime.step()
            if step % stream_every == 0:
                rgb = runtime.latest_yolo_rgb()
                if rgb is not None:
                    bridge.publish_frame(rgb[0].cpu().numpy().astype(np.uint8))
            step += 1
    except KeyboardInterrupt:
        log("Ctrl+C received.")
    finally:
        bridge.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_app()
