"""Serve a scripted three-object UR3e pick-and-place simulation over HTTP/WS."""

from __future__ import annotations

import argparse
import sys
import threading
import time
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

from vla_sim.isaac_app import args_cli, boot_app, close_app, log

def _run_with_progress(label, operation):
    """Run one startup stage with a low-noise heartbeat for SSH users."""
    started = time.monotonic()
    finished = threading.Event()

    def report_progress() -> None:
        while not finished.wait(15.0):
            elapsed = time.monotonic() - started
            print(f"[REMOTE] Still {label} ({elapsed:.0f}s elapsed)...", flush=True)

    reporter = threading.Thread(target=report_progress, daemon=True)
    reporter.start()
    try:
        return operation()
    finally:
        finished.set()
        reporter.join(timeout=1.0)


_app_start = time.monotonic()
print(
    "[REMOTE] Starting Isaac Sim application...",
    flush=True,
)
app = _run_with_progress("starting Isaac Sim", boot_app)
print(
    f"[REMOTE] Isaac Sim application started in "
    f"{time.monotonic() - _app_start:.1f}s.",
    flush=True,
)

import numpy as np
from vla_sim.actions import PoseTrajectoryPlayer
from vla_sim.config import (
    BOWL_DESTINATION_Z_OFFSET,
    BRIDGE_HOST,
    BRIDGE_PORT,
    CUTLERY_BOX_CENTER_X_OFFSET,
    CUTLERY_BOX_CENTER_Y_OFFSET,
    CUTLERY_BOX_TARGET_SLOT_Y_OFFSET,
    EE_ORIENT_DOWN,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOME_POS,
    MAX_TRIAL_SECONDS,
    MUG_DESTINATION_Z_OFFSET,
    PLACE_POSITIONS,
    SPOON_DESTINATION_Z_OFFSET,
    STREAM_EVERY_N_STEPS,
    STREAM_JPEG_QUALITY,
    TARGET_KEYS,
    TARGETS,
    TASK_INDEX_MAP,
)
from vla_sim.bridge import BridgeServer
from vla_sim.planning import build_pick_place_trajectory, detect_success
from vla_sim.visibility import goal_visibility_report, object_visibility_report
from vla_sim.runtime import (
    RuntimeOptions,
    SimulationRuntime,
    as_torch,
    pose_wxyz_from_sim,
)


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
    print(
        f"[REMOTE] Bridge listening on {BRIDGE_HOST}:{port}; state=starting.",
        flush=True,
    )
    try:
        runtime_start = time.monotonic()
        print(
            "[REMOTE] Loading robot, scene, fixtures, physics, and YOLO camera...",
            flush=True,
        )
        runtime = _run_with_progress(
            "initializing the scene and RTX renderer",
            lambda: SimulationRuntime(
                RuntimeOptions(
                    scene_profile="remote",
                    stream_width=_extra_args.stream_width,
                    stream_height=_extra_args.stream_height,
                    show_markers=_extra_args.show_markers,
                    device=args_cli.device,
                )
            ).start(),
        )
        print(
            f"[REMOTE] Scene initialized in {time.monotonic() - runtime_start:.1f}s; "
            "settling physics and warming the camera...",
            flush=True,
        )
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
        ready_announced = False
        bridge.set_state("resetting", seed=seed)

        while app.is_running():
            command = bridge.poll_command()
            if command is not None:
                bridge.command_applied(command)
                if command["type"] == "task" and state == "waiting":
                    trial = command
                    target = scene[command["object"]]
                    resting = as_torch(target.data.root_pos_w)[0].cpu().numpy()
                    rotation = (
                        pose_wxyz_from_sim(target.data.root_link_pose_w)[0, 3:7]
                        .cpu()
                        .numpy()
                    )
                    target_initial_z = float(resting[2])
                    best_lift = 0.0
                    max_gripper_command = float(GRIPPER_OPEN)
                    max_finger_joint = float(GRIPPER_OPEN)
                    run_t = 0.0
                    target_info = TARGETS[command["object"]]
                    semantic_destination = (
                        command["task_index"] == command["position_index"]
                    )
                    place_z_offset = 0.0
                    place_yaw_offset = 0.0
                    object_place_xy = None
                    place_nudge_scale = 1.0
                    if semantic_destination:
                        if command["object"] == "spoon":
                            place_z_offset = SPOON_DESTINATION_Z_OFFSET
                            place_yaw_offset = np.pi / 2.0
                            object_place_xy = (
                                PLACE_POSITIONS[0][0] + CUTLERY_BOX_CENTER_X_OFFSET,
                                PLACE_POSITIONS[0][1]
                                + CUTLERY_BOX_CENTER_Y_OFFSET
                                + CUTLERY_BOX_TARGET_SLOT_Y_OFFSET,
                            )
                            place_nudge_scale = 0.40
                        elif command["object"] == "red_mug":
                            place_z_offset = MUG_DESTINATION_Z_OFFSET
                        elif command["object"] == "bowl":
                            place_z_offset = BOWL_DESTINATION_Z_OFFSET
                            object_place_xy = PLACE_POSITIONS[2]
                            place_nudge_scale = 0.32
                    trajectory = build_pick_place_trajectory(
                        target_info,
                        resting,
                        rotation,
                        PLACE_POSITIONS[command["position_index"]],
                        place_z_offset=place_z_offset,
                        place_yaw_offset=place_yaw_offset,
                        object_place_xy=object_place_xy,
                        place_nudge_scale=place_nudge_scale,
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
                        name: as_torch(scene[name].data.root_pos_w)[0]
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
                    bridge.publish_frame(
                        as_torch(rgb)[0].cpu().numpy().astype(np.uint8)
                    )
                    if (
                        state == "waiting"
                        and not ready_announced
                        and bridge.has_encoded_frame
                    ):
                        camera_cfg = scene["camera_yolo"].cfg
                        print("\n" + "=" * 72, flush=True)
                        print(
                            "[REMOTE READY] Isaac scene, camera, and bridge are ready.",
                            flush=True,
                        )
                        print(
                            f"[REMOTE READY] Stream: {camera_cfg.width}x{camera_cfg.height} "
                            f"at ws://{BRIDGE_HOST}:{port}/unity",
                            flush=True,
                        )
                        print(
                            f"[REMOTE READY] Status: "
                            f"http://{BRIDGE_HOST}:{port}/status",
                            flush=True,
                        )
                        print("[REMOTE READY] Unity may start the experiment now.", flush=True)
                        print("=" * 72 + "\n", flush=True)
                        ready_announced = True
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
