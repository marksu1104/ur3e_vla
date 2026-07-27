"""Run persistent real-to-sim mirroring or the validated sim-to-real task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument(
    "--direction",
    choices=("real-to-sim", "sim-to-real"),
    default="real-to-sim",
)
_extra.add_argument("--joint-states-topic", default="/joint_states")
_extra.add_argument("--joint-state-timeout", type=float, default=0.5)
_extra.add_argument("--bridge-port", type=int, default=None)
_extra.add_argument("--stream-width", type=int, default=None)
_extra.add_argument("--stream-height", type=int, default=None)
_extra.add_argument("--stream-quality", type=int, default=None)
_extra.add_argument("--stream-every-n-steps", type=int, default=None)
_extra.add_argument("--show-markers", action="store_true")
_extra.add_argument("--seed", type=int, default=42)
_extra.add_argument("--enable-motion", action="store_true")
_extra.add_argument("--servo-topic", default="/servo_node/delta_twist_cmds")
_extra.add_argument("--frame-id", default="base_link")
_extra.add_argument("--gripper-io-service", default="/io_and_status_controller/set_io")
_extra.add_argument("--io-states-topic", default="/io_and_status_controller/io_states")
_extra.add_argument("--gripper-io-pin", type=int, default=0)
_extra.add_argument("--gripper-timeout", type=float, default=3.0)
_extra.add_argument("--waypoint-tolerance", type=float, default=0.015)
_extra.add_argument("--waypoint-timeout", type=float, default=90.0)
_extra.add_argument("--linear-gain", type=float, default=1.0)
_extra.add_argument("--max-linear-speed", type=float, default=0.05)
_extra.add_argument("--yaw-gain", type=float, default=1.0)
_extra.add_argument("--max-yaw-speed", type=float, default=0.20)
_extra.add_argument("--yaw-tolerance", type=float, default=0.05)
_extra.add_argument("--grasp-z-offset", type=float, default=0.02)
_extra.add_argument("--place-z-offset", type=float, default=0.02)
_extra.add_argument(
    "--compensate-frame-offset",
    action=argparse.BooleanOptionalAction,
    default=True,
)
_extra_args, _ = _extra.parse_known_args()

from vla_sim.isaac_app import boot_app, close_app, log

app = boot_app()

import numpy as np

from vla_sim.bridge import BridgeServer
from vla_sim.config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    PLACE_POSITIONS,
    STREAM_EVERY_N_STEPS,
    STREAM_JPEG_QUALITY,
    TARGET_KEYS,
    TASK_INDEX_MAP,
)
from vla_sim.runtime import RuntimeOptions, SimulationRuntime
from vla_sim.sim_real_sync import (
    JointSyncBackend,
    LatestJointState,
    ROSJointStateSubscriber,
    SUPPORTED_TASK_PAIR,
    SyncOptions,
    run_sim_to_real,
)
from vla_sim.visibility import object_visibility_report


def _scene_status(runtime: SimulationRuntime, backend: JointSyncBackend, seed: int) -> dict:
    scene = runtime.scene
    controller = runtime.robot_controller
    if scene is None or controller is None:
        raise RuntimeError("canonical runtime did not initialize")
    poses = {
        name: scene[name].data.root_pos_w[0].cpu().numpy().round(5).tolist()
        for name in TARGET_KEYS
    }
    return {
        "seed": seed,
        "object_poses": poses,
        "yolo_visibility": object_visibility_report(scene["camera_yolo"]),
        "joint_sync": backend.status,
        "ee_position": controller.ee_position.round(5).tolist(),
    }


def _sync_options() -> SyncOptions:
    return SyncOptions(
        enable_motion=_extra_args.enable_motion,
        gripper_timeout=_extra_args.gripper_timeout,
        waypoint_tolerance=_extra_args.waypoint_tolerance,
        waypoint_timeout=_extra_args.waypoint_timeout,
        linear_gain=_extra_args.linear_gain,
        max_linear_speed=_extra_args.max_linear_speed,
        yaw_gain=_extra_args.yaw_gain,
        max_yaw_speed=_extra_args.max_yaw_speed,
        yaw_tolerance=_extra_args.yaw_tolerance,
        grasp_z_offset=_extra_args.grasp_z_offset,
        place_z_offset=_extra_args.place_z_offset,
        compensate_frame_offset=_extra_args.compensate_frame_offset,
    )


def _validate_args() -> None:
    positive = {
        "--joint-state-timeout": _extra_args.joint_state_timeout,
        "--gripper-timeout": _extra_args.gripper_timeout,
        "--waypoint-tolerance": _extra_args.waypoint_tolerance,
        "--waypoint-timeout": _extra_args.waypoint_timeout,
        "--yaw-tolerance": _extra_args.yaw_tolerance,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    non_negative = {
        "--max-linear-speed": _extra_args.max_linear_speed,
        "--max-yaw-speed": _extra_args.max_yaw_speed,
    }
    for name, value in non_negative.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")


def main() -> None:
    _validate_args()
    stream_every = (
        STREAM_EVERY_N_STEPS
        if _extra_args.stream_every_n_steps is None
        else _extra_args.stream_every_n_steps
    )
    if stream_every < 1:
        raise ValueError("--stream-every-n-steps must be at least 1")
    port = BRIDGE_PORT if _extra_args.bridge_port is None else _extra_args.bridge_port
    quality = (
        STREAM_JPEG_QUALITY
        if _extra_args.stream_quality is None
        else _extra_args.stream_quality
    )

    import rclpy

    sim_to_real = _extra_args.direction == "sim-to-real"
    latest = LatestJointState(stale_timeout=_extra_args.joint_state_timeout)
    backend = JointSyncBackend(latest)
    bridge = BridgeServer(
        TASK_INDEX_MAP if sim_to_real else {},
        len(PLACE_POSITIONS) if sim_to_real else 0,
        host=BRIDGE_HOST,
        port=port,
        jpeg_quality=quality,
        allow_tasks=sim_to_real,
        allowed_task_pairs={SUPPORTED_TASK_PAIR} if sim_to_real else None,
    )
    bridge.start()
    bridge.set_state("starting")
    subscriber = None
    commander = None
    try:
        runtime = SimulationRuntime(
            RuntimeOptions(
                stream_width=_extra_args.stream_width,
                stream_height=_extra_args.stream_height,
                show_markers=_extra_args.show_markers,
            ),
            state_backend=backend,
        ).start()
        runtime.reset_targets()
        for _ in range(120):
            runtime.step()
        rclpy.init()
        subscriber = ROSJointStateSubscriber(_extra_args.joint_states_topic, latest)
        state = "starting"
        seed = _extra_args.seed
        step = 0

        if sim_to_real:
            from vla_sim.real_arm_io import RealArmCommander

            log("Waiting for a live joint state before commanding the real arm...")
            while app.is_running() and not backend.last_snapshot.is_live:
                subscriber.spin_once()
                runtime.step()
            commander = RealArmCommander(
                servo_topic=_extra_args.servo_topic,
                gripper_io_service=_extra_args.gripper_io_service,
                io_states_topic=_extra_args.io_states_topic,
                gripper_io_pin=_extra_args.gripper_io_pin,
                frame_id=_extra_args.frame_id,
                motion_enabled=_extra_args.enable_motion,
                log=log,
            )
            run_sim_to_real(
                app=app,
                runtime=runtime,
                controller=runtime.robot_controller,
                bridge=bridge,
                commander=commander,
                subscriber=subscriber,
                backend=backend,
                seed=seed,
                stream_every=stream_every,
                options=_sync_options(),
                log=log,
            )
            return

        log(
            "Read-only real-to-sim sync active: subscribing to "
            f"{_extra_args.joint_states_topic}; no ROS publishers are created."
        )
        while app.is_running():
            subscriber.spin_once()
            command = bridge.poll_command()
            if command is not None:
                bridge.command_applied(command)
                if command["type"] == "control":
                    action = command["action"]
                    if action == "pause":
                        backend.paused = True
                        state = "paused"
                        bridge.set_state(state, **_scene_status(runtime, backend, seed))
                    elif action == "resume":
                        backend.paused = False
                        state = (
                            "running" if backend.last_snapshot.is_live else "hold"
                        )
                        bridge.set_state(state, **_scene_status(runtime, backend, seed))
                    elif action == "reset":
                        seed = seed if command["seed"] is None else command["seed"]
                        runtime.reset_targets()
                        bridge.set_state("resetting", seed=seed, joint_sync=backend.status)
                        state = "resetting"

            runtime.step()
            snapshot = backend.status
            ee_position = runtime.robot_controller.ee_position.round(5).tolist()
            if state != "paused":
                next_state = "running" if snapshot["state"] == "live" else "hold"
                if next_state != state:
                    state = next_state
                    bridge.set_state(state, **_scene_status(runtime, backend, seed))
                else:
                    bridge.update_status(joint_sync=snapshot, ee_position=ee_position)
            else:
                bridge.update_status(joint_sync=snapshot, ee_position=ee_position)

            if step % stream_every == 0:
                rgb = runtime.latest_yolo_rgb()
                if rgb is not None:
                    bridge.publish_frame(rgb[0].cpu().numpy().astype(np.uint8))
            step += 1
    except KeyboardInterrupt:
        log("Ctrl+C received.")
    finally:
        if commander is not None:
            commander.send_zero_twist()
            commander.close()
        if subscriber is not None:
            subscriber.close()
        if rclpy.ok():
            rclpy.shutdown()
        bridge.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_app()
