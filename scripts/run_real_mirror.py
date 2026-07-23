"""Mirror real UR3e joint states into the canonical Isaac scene, read-only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument("--joint-states-topic", default="/joint_states")
_extra.add_argument("--joint-state-timeout", type=float, default=0.5)
_extra.add_argument("--bridge-port", type=int, default=None)
_extra.add_argument("--stream-width", type=int, default=None)
_extra.add_argument("--stream-height", type=int, default=None)
_extra.add_argument("--stream-quality", type=int, default=None)
_extra.add_argument("--stream-every-n-steps", type=int, default=None)
_extra.add_argument("--show-markers", action="store_true")
_extra.add_argument("--seed", type=int, default=42)
_extra_args, _ = _extra.parse_known_args()

from vla_sim.isaac_app import boot_app, close_app, log

app = boot_app()

import numpy as np

from vla_sim.joint_state_mirror import LatestJointState
from vla_sim.remote_bridge import RemoteBridge
from vla_sim.remote_config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    STREAM_EVERY_N_STEPS,
    STREAM_JPEG_QUALITY,
)
from vla_sim.remote_visibility import object_visibility_report
from vla_sim.ros_joint_state import ROSJointStateSubscriber
from vla_sim.runtime import RuntimeOptions, SimulationRuntime
from vla_sim.state_backend import JointStateMirrorBackend


def _scene_status(runtime: SimulationRuntime, backend: JointStateMirrorBackend, seed: int) -> dict:
    scene = runtime.scene
    if scene is None:
        raise RuntimeError("canonical runtime did not initialize")
    from vla_sim.remote_config import REMOTE_TARGET_KEYS

    poses = {
        name: scene[name].data.root_pos_w[0].cpu().numpy().round(5).tolist()
        for name in REMOTE_TARGET_KEYS
    }
    return {
        "seed": seed,
        "object_poses": poses,
        "yolo_visibility": object_visibility_report(scene["camera_yolo"]),
        "mirror": backend.status,
    }


def main() -> None:
    if _extra_args.joint_state_timeout <= 0.0:
        raise ValueError("--joint-state-timeout must be positive")
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

    latest = LatestJointState(stale_timeout=_extra_args.joint_state_timeout)
    backend = JointStateMirrorBackend(latest)
    bridge = RemoteBridge(
        {},
        0,
        host=BRIDGE_HOST,
        port=port,
        jpeg_quality=quality,
        allow_tasks=False,
    )
    bridge.start()
    bridge.set_state("starting")
    subscriber = None
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
        # Match the remote scene's reset settling before reporting its initial
        # YOLO composition. The external backend is HOLD until a message arrives.
        for _ in range(120):
            runtime.step()
        rclpy.init()
        subscriber = ROSJointStateSubscriber(_extra_args.joint_states_topic, latest)
        state = "starting"
        seed = _extra_args.seed
        step = 0
        log(
            "Read-only mirror active: subscribing to "
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
                            "running"
                            if backend.last_snapshot.is_live
                            else "hold"
                        )
                        bridge.set_state(state, **_scene_status(runtime, backend, seed))
                    elif action == "reset":
                        seed = seed if command["seed"] is None else command["seed"]
                        runtime.reset_targets()
                        bridge.set_state("resetting", seed=seed, mirror=backend.status)
                        state = "resetting"

            runtime.step()
            snapshot = backend.status
            if state != "paused":
                next_state = "running" if snapshot["state"] == "live" else "hold"
                if next_state != state:
                    state = next_state
                    bridge.set_state(state, **_scene_status(runtime, backend, seed))
                else:
                    bridge.update_status(mirror=snapshot)
            else:
                bridge.update_status(mirror=snapshot)

            if step % stream_every == 0:
                rgb = runtime.latest_yolo_rgb()
                if rgb is not None:
                    bridge.publish_frame(rgb[0].cpu().numpy().astype(np.uint8))
            step += 1
    except KeyboardInterrupt:
        log("Ctrl+C received.")
    finally:
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
