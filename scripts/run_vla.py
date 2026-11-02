"""Run OpenVLA in the canonical three-object Isaac scene."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument("--instruction", default="pick up the red mug")
_extra.add_argument("--vla-server", default="http://localhost:8000")
_extra.add_argument("--vla-step-interval", type=int, default=12)
_extra.add_argument("--action-scale", type=float, default=0.5)
_extra.add_argument("--vla-timeout", type=float, default=10.0)
_extra.add_argument("--max-steps", type=int, default=6000)
_extra.add_argument("--camera", choices=("camera_policy",), default="camera_policy")
_extra.add_argument("--unnorm-key", default="ur3e_vla_dataset")
_extra.add_argument("--lock-orientation", action=argparse.BooleanOptionalAction, default=True)
_extra_args, _ = _extra.parse_known_args()

from vla_sim.isaac_app import args_cli, boot_app, close_app, log

app = boot_app()

from vla_sim.actions import apply_delta_action, clamp_action
from vla_sim.config import TARGET_KEYS, WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z
from vla_sim.simulation import (
    RuntimeOptions,
    SimulationRuntime,
    as_torch,
    pose_wxyz_from_sim,
)
from vla_sim.vla_client import VLAClient


def main() -> None:
    target_name = args_cli.target
    if target_name not in TARGET_KEYS:
        raise ValueError(
            f"--target must be one of {TARGET_KEYS} in the canonical scene; "
            f"received {target_name!r}"
        )
    if _extra_args.vla_step_interval < 1:
        raise ValueError("--vla-step-interval must be at least 1")
    if not np.isfinite(_extra_args.action_scale) or _extra_args.action_scale < 0:
        raise ValueError("--action-scale must be finite and nonnegative")
    if _extra_args.max_steps < 0:
        raise ValueError("--max-steps must be nonnegative (0 runs until stopped)")

    vla = VLAClient(_extra_args.vla_server, timeout=_extra_args.vla_timeout)
    if not vla.health_check(log):
        raise RuntimeError("VLA server is unavailable or not ready")

    runtime = SimulationRuntime(
        RuntimeOptions(scene_profile="vla", device=args_cli.device)
    ).start()
    runtime.reset_targets()
    controller = runtime.robot_controller
    scene = runtime.scene
    if controller is None or scene is None:
        raise RuntimeError("canonical runtime did not initialize")
    controller.reset_home()
    for _ in range(120):
        runtime.step()

    robot = runtime.robot
    ee_pose = pose_wxyz_from_sim(robot.data.body_link_pose_w)[0, controller.ee_body_idx]
    ee_target_pos = ee_pose[:3].clone()
    ee_target_quat = ee_pose[3:].clone()
    gripper_command = 0.0
    gripper_latched_closed = False
    last_action = np.zeros(7, dtype=np.float32)
    last_action_id = 0
    applied_action_id = 0
    vla_pending = False
    prediction_worker = None
    prediction_error = None
    action_lock = threading.Lock()
    last_log_at = time.monotonic()

    def start_prediction(rgb: np.ndarray) -> None:
        nonlocal last_action, last_action_id, vla_pending, prediction_error
        try:
            action = clamp_action(
                vla.predict(
                    rgb,
                    _extra_args.instruction,
                    unnorm_key=_extra_args.unnorm_key,
                    log=log,
                )
            )
            with action_lock:
                last_action = action
                last_action_id += 1
        except Exception as error:
            with action_lock:
                prediction_error = error
        finally:
            with action_lock:
                vla_pending = False

    log(
        f"VLA target={target_name} camera={_extra_args.camera} "
        f"rate={60 / _extra_args.vla_step_interval:.1f}Hz"
    )
    step = 0
    interrupted = False
    try:
        while app.is_running():
            if step % _extra_args.vla_step_interval == 0:
                camera = scene[_extra_args.camera].data.output.get("rgb")
                if camera is not None:
                    rgb = as_torch(camera)[0].cpu().numpy().astype(np.uint8)
                    with action_lock:
                        should_start = not vla_pending
                        if should_start:
                            vla_pending = True
                    if should_start:
                        prediction_worker = threading.Thread(
                            target=start_prediction, args=(rgb,), daemon=True
                        )
                        prediction_worker.start()

            with action_lock:
                worker_error = prediction_error
                action = (
                    last_action.copy()
                    if last_action_id > applied_action_id
                    else None
                )
                if action is not None:
                    applied_action_id = last_action_id
            if worker_error is not None:
                raise RuntimeError("VLA prediction worker failed") from worker_error
            if action is not None and np.any(action):
                next_pos, next_quat, next_grip = apply_delta_action(
                    ee_target_pos,
                    ee_target_quat,
                    action,
                    _extra_args.action_scale,
                    WORKSPACE_X,
                    WORKSPACE_Y,
                    WORKSPACE_Z,
                )
                ee_target_pos = next_pos
                if not _extra_args.lock_orientation:
                    ee_target_quat = next_quat
                if next_grip > 0.75:
                    gripper_latched_closed = True
                raw_binary = 1.0 if gripper_latched_closed or next_grip > 0.75 else 0.0
                gripper_command += 0.12 * (raw_binary - gripper_command)
                log(f"[VLA] action#{applied_action_id}: {action.round(4).tolist()}")

            controller.set_pose_target(ee_target_pos, ee_target_quat)
            controller.set_gripper_command(gripper_command, runtime.sim_dt)
            runtime.step()

            now = time.monotonic()
            if now - last_log_at >= 1.0:
                obj = as_torch(scene[target_name].data.root_pos_w)[0].cpu().numpy().round(3)
                log(
                    f"step={step} target={ee_target_pos.cpu().numpy().round(3).tolist()} "
                    f"grip={gripper_command:.2f} {target_name}={obj.tolist()} "
                    f"requests={vla.stats['requests']} errors={vla.stats['errors']}"
                )
                last_log_at = now
            step += 1
            if _extra_args.max_steps > 0 and step >= _extra_args.max_steps:
                break
    except KeyboardInterrupt:
        interrupted = True
        log("Ctrl+C received.")
    finally:
        if prediction_worker is not None:
            prediction_worker.join(timeout=vla.timeout + 1.0)
        log(f"VLA stats: {vla.stats} consumed_actions={applied_action_id}")

    if not interrupted and _extra_args.max_steps > 0:
        if prediction_error is not None:
            raise RuntimeError("VLA prediction worker failed") from prediction_error
        if prediction_worker is not None and prediction_worker.is_alive():
            raise RuntimeError("VLA prediction did not finish before shutdown")
        if step < _extra_args.max_steps:
            raise RuntimeError(
                f"VLA rollout stopped early: steps={step} expected={_extra_args.max_steps}"
            )
        stats = vla.stats
        if stats["requests"] == 0 or stats["errors"] or applied_action_id == 0:
            raise RuntimeError(
                f"VLA rollout incomplete: requests={stats['requests']} "
                f"errors={stats['errors']} consumed_actions={applied_action_id}"
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        close_app()
