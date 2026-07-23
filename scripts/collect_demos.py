"""Collect 5 Hz H5 demonstrations in the canonical three-object scene."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument("--episodes", type=int, default=5)
_extra.add_argument("--output-dir", default="outputs/h5/canonical_remote")
_extra.add_argument("--max-episodes-tried", type=int, default=0)
_extra.add_argument("--seed", type=int, default=42)
_extra.add_argument("--overwrite", action="store_true")
_extra.add_argument("--no-save-h5", action="store_true")
_extra.add_argument("--show-gui", action="store_true")
_extra_args, _ = _extra.parse_known_args()

if not _extra_args.show_gui and "--headless" not in sys.argv:
    sys.argv.append("--headless")

from vla_sim.isaac_app import args_cli, boot_app, close_app, log

app = boot_app()

from vla_sim.actions import PoseTrajectoryPlayer, compute_action_from_ee_poses
from vla_sim.data_collector import EpisodeBuffer, append_episode_h5
from vla_sim.demo_planning import (
    build_canonical_pick_place_trajectory,
    detect_success,
)
from vla_sim.remote_config import PLACE_POSITIONS, REMOTE_TARGETS
from vla_sim.runtime import RuntimeOptions, SimulationRuntime


SCENE_PROFILE = "canonical_remote_v1"
RECORD_EVERY_N_STEPS = 12  # 60 Hz simulation / 12 = 5 Hz policy data.


def _rgb(scene, camera_name: str) -> np.ndarray:
    return scene[camera_name].data.output["rgb"][0].cpu().numpy().astype(np.uint8)


def run_one_episode(
    runtime: SimulationRuntime,
    target_name: str,
    place_xy: tuple[float, float],
    instruction: str,
) -> tuple[bool, EpisodeBuffer | None, dict]:
    """Execute one canonical scripted trajectory and retain the legacy H5 fields."""
    scene = runtime.scene
    controller = runtime.robot_controller
    if scene is None or controller is None:
        raise RuntimeError("canonical runtime did not initialize")

    runtime.reset_targets()
    controller.reset_home()
    for _ in range(120):
        runtime.step()

    target = scene[target_name]
    target_resting = target.data.root_pos_w[0].cpu().numpy()
    target_rot = target.data.root_state_w[0, 3:7].cpu().numpy()
    target_initial_z = float(target_resting[2])
    trajectory = build_canonical_pick_place_trajectory(
        REMOTE_TARGETS[target_name], target_resting, target_rot, place_xy
    )
    player = PoseTrajectoryPlayer(trajectory, device=runtime.device)
    buffer = EpisodeBuffer([], [], [], [], [], [], [])

    previous_pos = None
    previous_quat = None
    best_lift_height = 0.0
    finished_at = None
    sim_time = 0.0
    step = 0
    last_logical_grip = 0.0

    while True:
        target_pos, target_quat, logical_grip, finished = player.sample(sim_time)
        if finished and finished_at is None:
            finished_at = step
        if finished_at is not None and step - finished_at > 60:
            break

        controller.set_pose_target(target_pos, target_quat)
        controller.set_gripper_command(float(logical_grip), runtime.sim_dt)
        runtime.step()
        last_logical_grip = float(logical_grip)

        object_pos = target.data.root_pos_w[0].cpu().numpy()
        best_lift_height = max(best_lift_height, float(object_pos[2] - target_initial_z))

        if step % RECORD_EVERY_N_STEPS == 0:
            ee_pose = (
                runtime.robot.data.body_state_w[0, controller.ee_body_idx, :7]
                .cpu()
                .numpy()
            )
            joint_positions = (
                runtime.robot.data.joint_pos[0, controller.arm_ids_t].cpu().numpy()
            )
            grip_binary = 1.0 if logical_grip >= 0.5 else 0.0
            if previous_pos is None:
                action = np.zeros(7, dtype=np.float32)
                action[6] = grip_binary
            else:
                action = compute_action_from_ee_poses(
                    previous_pos,
                    previous_quat,
                    ee_pose[:3],
                    ee_pose[3:7],
                    gripper_target=grip_binary,
                )
            buffer.main_images.append(_rgb(scene, "camera_policy"))
            buffer.wrist_images.append(_rgb(scene, "camera_wrist"))
            buffer.ee_poses.append(ee_pose.tolist())
            buffer.joint_positions.append(joint_positions.tolist())
            buffer.gripper_states.append(grip_binary)
            buffer.actions_7d.append(action.tolist())
            buffer.timestamps.append(sim_time)
            previous_pos = ee_pose[:3].copy()
            previous_quat = ee_pose[3:7].copy()

        sim_time += runtime.sim_dt
        step += 1

    success, diag = detect_success(
        target,
        controller.ee_position,
        target_initial_z,
        controller.finger_position,
        (*place_xy, 0.0),
        best_lift_height,
        place_xy_threshold=0.12 if target_name == "bowl" else 0.10,
    )
    diag.update(
        scene_profile=SCENE_PROFILE,
        target_initial_pos=target_resting.tolist(),
        place_xy=list(place_xy),
        num_steps_recorded=len(buffer.main_images),
        record_every_n_steps=RECORD_EVERY_N_STEPS,
        gripper_action_encoding="logical_binary_0_open_1_closed",
        final_logical_gripper_command=last_logical_grip,
        instruction=instruction,
    )
    return success, buffer if success else None, diag


def main() -> None:
    target_name = args_cli.target
    if target_name not in REMOTE_TARGETS:
        raise ValueError(
            "canonical collection supports "
            f"{tuple(REMOTE_TARGETS)}; received {target_name!r}"
        )
    if _extra_args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    instruction = f"pick up the {target_name.replace('_', ' ')}"
    output_root = Path(_extra_args.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_dir = output_root if output_root.name == target_name else output_root / target_name
    h5_path = output_dir / "demos.h5"
    output_dir.mkdir(parents=True, exist_ok=True)
    if _extra_args.overwrite and h5_path.exists():
        h5_path.unlink()
    if h5_path.exists() and not _extra_args.no_save_h5:
        raise FileExistsError(
            f"Refusing to append {h5_path}; use --overwrite so canonical data "
            "cannot mix with an existing dataset."
        )

    max_tries = _extra_args.max_episodes_tried or max(
        _extra_args.episodes * 3, _extra_args.episodes + 10
    )
    log(f"Canonical H5 scene: {SCENE_PROFILE}")
    log(f"Target={target_name} episodes={_extra_args.episodes} output={output_dir}")
    runtime = SimulationRuntime(RuntimeOptions()).start()
    successes = 0
    attempts = 0
    started = time.monotonic()

    while successes < _extra_args.episodes and attempts < max_tries:
        attempts += 1
        place_xy = PLACE_POSITIONS[successes % len(PLACE_POSITIONS)]
        success, buffer, diag = run_one_episode(
            runtime, target_name, place_xy, instruction
        )
        if not success:
            log(
                f"attempt {attempts}: failed lift={diag['best_lift_height']:.3f} "
                f"place={diag['obj_place_xy_dist']:.3f}"
            )
            continue
        if not _extra_args.no_save_h5:
            append_episode_h5(
                h5_path,
                successes,
                buffer,
                {
                    "episode_id": successes,
                    "target": target_name,
                    "instruction": instruction,
                    "success": True,
                    "scene_profile": SCENE_PROFILE,
                    "gripper_action_encoding": "logical_binary_0_open_1_closed",
                    "num_steps": diag["num_steps_recorded"],
                    "record_hz": 5.0,
                    "seed": _extra_args.seed,
                },
            )
        successes += 1
        log(f"episode {successes}/{_extra_args.episodes}: success")

    elapsed = time.monotonic() - started
    if successes != _extra_args.episodes:
        raise RuntimeError(f"collected {successes}/{_extra_args.episodes} successful episodes")
    destination = "H5 saving disabled" if _extra_args.no_save_h5 else str(h5_path)
    log(f"Collected {successes} demos in {elapsed:.1f}s: {destination}")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_app()
