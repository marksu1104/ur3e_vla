"""Dataset collection buffers and HDF5 export helpers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EpisodeBuffer:
    """In-memory data for one collected demonstration episode."""

    main_images: list
    wrist_images: list
    ee_poses: list
    joint_positions: list
    gripper_states: list
    actions_7d: list
    timestamps: list


def build_episode_arrays(buffer: EpisodeBuffer, instruction: str) -> dict:
    """Convert an episode buffer into fixed arrays for HDF5 storage."""
    num_steps = len(buffer.ee_poses)
    raw_actions = [
        np.asarray(action, dtype=np.float32) for action in buffer.actions_7d
    ]

    robot_states = []
    actions = []
    for idx in range(num_steps):
        ee_pose = np.asarray(buffer.ee_poses[idx], dtype=np.float32)
        joint_pos = np.asarray(buffer.joint_positions[idx], dtype=np.float32)
        grip_binary = 1.0 if float(buffer.gripper_states[idx]) > 0.2 else 0.0

        robot_states.append(
            np.concatenate(
                [
                    joint_pos[:6],
                    ee_pose[:3],
                    ee_pose[3:7],
                    np.array([grip_binary, 0.0], dtype=np.float32),
                ]
            ).astype(np.float32)
        )

        is_last = idx == num_steps - 1
        if is_last:
            policy_action = np.zeros(7, dtype=np.float32)
            policy_action[6] = raw_actions[idx][6]
        else:
            # The action stored with the next recorded frame is the transition
            # from the current observation to that next frame.
            policy_action = raw_actions[idx + 1]

        actions.append(
            np.concatenate(
                [
                    policy_action,
                    np.array([1.0 if is_last else 0.0], dtype=np.float32),
                ]
            ).astype(np.float32)
        )

    return {
        "image": np.stack(buffer.main_images).astype(np.uint8),
        "hand_image": np.stack(buffer.wrist_images).astype(np.uint8),
        "robot_state": np.stack(robot_states).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "task": instruction,
    }


def append_episode_h5(h5_path: Path, episode_id: int, buffer: EpisodeBuffer, meta: dict):
    """Append one episode to a single HDF5 dataset file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for HDF5 dataset export. Install it in the "
            "Isaac Lab collection environment, for example: conda install h5py"
        ) from exc

    instruction = str(meta["instruction"])
    arrays = build_episode_arrays(buffer, instruction)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    group_name = f"demo_{episode_id}"

    with h5py.File(h5_path, "a") as h5_file:
        h5_file.attrs.setdefault("schema_version", "v1")
        h5_file.attrs.setdefault("format", "vla_demo_hdf5")
        h5_file.attrs.setdefault(
            "fields", "image, other/hand_image, robot_state, action, task"
        )

        data_group = h5_file.require_group("data")
        if group_name in data_group:
            del data_group[group_name]
        group = data_group.create_group(group_name)
        other_group = group.create_group("other")

        group.create_dataset(
            "image",
            data=arrays["image"],
            compression="gzip",
            compression_opts=4,
            chunks=(1, *arrays["image"].shape[1:]),
        )
        other_group.create_dataset(
            "hand_image",
            data=arrays["hand_image"],
            compression="gzip",
            compression_opts=4,
            chunks=(1, *arrays["hand_image"].shape[1:]),
        )
        group.create_dataset("robot_state", data=arrays["robot_state"])
        group.create_dataset("action", data=arrays["action"])
        group.create_dataset("task", data=arrays["task"], dtype=string_dtype)

        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                group.attrs[key] = value

        group.attrs["success"] = bool(meta.get("success", True))
        h5_file.attrs["num_demos"] = len(data_group)
