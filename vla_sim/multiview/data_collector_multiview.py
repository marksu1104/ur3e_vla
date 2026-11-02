"""MultiView buffers and HDF5 extensions written with the core episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vla_sim.config import TARGET_KEYS
from vla_sim.h5_dataset import EpisodeBuffer, append_episode_h5, stack_uint8_images


def _empty_pose_lists() -> dict:
    return {name: [] for name in TARGET_KEYS}


@dataclass
class MultiviewEpisodeBuffer(EpisodeBuffer):
    """EpisodeBuffer plus three camera streams, object poses and tip positions."""

    top_images: list = field(default_factory=list)
    left_images: list = field(default_factory=list)
    right_images: list = field(default_factory=list)
    object_poses: dict = field(default_factory=_empty_pose_lists)
    gripper_tip_positions: list = field(default_factory=list)


def append_episode_h5_multiview(
    h5_path: Path,
    episode_id: int,
    buffer: MultiviewEpisodeBuffer,
    meta: dict,
) -> None:
    """Validate extra observations, then append all five views in one write."""
    num_steps = len(buffer.ee_poses)
    main_shape = stack_uint8_images(
        buffer.main_images, "main_images", num_steps
    ).shape

    images = {}
    for name in ("main", "wrist", "top", "left", "right"):
        values = stack_uint8_images(
            getattr(buffer, f"{name}_images"),
            f"{name}_images",
            num_steps,
            expected_shape=main_shape,
        )
        if name not in ("main", "wrist"):
            images[name] = values

    object_poses = {}
    for name in TARGET_KEYS:
        poses = np.asarray(buffer.object_poses.get(name, []), dtype=np.float32)
        if poses.shape != (num_steps, 7):
            raise ValueError(f"object_poses/{name} must have shape ({num_steps}, 7)")
        if not np.isfinite(poses).all():
            raise ValueError(f"object_poses/{name} contains non-finite values")
        object_poses[name] = poses

    tip_positions = np.asarray(buffer.gripper_tip_positions, dtype=np.float32)
    if tip_positions.shape != (num_steps, 3):
        raise ValueError(f"gripper_tip_positions must have shape ({num_steps}, 3)")
    if not np.isfinite(tip_positions).all():
        raise ValueError("gripper_tip_positions contains non-finite values")

    def write_extensions(group):
        other_group = group["other"]
        for name, values in images.items():
            other_group.create_dataset(
                f"{name}_image", data=values,
                compression="gzip", compression_opts=4,
                chunks=(1, *values.shape[1:]),
            )
        obj_group = group.create_group("object_poses")
        for name, poses in object_poses.items():
            obj_group.create_dataset(name, data=poses)
        group.create_dataset("gripper_tip_position", data=tip_positions)
        group.file.attrs["fields"] = (
            "image, other/hand_image, other/top_image, other/left_image, other/right_image, "
            "object_poses/<red_mug|spoon|bowl> (per-step, xyz+wxyz quat), "
            "gripper_tip_position, robot_state, action, task"
        )

    append_episode_h5(
        h5_path, episode_id, buffer, meta, write_extensions=write_extensions
    )
