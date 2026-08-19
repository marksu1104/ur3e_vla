"""新增 top, left, right cameras & targets poses
Episode buffer + HDF5 export additions for the two extra camera views.

Does not modify vla_sim/data_collector.py. ``EpisodeBuffer`` is a plain
dataclass with 7 required fields and no extension point, and
``append_episode_h5`` writes a fixed set of datasets, so instead of
editing either:

- ``MultiviewEpisodeBuffer`` subclasses ``EpisodeBuffer`` and adds two
  new fields with empty-list defaults, so existing positional/keyword
  construction of ``EpisodeBuffer`` elsewhere keeps working unchanged.
- ``append_episode_h5_multiview`` calls the original
  ``append_episode_h5`` first (writing the 5 core datasets exactly as
  before), then reopens the same HDF5 group to add
  ``other/left_image`` and ``other/right_image``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vla_sim.config import TARGET_KEYS
from vla_sim.data_collector import EpisodeBuffer, append_episode_h5


def _empty_pose_lists() -> dict:
    return {name: [] for name in TARGET_KEYS}


@dataclass
class MultiviewEpisodeBuffer(EpisodeBuffer):
    """EpisodeBuffer plus the two extra third-person camera streams."""

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
    """Append one episode, including the two extra camera views.

    Reuses ``vla_sim.data_collector.append_episode_h5`` for the 5 core
    datasets (image, other/hand_image, robot_state, action, task) so
    that logic never has to be duplicated/kept in sync by hand, then
    adds the two new views under the same episode group.
    """
    import h5py

    append_episode_h5(h5_path, episode_id, buffer, meta)

    group_name = f"demo_{episode_id}"
    with h5py.File(h5_path, "a") as h5_file:
        other_group = h5_file[f"data/{group_name}/other"]
        top = np.stack(buffer.top_images).astype(np.uint8)
        left = np.stack(buffer.left_images).astype(np.uint8)
        right = np.stack(buffer.right_images).astype(np.uint8)
        other_group.create_dataset(
            "top_image", data=top,
            compression="gzip", compression_opts=4,
            chunks=(1, *top.shape[1:]),
        )
        other_group.create_dataset(
            "left_image",
            data=left,
            compression="gzip",
            compression_opts=4,
            chunks=(1, *left.shape[1:]),
        )
        other_group.create_dataset(
            "right_image",
            data=right,
            compression="gzip",
            compression_opts=4,
            chunks=(1, *right.shape[1:]),
        )
        obj_group = h5_file.require_group(f"data/{group_name}/object_poses")
        for name in TARGET_KEYS:
            poses = np.asarray(buffer.object_poses[name], dtype=np.float32)
            obj_group.create_dataset(name, data=poses)
        tip_pos = np.asarray(buffer.gripper_tip_positions, dtype=np.float32)
        h5_file[f"data/{group_name}"].create_dataset("gripper_tip_position", data=tip_pos)
        h5_file.attrs["fields"] = (
            "image, other/hand_image, other/top_image, other/left_image, other/right_image, "
            "object_poses/<red_mug|spoon|bowl> (per-step, xyz+wxyz quat), "
            "robot_state, action, task"
        )
