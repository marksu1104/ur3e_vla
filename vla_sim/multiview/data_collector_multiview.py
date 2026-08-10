"""Episode buffer + HDF5 export additions for the two extra camera views.

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

from vla_sim.data_collector import EpisodeBuffer, append_episode_h5


@dataclass
class MultiviewEpisodeBuffer(EpisodeBuffer):
    """EpisodeBuffer plus the two extra third-person camera streams."""

    top_images: list = field(default_factory=list)
    left_images: list = field(default_factory=list)
    right_images: list = field(default_factory=list)


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
        h5_file.attrs["fields"] = (
            "image, other/hand_image, other/top_image, other/left_image, other/right_image, "
            "robot_state, action, task"
        )