"""Fast checks for in-memory H5 collection state; no Isaac app is required."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np

from vla_sim.h5_dataset import EpisodeBuffer, append_episode_h5, build_episode_arrays


def make_buffer() -> EpisodeBuffer:
    """Three distinct transitions make action alignment observable."""
    return EpisodeBuffer(
        main_images=[np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)],
        wrist_images=[np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)],
        ee_poses=[[0., 0., 1., 1., 0., 0., 0.] for _ in range(3)],
        joint_positions=[[0.] * 6 for _ in range(3)],
        gripper_states=[0., 1., 1.],
        actions_7d=[[0., 0., 0., 0., 0., 0., 0.],
                    [.01, 0., 0., 0., 0., 0., 1.],
                    [.02, 0., 0., 0., 0., 0., 1.]],
    )


class EpisodeBufferTest(unittest.TestCase):
    def test_default_buffers_do_not_share_mutable_lists(self) -> None:
        first = EpisodeBuffer()
        second = EpisodeBuffer()

        first.main_images.append("frame")

        self.assertEqual(first.main_images, ["frame"])
        self.assertEqual(second.main_images, [])

    def test_actions_describe_next_frame_and_end_with_terminal_flag(self) -> None:
        buffer = make_buffer()
        arrays = build_episode_arrays(buffer, "pick up the red mug")
        np.testing.assert_array_equal(
            arrays["action"][:-1, :7], np.asarray(buffer.actions_7d[1:], dtype=np.float32)
        )
        np.testing.assert_array_equal(arrays["action"][-1], [0., 0., 0., 0., 0., 0., 1., 1.])
        np.testing.assert_array_equal(arrays["action"][:, -1], [0., 0., 1.])
        self.assertEqual(arrays["robot_state"].shape, (3, 15))

    def test_empty_episode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty episode"):
            build_episode_arrays(EpisodeBuffer(), "pick up the red mug")

    def test_missing_camera_frame_is_rejected(self) -> None:
        buffer = make_buffer()
        buffer.wrist_images.pop()
        with self.assertRaisesRegex(ValueError, "wrist_images has 2 steps; expected 3"):
            build_episode_arrays(buffer, "pick up the red mug")

    def test_wrong_action_width_is_rejected(self) -> None:
        buffer = make_buffer()
        buffer.actions_7d = [[0.] * 8 for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "actions_7d must have shape"):
            build_episode_arrays(buffer, "pick up the red mug")

    def test_nonfinite_joint_state_is_rejected(self) -> None:
        buffer = make_buffer()
        buffer.joint_positions[0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "joint_positions contains non-finite"):
            build_episode_arrays(buffer, "pick up the red mug")


class H5ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "demos.h5"
        self.meta = {
            "instruction": "pick up the red mug",
            "scene_profile": "canonical_scene_v1",
        }
        append_episode_h5(self.path, 0, make_buffer(), self.meta)

    def test_duplicate_episode_does_not_replace_existing_data(self) -> None:
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            append_episode_h5(self.path, 0, make_buffer(), self.meta)
        with h5py.File(self.path, "r") as dataset:
            self.assertEqual(dataset.attrs["num_demos"], 1)
            self.assertEqual(dataset["data/demo_0/task"][()].decode(), self.meta["instruction"])
        self.assertEqual(self.path.read_bytes(), original)

    def test_new_episode_appends_without_changing_previous_episode(self) -> None:
        append_episode_h5(self.path, 1, make_buffer(), self.meta)
        with h5py.File(self.path, "r") as dataset:
            self.assertEqual(dataset.attrs["num_demos"], 2)
            self.assertEqual(set(dataset["data"]), {"demo_0", "demo_1"})

    def test_incompatible_scene_cannot_be_added(self) -> None:
        with self.assertRaisesRegex(ValueError, "scene_profile"):
            append_episode_h5(self.path, 1, make_buffer(), dict(self.meta, scene_profile="legacy"))
        with h5py.File(self.path, "r") as dataset:
            self.assertEqual(dataset.attrs["num_demos"], 1)
            self.assertNotIn("demo_1", dataset["data"])


if __name__ == "__main__":
    unittest.main()
