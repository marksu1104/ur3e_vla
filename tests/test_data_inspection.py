"""Validate corrupt artifacts and preview gripper labels without Isaac startup."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from scripts.tools.export_h5_frames import export_images, export_videos
from scripts.tools.validate_h5 import validate_h5


class DataInspectionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "demos.h5"
        with h5py.File(self.path, "w") as file:
            file.attrs.update(scene_profile="canonical_scene_v1", schema_version="v1", format="vla_demo_hdf5", num_demos=1)
            demo = file.create_group("data/demo_0")
            demo.attrs.update(scene_profile="canonical_scene_v1", success=True, record_hz=5.0, num_steps=2, instruction="pick up the red mug", gripper_action_encoding="logical_binary_0_open_1_closed")
            images = np.full((2, 480, 640, 3), 100, np.uint8)
            demo.create_dataset("image", data=images, compression="gzip")
            demo.create_dataset("other/hand_image", data=images, compression="gzip")
            state = np.zeros((2, 15), np.float32)
            state[:, 9] = 1
            state[1, 13] = 1
            demo.create_dataset("robot_state", data=state)
            demo.create_dataset("action", data=np.array([[.01, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 1]], np.float32))
            demo.create_dataset("task", data="pick up the red mug", dtype=h5py.string_dtype())

    def test_valid_file_is_read_only(self):
        before = self.path.read_bytes()
        result = validate_h5(self.path, expected_episodes=1, expected_profile="canonical_scene_v1")
        self.assertEqual((result["episodes"], result["steps"]), (1, 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_incomplete_multiview_extension_is_rejected(self):
        with h5py.File(self.path, "r+") as file:
            file.attrs["scene_profile"] = "multiview_scene_v1"
            file["data/demo_0"].attrs["scene_profile"] = "multiview_scene_v1"
        with self.assertRaisesRegex(ValueError, "missing other/top_image"):
            validate_h5(self.path)

    def test_camera_length_mismatch_is_rejected(self):
        with h5py.File(self.path, "r+") as file:
            del file["data/demo_0/other/hand_image"]
            file.create_dataset("data/demo_0/other/hand_image", data=np.ones((1, 480, 640, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "synchronized"):
            validate_h5(self.path)

    def test_dangling_numeric_or_camera_links_are_rejected_read_only(self):
        for field in ("action", "other/hand_image"):
            with self.subTest(field=field):
                corrupt_path = self.root / (field.replace("/", "_") + ".h5")
                corrupt_path.write_bytes(self.path.read_bytes())
                with h5py.File(corrupt_path, "r+") as file:
                    demo = file["data/demo_0"]
                    del demo[field]
                    demo[field] = h5py.SoftLink("/missing")
                before = corrupt_path.read_bytes()
                with self.assertRaisesRegex(ValueError, f"{field}: expected dataset"):
                    validate_h5(corrupt_path)
                self.assertEqual(corrupt_path.read_bytes(), before)

    def test_nonfinite_action_is_rejected(self):
        with h5py.File(self.path, "r+") as file:
            file["data/demo_0/action"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_h5(self.path)

    def test_early_terminal_flag_is_rejected(self):
        with h5py.File(self.path, "r+") as file:
            file["data/demo_0/action"][0, 7] = 1
        with self.assertRaisesRegex(ValueError, "terminal flag"):
            validate_h5(self.path)

    def test_black_camera_frame_is_rejected(self):
        with h5py.File(self.path, "r+") as file:
            file["data/demo_0/image"][1] = 0
        with self.assertRaisesRegex(ValueError, "all-black frame at index 1"):
            validate_h5(self.path)

    def test_episode_count_expectation_is_checked(self):
        with self.assertRaisesRegex(ValueError, "stored 1 episodes, expected 2"):
            validate_h5(self.path, expected_episodes=2)

    def test_contradictory_file_gripper_encoding_is_rejected(self):
        with h5py.File(self.path, "r+") as file:
            file.attrs["gripper_action_encoding"] = "physical_joint"
        with self.assertRaisesRegex(ValueError, "unsupported file gripper_action_encoding"):
            validate_h5(self.path)

    def test_preview_labels_use_gripper_column_for_images_and_video(self):
        from PIL import Image

        calls = []

        def overlay(image, index, gripper, name):
            self.assertIsInstance(image, Image.Image)
            calls.append(gripper)
            return image

        with patch("scripts.tools.export_h5_frames.draw_overlay", side_effect=overlay), contextlib.redirect_stdout(io.StringIO()):
            export_images(str(self.path), ["demo_0"], 1, self.root / "images")
            export_videos(str(self.path), ["demo_0"], self.root / "videos", fps=5)
        self.assertEqual(calls, [0., 1., 0., 1.])

    def test_negative_sampling_interval_is_rejected_before_export(self):
        destination = self.root / "bad_interval"
        with self.assertRaisesRegex(ValueError, "every-n must be at least 1"):
            export_images(str(self.path), ["demo_0"], -1, destination)
        self.assertFalse(destination.exists())

    def test_failed_video_writer_is_reported_and_released(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with patch("cv2.VideoWriter", return_value=writer):
            with self.assertRaisesRegex(RuntimeError, "failed to open video writer"):
                export_videos(str(self.path), ["demo_0"], self.root / "failed_video", fps=5)
        writer.write.assert_not_called()
        writer.release.assert_called_once()

    def test_nonpositive_video_rate_is_rejected_before_export(self):
        destination = self.root / "bad_fps"
        with self.assertRaisesRegex(ValueError, "fps must be at least 1"):
            export_videos(str(self.path), ["demo_0"], destination, fps=0)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
