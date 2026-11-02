"""Fast checks for in-memory H5 collection state; no Isaac app is required."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

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

    def test_camera_frames_require_uniform_uint8_rgb_without_lossy_cast(self) -> None:
        for name in ("main_images", "wrist_images"):
            for defect in (
                "shape",
                "channels",
                "nonnumeric",
                "float",
                "negative",
                "overflow",
                "nonfinite",
            ):
                with self.subTest(stream=name, defect=defect):
                    buffer = make_buffer()
                    if defect == "shape":
                        getattr(buffer, name)[0] = np.zeros((3, 3, 3), dtype=np.uint8)
                    elif defect == "channels":
                        setattr(
                            buffer,
                            name,
                            [np.zeros((2, 3, 4), dtype=np.uint8) for _ in range(3)],
                        )
                    elif defect == "nonnumeric":
                        setattr(
                            buffer,
                            name,
                            [np.full((2, 3, 3), "bad") for _ in range(3)],
                        )
                    else:
                        values = [
                            image.astype(np.float32) for image in getattr(buffer, name)
                        ]
                        values[0][0, 0, 0] = {
                            "float": 0.5,
                            "negative": -1.0,
                            "overflow": 256.0,
                            "nonfinite": np.nan,
                        }[defect]
                        setattr(buffer, name, values)
                    expected = (
                        "non-finite"
                        if defect == "nonfinite"
                        else (
                            "numeric values"
                            if defect == "nonnumeric"
                            else (
                                "dtype uint8"
                                if defect in {"float", "negative", "overflow"}
                                else "must have shape"
                            )
                        )
                    )
                    with self.assertRaisesRegex(ValueError, expected):
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

    def test_nonfinite_gripper_state_cannot_be_converted_to_binary(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                buffer = make_buffer()
                buffer.gripper_states[1] = value
                with self.assertRaisesRegex(ValueError, "gripper_states contains non-finite"):
                    build_episode_arrays(buffer, "pick up the red mug")

    def test_gripper_state_requires_one_scalar_per_step(self) -> None:
        buffer = make_buffer()
        buffer.gripper_states = [[0.], [1.], [1.]]
        with self.assertRaisesRegex(ValueError, "gripper_states must have shape"):
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

    def snapshot(self) -> tuple[dict, dict, dict]:
        with h5py.File(self.path, "r") as file:
            datasets = {}

            def read_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj[()]

            file["data"].visititems(read_dataset)
            return (
                dict(file.attrs),
                {name: dict(group.attrs) for name, group in file["data"].items()},
                datasets,
            )

    def assert_snapshot_unchanged(self, expected: tuple[dict, dict, dict]) -> None:
        actual = self.snapshot()
        self.assertEqual(actual[:2], expected[:2])
        self.assertEqual(set(actual[2]), set(expected[2]))
        for name in expected[2]:
            np.testing.assert_array_equal(actual[2][name], expected[2][name])

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

    def test_instruction_null_is_rejected_before_file_changes(self) -> None:
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "NULL"):
            append_episode_h5(self.path, 1, make_buffer(), dict(self.meta, instruction="bad\x00instruction"))
        self.assertEqual(self.path.read_bytes(), original)

    def test_dataset_failure_and_interrupt_remove_only_new_episode(self) -> None:
        original_create = h5py.Group.create_dataset
        for episode_id, error_type in enumerate((OSError, KeyboardInterrupt), start=1):
            with self.subTest(error=error_type.__name__):
                original = self.snapshot()

                def fail_after_write(group, name, *args, **kwargs):
                    dataset = original_create(group, name, *args, **kwargs)
                    if group.name == f"/data/demo_{episode_id}" and name == "action":
                        raise error_type("injected dataset failure")
                    return dataset

                with patch.object(h5py.Group, "create_dataset", new=fail_after_write):
                    with self.assertRaisesRegex(error_type, "injected dataset failure"):
                        append_episode_h5(self.path, episode_id, make_buffer(), self.meta)
                self.assert_snapshot_unchanged(original)
                append_episode_h5(self.path, episode_id, make_buffer(), self.meta)
                with h5py.File(self.path, "r") as file:
                    self.assertEqual(file.attrs["num_demos"], episode_id + 1)

    def test_episode_metadata_write_failure_allows_retry(self) -> None:
        original = self.snapshot()
        with self.assertRaisesRegex(ValueError, "embedded NULL"):
            append_episode_h5(self.path, 1, make_buffer(), dict(self.meta, bad_metadata="bad\x00value"))
        self.assert_snapshot_unchanged(original)
        append_episode_h5(self.path, 1, make_buffer(), self.meta)
        with h5py.File(self.path, "r") as file:
            self.assertEqual(file.attrs["num_demos"], 2)

    def test_root_metadata_failure_restores_values_and_absent_attributes(self) -> None:
        with h5py.File(self.path, "a") as file:
            del file.attrs["fields"]
            file.attrs["owner"] = "keep existing custom metadata"
        original = self.snapshot()
        original_setitem = h5py.AttributeManager.__setitem__
        failed = False

        def fail_after_count_write(attrs, name, value):
            nonlocal failed
            original_setitem(attrs, name, value)
            if name == "num_demos" and not failed:
                failed = True
                raise OSError("injected root metadata failure")

        with patch.object(h5py.AttributeManager, "__setitem__", new=fail_after_count_write):
            with self.assertRaisesRegex(OSError, "injected root metadata failure"):
                append_episode_h5(self.path, 1, make_buffer(), self.meta)
        self.assert_snapshot_unchanged(original)
        append_episode_h5(self.path, 1, make_buffer(), self.meta)
        with h5py.File(self.path, "r") as file:
            self.assertEqual(file.attrs["num_demos"], 2)

    def test_flush_failure_rolls_back_completed_episode(self) -> None:
        original = self.snapshot()
        with patch.object(h5py.File, "flush", side_effect=OSError("injected flush failure")):
            with self.assertRaisesRegex(OSError, "injected flush failure"):
                append_episode_h5(self.path, 1, make_buffer(), self.meta)
        self.assert_snapshot_unchanged(original)


if __name__ == "__main__":
    unittest.main()
