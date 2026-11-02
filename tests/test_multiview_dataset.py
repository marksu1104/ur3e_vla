"""CPU checks for complete MultiView writes and Python exception rollback."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from vla_sim.config import TARGET_KEYS
from vla_sim.multiview.data_collector_multiview import (
    MultiviewEpisodeBuffer,
    append_episode_h5_multiview,
)


def make_buffer() -> MultiviewEpisodeBuffer:
    def images(value):
        return [np.full((2, 3, 3), value + step, dtype=np.uint8) for step in range(3)]

    return MultiviewEpisodeBuffer(
        main_images=images(10),
        wrist_images=images(20),
        top_images=images(30),
        left_images=images(40),
        right_images=images(50),
        ee_poses=[[0., 0., 1., 1., 0., 0., 0.] for _ in range(3)],
        joint_positions=[[0.] * 6 for _ in range(3)],
        gripper_states=[0., 1., 1.],
        actions_7d=[[step * .01, 0., 0., 0., 0., 0., 1.] for step in range(3)],
        object_poses={
            name: [[index, step, 0., 1., 0., 0., 0.] for step in range(3)]
            for index, name in enumerate(TARGET_KEYS)
        },
        gripper_tip_positions=[[step * .01, 0., 1.] for step in range(3)],
    )


class MultiviewDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.meta = {
            "instruction": "pick up the red mug",
            "scene_profile": "multiview_scene_v1",
            "gripper_action_encoding": "logical_binary_0_open_1_closed",
            "num_steps": 3,
            "record_hz": 5.,
            "success": True,
        }

    def snapshot(self, path):
        with h5py.File(path, "r") as file:
            datasets = {}
            group_attrs = {}

            def read_object(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj[()]
                elif isinstance(obj, h5py.Group):
                    group_attrs[name] = dict(obj.attrs)

            file["data"].visititems(read_object)
            return dict(file.attrs), group_attrs, datasets

    def assert_snapshot_unchanged(self, path, expected):
        actual = self.snapshot(path)
        self.assertEqual(actual[:2], expected[:2])
        self.assertEqual(set(actual[2]), set(expected[2]))
        for name, values in expected[2].items():
            np.testing.assert_array_equal(actual[2][name], values)

    def prepare_case(self, path, existing):
        if existing:
            append_episode_h5_multiview(path, 0, make_buffer(), self.meta)
            with h5py.File(path, "a") as file:
                file.attrs["owner"] = "retain unrelated metadata"
                file.attrs["fields"] = "retain previous fields until append succeeds"
            return 1, self.snapshot(path)
        return 0, ({}, {}, {})

    def assert_rollback_then_retry(self, path, episode_id, expected):
        self.assert_snapshot_unchanged(path, expected)
        with h5py.File(path, "r") as file:
            self.assertNotIn(f"demo_{episode_id}", file["data"])
        append_episode_h5_multiview(path, episode_id, make_buffer(), self.meta)
        with h5py.File(path, "r") as file:
            self.assertEqual(file.attrs["num_demos"], episode_id + 1)
            self.assertIn(f"demo_{episode_id}/gripper_tip_position", file["data"])

    def test_success_has_five_views_poses_tip_and_metadata_in_one_open(self) -> None:
        path = self.root / "success.h5"
        buffer = make_buffer()
        with patch.object(h5py, "File", wraps=h5py.File) as open_file:
            append_episode_h5_multiview(path, 0, buffer, self.meta)
        self.assertEqual(open_file.call_count, 1)
        with h5py.File(path, "r") as file:
            group = file["data/demo_0"]
            self.assertEqual(file.attrs["num_demos"], 1)
            self.assertEqual(file.attrs["scene_profile"], "multiview_scene_v1")
            self.assertEqual(file.attrs["schema_version"], "v1")
            self.assertEqual(file.attrs["format"], "vla_demo_hdf5")
            for name, value in self.meta.items():
                self.assertEqual(group.attrs[name], value)
            for field, source in (
                ("image", "main_images"),
                ("other/hand_image", "wrist_images"),
                ("other/top_image", "top_images"),
                ("other/left_image", "left_images"),
                ("other/right_image", "right_images"),
            ):
                dataset = group[field]
                np.testing.assert_array_equal(dataset[:], np.asarray(getattr(buffer, source)))
                self.assertEqual(dataset.shape, (3, 2, 3, 3))
                self.assertEqual(dataset.dtype, np.dtype("uint8"))
                self.assertEqual(dataset.compression, "gzip")
                self.assertEqual(dataset.compression_opts, 4)
                self.assertEqual(dataset.chunks, (1, 2, 3, 3))
                self.assertIn(field, file.attrs["fields"])
            for name in TARGET_KEYS:
                dataset = group[f"object_poses/{name}"]
                np.testing.assert_array_equal(dataset[:], buffer.object_poses[name])
                self.assertEqual(dataset.dtype, np.dtype("float32"))
                self.assertEqual(dataset.shape, (3, 7))
            np.testing.assert_allclose(
                group["gripper_tip_position"][:], buffer.gripper_tip_positions
            )
            self.assertIn("gripper_tip_position", file.attrs["fields"])
            self.assertEqual(group["robot_state"].shape, (3, 15))
            self.assertEqual(group["action"].shape, (3, 8))
            self.assertEqual(group["task"][()].decode(), self.meta["instruction"])

    def test_each_extension_dataset_failure_or_interrupt_rolls_back_and_retries(self) -> None:
        original_create = h5py.Group.create_dataset
        extensions = (
            "other/top_image", "other/left_image", "other/right_image",
            *(f"object_poses/{name}" for name in TARGET_KEYS),
            "gripper_tip_position",
        )
        for existing in (False, True):
            for extension in extensions:
                for error_type in (OSError, KeyboardInterrupt):
                    with self.subTest(existing=existing, field=extension, error=error_type.__name__):
                        path = self.root / f"dataset_{existing}_{extension.replace('/', '_')}_{error_type.__name__}.h5"
                        episode_id, expected = self.prepare_case(path, existing)
                        target = f"/data/demo_{episode_id}/{extension}"

                        def fail_after_write(group, name, *args, **kwargs):
                            dataset = original_create(group, name, *args, **kwargs)
                            if dataset.name == target:
                                raise error_type("injected extension failure")
                            return dataset

                        with patch.object(h5py.Group, "create_dataset", new=fail_after_write):
                            with self.assertRaisesRegex(error_type, "injected extension failure"):
                                append_episode_h5_multiview(path, episode_id, make_buffer(), self.meta)
                        self.assert_rollback_then_retry(path, episode_id, expected)

    def test_extension_fields_and_count_failure_restore_root_metadata(self) -> None:
        original_setitem = h5py.AttributeManager.__setitem__
        for existing in (False, True):
            for field in ("fields", "num_demos"):
                with self.subTest(existing=existing, field=field):
                    path = self.root / f"metadata_{existing}_{field}.h5"
                    episode_id, expected = self.prepare_case(path, existing)
                    failed = False

                    def fail_after_write(attrs, name, value):
                        nonlocal failed
                        original_setitem(attrs, name, value)
                        is_extension = name != "fields" or "other/top_image" in str(value)
                        if name == field and is_extension and not failed:
                            failed = True
                            raise OSError("injected root metadata failure")

                    with patch.object(h5py.AttributeManager, "__setitem__", new=fail_after_write):
                        with self.assertRaisesRegex(OSError, "injected root metadata failure"):
                            append_episode_h5_multiview(path, episode_id, make_buffer(), self.meta)
                    self.assert_rollback_then_retry(path, episode_id, expected)

    def test_flush_failure_rolls_back_extensions_and_allows_retry(self) -> None:
        for existing in (False, True):
            with self.subTest(existing=existing):
                path = self.root / f"flush_{existing}.h5"
                episode_id, expected = self.prepare_case(path, existing)
                with patch.object(h5py.File, "flush", side_effect=OSError("injected flush failure")):
                    with self.assertRaisesRegex(OSError, "injected flush failure"):
                        append_episode_h5_multiview(path, episode_id, make_buffer(), self.meta)
                self.assert_rollback_then_retry(path, episode_id, expected)

    def test_duplicate_episode_preserves_file_bytes(self) -> None:
        path = self.root / "duplicate.h5"
        append_episode_h5_multiview(path, 0, make_buffer(), self.meta)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            append_episode_h5_multiview(path, 0, make_buffer(), self.meta)
        self.assertEqual(path.read_bytes(), original)

    def assert_invalid_before_open(self, buffer, expected_message):
        for existing in (False, True):
            with self.subTest(existing=existing):
                path = self.root / "invalid.h5"
                if existing:
                    append_episode_h5_multiview(path, 0, make_buffer(), self.meta)
                    original = path.read_bytes()
                with patch.object(h5py, "File", side_effect=AssertionError("file must not be opened")):
                    with self.assertRaisesRegex(ValueError, expected_message):
                        append_episode_h5_multiview(path, 1, buffer, self.meta)
                if existing:
                    self.assertEqual(path.read_bytes(), original)
                    path.unlink()
                else:
                    self.assertFalse(path.exists())

    def test_each_camera_must_match_main_length_and_shape_before_open(self) -> None:
        for name in ("wrist", "top", "left", "right"):
            for defect in ("short", "shape", "channels"):
                with self.subTest(camera=name, defect=defect):
                    buffer = make_buffer()
                    values = getattr(buffer, f"{name}_images")
                    if defect == "short":
                        values.pop()
                    else:
                        shape = (3, 3, 3) if defect == "shape" else (2, 3, 4)
                        setattr(buffer, f"{name}_images", [np.zeros(shape) for _ in range(3)])
                    self.assert_invalid_before_open(buffer, f"{name}_images must have shape")

    def test_nonfinite_camera_values_rejected_before_uint8_conversion(self) -> None:
        for name in ("main", "wrist", "top", "left", "right"):
            with self.subTest(camera=name):
                buffer = make_buffer()
                values = [image.astype(float) for image in getattr(buffer, f"{name}_images")]
                values[0][0, 0, 0] = np.nan
                setattr(buffer, f"{name}_images", values)
                self.assert_invalid_before_open(buffer, f"{name}_images contains non-finite")

    def test_each_camera_rejects_lossy_float_or_out_of_range_conversion(self) -> None:
        for name in ("main", "wrist", "top", "left", "right"):
            for value in (0.5, -1.0, 256.0):
                with self.subTest(camera=name, value=value):
                    buffer = make_buffer()
                    values = [
                        image.astype(np.float32)
                        for image in getattr(buffer, f"{name}_images")
                    ]
                    values[0][0, 0, 0] = value
                    setattr(buffer, f"{name}_images", values)
                    self.assert_invalid_before_open(
                        buffer, f"{name}_images must have dtype uint8"
                    )

    def test_main_image_shape_rejected_before_open(self) -> None:
        for defect in ("short", "channels"):
            with self.subTest(defect=defect):
                buffer = make_buffer()
                if defect == "short":
                    buffer.main_images.pop()
                else:
                    buffer.main_images = [np.zeros((2, 3, 4)) for _ in range(3)]
                self.assert_invalid_before_open(buffer, "main_images must have shape")

    def test_object_pose_length_width_missing_key_and_nonfinite_rejected_before_open(self) -> None:
        for name in TARGET_KEYS:
            for defect in ("short", "width", "missing", "nonfinite"):
                with self.subTest(object=name, defect=defect):
                    buffer = make_buffer()
                    if defect == "short":
                        buffer.object_poses[name].pop()
                    elif defect == "width":
                        buffer.object_poses[name] = [[0.] * 6 for _ in range(3)]
                    elif defect == "missing":
                        del buffer.object_poses[name]
                    else:
                        buffer.object_poses[name][0][0] = np.inf
                    self.assert_invalid_before_open(buffer, f"object_poses/{name}")

    def test_tip_length_width_and_nonfinite_rejected_before_open(self) -> None:
        for defect in ("short", "width", "nonfinite"):
            with self.subTest(defect=defect):
                buffer = make_buffer()
                if defect == "short":
                    buffer.gripper_tip_positions.pop()
                elif defect == "width":
                    buffer.gripper_tip_positions = [[0.] * 2 for _ in range(3)]
                else:
                    buffer.gripper_tip_positions[0][0] = -np.inf
                self.assert_invalid_before_open(buffer, "gripper_tip_positions")


if __name__ == "__main__":
    unittest.main()
