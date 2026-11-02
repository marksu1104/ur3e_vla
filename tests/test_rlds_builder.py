"""Exercise the real builder methods without loading TensorFlow or USE.

The TFDS serialization path is checked separately in rlds_env. These tests
compile the builder class from its source with a minimal TFDS base/version stub.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "rlds_builder/ur3e_vla_dataset/ur3e_vla_dataset_dataset_builder.py"
)
ENCODING = "logical_binary_0_open_1_closed"


def load_builder_methods():
    node = next(item for item in ast.parse(SOURCE.read_text()).body if isinstance(item, ast.ClassDef))
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    namespace = dict(
        Path=Path, hashlib=hashlib, os=os, random=random, np=np,
        tfds=SimpleNamespace(core=SimpleNamespace(GeneratorBasedBuilder=object, Version=str)),
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["Ur3eVlaDataset"]


Builder = load_builder_methods()


class RldsBuilderTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.builder = Builder.__new__(Builder)
        self.builder._embed = lambda tasks: [SimpleNamespace(numpy=lambda: np.zeros(512, np.float32))]

    def make_h5(self, name, count=2, *, profile="canonical_scene_v1", failed=()):
        path = self.root / name / "demos.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as file:
            file.attrs["scene_profile"] = profile
            data = file.create_group("data")
            for index in range(count):
                demo = data.create_group(f"demo_{index}")
                demo.attrs["success"] = index not in failed
                demo.attrs["scene_profile"] = profile
                demo.attrs["gripper_action_encoding"] = ENCODING
                demo.create_dataset("image", data=np.zeros((2, 2, 3, 3), np.uint8))
                demo.create_dataset("other/hand_image", data=np.zeros((2, 2, 3, 3), np.uint8))
                demo.create_dataset("robot_state", data=np.tile(np.arange(15, dtype=np.float32), (2, 1)))
                demo.create_dataset("action", data=np.array([[.01, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 1]], np.float32))
                demo.create_dataset("task", data="pick up the red mug", dtype=h5py.string_dtype())
        return path

    def split(self, paths, ratio=.1, seed=42):
        env = {
            "UR3E_VLA_H5_PATHS": ",".join(map(str, paths)),
            "UR3E_VLA_VAL_RATIO": str(ratio),
            "UR3E_VLA_SPLIT_SEED": str(seed),
        }
        with patch.dict(os.environ, env):
            return {name: list(examples) for name, examples in self.builder._split_generators(None).items()}

    @staticmethod
    def identities(splits):
        return {
            name: [(key, episode["episode_metadata"]) for key, episode in examples]
            for name, examples in splits.items()
        }

    def test_same_target_demo_ids_from_different_runs_have_unique_keys(self):
        paths = [self.make_h5("run_a/red_mug"), self.make_h5("run_b/red_mug")]
        examples = self.split(paths, ratio=0)["train"]
        keys = [key for key, _ in examples]
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(set(keys)), 4)
        self.assertEqual(keys, [key for key, _ in self.split(paths, ratio=0)["train"]])

    def test_duplicate_resolved_paths_are_counted_once(self):
        path = self.make_h5("red_mug")
        alias = self.root / "alias.h5"
        alias.symlink_to(path)
        splits = self.split([path, path, alias], ratio=.5)
        self.assertEqual({name: len(examples) for name, examples in splits.items()}, {"train": 1, "val": 1})
        self.assertEqual(Builder._source_id(path), Builder._source_id(alias))

    def test_split_is_reproducible_independent_of_input_order(self):
        paths = [self.make_h5(target, count=4) for target in ("spoon", "red_mug", "bowl")]
        first = self.identities(self.split(paths, ratio=.25))
        self.assertEqual(first, self.identities(self.split(paths, ratio=.25)))
        self.assertEqual(first, self.identities(self.split(list(reversed(paths)), ratio=.25)))
        self.assertEqual({name: len(examples) for name, examples in first.items()}, {"train": 9, "val": 3})
        self.assertTrue(set(key for key, _ in first["train"]).isdisjoint(key for key, _ in first["val"]))
        self.assertNotEqual(first, self.identities(self.split(paths, ratio=.25, seed=7)))

    def test_failed_episodes_are_removed_before_split(self):
        path = self.make_h5("red_mug", count=3, failed=(1,))
        splits = self.split([path], ratio=.5)
        self.assertEqual({name: len(examples) for name, examples in splits.items()}, {"train": 1, "val": 1})
        self.assertEqual(
            {episode["episode_metadata"]["demo_id"] for examples in splits.values() for _, episode in examples},
            {"demo_0", "demo_2"},
        )

    def test_single_success_is_train_only_even_with_high_val_ratio(self):
        path = self.make_h5("red_mug", failed=(1,))
        splits = self.split([path], ratio=.99)
        self.assertEqual(set(splits), {"train"})
        self.assertEqual(len(splits["train"]), 1)

    def test_nonzero_ratio_reserves_train_and_val_for_two_episodes(self):
        path = self.make_h5("red_mug")
        for ratio in (.001, .1, .99):
            with self.subTest(ratio=ratio):
                self.assertEqual(
                    {name: len(examples) for name, examples in self.split([path], ratio=ratio).items()},
                    {"train": 1, "val": 1},
                )

    def test_zero_ratio_keeps_all_successes_in_train(self):
        path = self.make_h5("red_mug", count=3)
        splits = self.split([path], ratio=0)
        self.assertEqual(set(splits), {"train"})
        self.assertEqual(len(splits["train"]), 3)

    def test_empty_or_all_failed_input_is_rejected(self):
        for path in (self.make_h5("empty", count=0), self.make_h5("failed", failed=(0, 1))):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "No successful episodes"):
                self.split([path])

    def test_each_supported_scene_can_be_built_separately(self):
        for profile in ("canonical_scene_v1", "multiview_scene_v1"):
            path = self.make_h5(profile, profile=profile)
            with self.subTest(profile=profile):
                self.assertEqual(sum(map(len, self.split([path]).values())), 2)

    def test_mixing_supported_scenes_is_rejected(self):
        paths = [self.make_h5("canonical"), self.make_h5("multiview", profile="multiview_scene_v1")]
        with self.assertRaisesRegex(ValueError, "Cannot mix scene profiles"):
            self.split(paths)

    def test_unknown_or_missing_file_scene_is_rejected(self):
        for profile in ("legacy", ""):
            path = self.make_h5(profile or "missing", profile=profile)
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "unsupported scene_profile"):
                self.split([path])

    def test_episode_scene_must_match_file(self):
        path = self.make_h5("red_mug")
        with h5py.File(path, "a") as file:
            file["data/demo_0"].attrs["scene_profile"] = "multiview_scene_v1"
        with self.assertRaisesRegex(ValueError, "scene_profile does not match"):
            self.split([path])

    def test_unknown_or_missing_episode_gripper_encoding_is_rejected(self):
        for encoding in ("physical_joint", None):
            path = self.make_h5(str(encoding))
            with h5py.File(path, "a") as file:
                attrs = file["data/demo_0"].attrs
                if encoding is None:
                    del attrs["gripper_action_encoding"]
                else:
                    attrs["gripper_action_encoding"] = encoding
            with self.subTest(encoding=encoding), self.assertRaisesRegex(ValueError, "gripper_action_encoding"):
                self.split([path])

    def test_file_gripper_encoding_is_checked_when_present(self):
        path = self.make_h5("red_mug")
        with h5py.File(path, "a") as file:
            file.attrs["gripper_action_encoding"] = "physical_joint"
        with self.assertRaisesRegex(ValueError, "gripper_action_encoding"):
            self.split([path])

    def test_state_action_and_episode_flags_preserve_h5_contract(self):
        path = self.make_h5("red_mug")
        episode = self.split([path], ratio=0)["train"][0][1]
        steps = episode["steps"]
        with h5py.File(path, "r") as file:
            np.testing.assert_array_equal(steps[0]["observation"]["state"], file["data/demo_0/robot_state"][0])
            np.testing.assert_array_equal(steps[0]["action"], file["data/demo_0/action"][0])
        self.assertEqual([step["is_first"] for step in steps], [True, False])
        self.assertEqual([step["is_last"] for step in steps], [False, True])
        self.assertEqual([step["is_terminal"] for step in steps], [False, True])
        self.assertEqual([step["reward"] for step in steps], [0, 1])
        self.assertEqual(steps[0]["language_instruction"], "pick up the red mug")
        self.assertEqual(steps[0]["language_embedding"].shape, (512,))

    def test_invalid_val_ratio_is_rejected(self):
        path = self.make_h5("red_mug")
        for ratio in (-.1, 1, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "VAL_RATIO"):
                self.split([path], ratio=ratio)


if __name__ == "__main__":
    unittest.main()
