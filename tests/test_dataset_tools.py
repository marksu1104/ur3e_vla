"""Exercise H5 tool failures that must preserve existing demonstration files."""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from scripts.tools import subset_h5_demos


class DatasetToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source.h5"
        self.output = self.root / "output.h5"
        with h5py.File(self.source, "w") as file:
            file.attrs["num_demos"] = 1
            file.create_dataset("data/demo_0/state", data=np.arange(3))

    def subset(self, *, output: Path | None = None, episodes: int = 1, overwrite: bool = True) -> int:
        argv = [
            "subset_h5_demos.py", "--input", str(self.source),
            "--output", str(output if output is not None else self.output),
            "--episodes", str(episodes),
        ]
        if overwrite:
            argv.append("--overwrite")
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return subset_h5_demos.main()

    def assert_no_temporary_files(self) -> None:
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_same_file_and_aliases_cannot_delete_source(self) -> None:
        original = self.source.read_bytes()
        symlink = self.root / "symlink.h5"
        symlink.symlink_to(self.source)
        hardlink = self.root / "hardlink.h5"
        os.link(self.source, hardlink)
        for output in (self.source, symlink, hardlink):
            with self.subTest(output=output.name):
                with self.assertRaisesRegex(ValueError, "different files"):
                    self.subset(output=output)
                self.assertEqual(self.source.read_bytes(), original)
                self.assertTrue(output.samefile(self.source))

    def test_excess_episode_request_preserves_output(self) -> None:
        original = b"previous dataset must survive"
        self.output.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "only has 1"):
            self.subset(episodes=2)
        self.assertEqual(self.output.read_bytes(), original)
        self.assert_no_temporary_files()

    def test_invalid_source_preserves_output(self) -> None:
        original = b"previous dataset must survive"
        self.output.write_bytes(original)
        with h5py.File(self.source, "w") as file:
            file.create_dataset("data", data=1)
        with self.assertRaisesRegex(KeyError, "/data group"):
            self.subset()
        self.assertEqual(self.output.read_bytes(), original)
        self.assert_no_temporary_files()

    def test_copy_failure_preserves_output_and_cleans_temporary_file(self) -> None:
        original = b"previous dataset must survive"
        self.output.write_bytes(original)
        with h5py.File(self.source, "a") as file:
            file["data/demo_1"] = h5py.SoftLink("/missing")
        with self.assertRaises(KeyError):
            self.subset(episodes=2)
        self.assertEqual(self.output.read_bytes(), original)
        self.assert_no_temporary_files()

    def test_existing_output_requires_overwrite(self) -> None:
        original = b"previous dataset must survive"
        self.output.write_bytes(original)
        with self.assertRaises(FileExistsError):
            self.subset(overwrite=False)
        self.assertEqual(self.output.read_bytes(), original)

    def test_completed_copy_replaces_output_and_preserves_source(self) -> None:
        self.output.write_bytes(b"previous dataset")
        original = self.source.read_bytes()
        self.assertEqual(self.subset(), 0)
        self.assertEqual(self.source.read_bytes(), original)
        with h5py.File(self.output, "r") as file:
            self.assertEqual(file.attrs["num_demos"], 1)
            np.testing.assert_array_equal(file["data/demo_0/state"][:], np.arange(3))
        self.assert_no_temporary_files()

    def test_validator_rejects_dangling_episode_links_with_short_failure(self) -> None:
        validator = Path(subset_h5_demos.__file__).with_name("validate_h5.py")
        links = (
            h5py.SoftLink("/missing"),
            h5py.ExternalLink(str(self.root / "missing.h5"), "/episode"),
        )
        for link in links:
            with self.subTest(link=type(link).__name__):
                with h5py.File(self.source, "w") as file:
                    file.attrs.update(
                        scene_profile="canonical_scene_v1", schema_version="v1",
                        format="vla_demo_hdf5", num_demos=1,
                    )
                    file.create_group("data")["demo_0"] = link
                original = self.source.read_bytes()
                result = subprocess.run(
                    [sys.executable, str(validator), str(self.source)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr.startswith("FAIL:"), result.stderr)
                self.assertIn("/data/demo_0: expected group", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(self.source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
