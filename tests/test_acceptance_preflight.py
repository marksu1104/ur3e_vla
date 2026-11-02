"""Check that acceptance preflight failures cannot produce a green report."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.tools.acceptance_preflight import MINIMUM_UNIT_TESTS, h5_check, run_check


class AcceptancePreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def run_unit_output(self, text: str) -> dict:
        command = [sys.executable, "-c", f"import sys; sys.stderr.write({text!r})"]
        return run_check(
            "unit_tests", command, self.root, minimum_tests=MINIMUM_UNIT_TESTS
        )

    def test_successful_process_with_too_few_tests_fails(self) -> None:
        result = self.run_unit_output(
            f"Ran {MINIMUM_UNIT_TESTS - 1} tests in 0.01s\n\nOK\n"
        )
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["passed"])
        self.assertIn(f"expected at least {MINIMUM_UNIT_TESTS}", result["error"])

    def test_successful_process_with_a_skipped_test_fails(self) -> None:
        result = self.run_unit_output(
            f"Ran {MINIMUM_UNIT_TESTS} tests in 0.01s\n\nOK (skipped=1)\n"
        )
        self.assertEqual((result["tests"], result["skipped"]), (MINIMUM_UNIT_TESTS, 1))
        self.assertFalse(result["passed"])
        self.assertIn("1 unit tests were skipped", result["error"])

    def test_minimum_tests_without_skips_passes(self) -> None:
        result = self.run_unit_output(
            f"Ran {MINIMUM_UNIT_TESTS} tests in 0.01s\n\nOK\n"
        )
        self.assertTrue(result["passed"])
        self.assertEqual((result["tests"], result["skipped"]), (MINIMUM_UNIT_TESTS, 0))
        self.assertNotIn("error", result)

    def test_validator_mutation_or_deletion_is_detected_after_failure(self) -> None:
        for mode in ("mutate", "delete"):
            with self.subTest(mode=mode):
                path = self.root / f"{mode}.h5"
                path.write_bytes(b"original")
                before = hashlib.sha256(b"original").hexdigest()

                def fail_after_change(candidate: Path) -> None:
                    if mode == "mutate":
                        candidate.write_bytes(b"changed")
                    else:
                        candidate.unlink()
                    raise ValueError("synthetic invalid H5")

                with patch(
                    "scripts.tools.validate_h5.validate_h5", side_effect=fail_after_change
                ):
                    result = h5_check(path)

                self.assertFalse(result["passed"])
                self.assertEqual(result["sha256_before"], before)
                self.assertFalse(result["unchanged"])
                self.assertIn("validation failed: synthetic invalid H5", result["error"])
                if mode == "mutate":
                    self.assertIsNotNone(result["sha256_after"])
                    self.assertIn("H5 changed while it was being checked", result["error"])
                else:
                    self.assertIsNone(result["sha256_after"])
                    self.assertIn("H5 unavailable after validation", result["error"])

    def test_invalid_unchanged_h5_reports_both_hashes(self) -> None:
        path = self.root / "invalid.h5"
        path.write_bytes(b"invalid but unchanged")
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch(
            "scripts.tools.validate_h5.validate_h5",
            side_effect=ValueError("synthetic invalid H5"),
        ):
            result = h5_check(path)

        self.assertFalse(result["passed"])
        self.assertTrue(result["unchanged"])
        self.assertEqual(result["sha256_before"], expected)
        self.assertEqual(result["sha256_after"], expected)
        self.assertIn("validation failed: synthetic invalid H5", result["error"])


if __name__ == "__main__":
    unittest.main()
