"""Exercise shutdown status without launching Isaac or allocating a GPU."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class FakeLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        pass


class FastShutdownApp:
    def close(self, *, wait_for_replicator, exit_code):
        raise SystemExit(exit_code)


class ShutdownStatusTest(unittest.TestCase):
    def setUp(self):
        app_module = types.ModuleType("isaaclab.app")
        app_module.AppLauncher = FakeLauncher
        path = Path(__file__).resolve().parents[1] / "vla_sim/isaac_app.py"
        spec = importlib.util.spec_from_file_location("shutdown_test_helper", path)
        self.helper = importlib.util.module_from_spec(spec)
        with (
            patch.dict(sys.modules, {"isaaclab.app": app_module}),
            patch.object(sys, "argv", ["shutdown_test"]),
            patch.dict(os.environ, {"ISAAC_ROS2_KIT_CACHE_DIR": ""}),
        ):
            spec.loader.exec_module(self.helper)
        self.helper._app = FastShutdownApp()

    def test_success_returns_zero(self):
        with self.assertRaises(SystemExit) as result:
            self.helper.close_app()
        self.assertEqual(result.exception.code, 0)

    def test_unhandled_error_is_reported_and_returns_nonzero(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as result:
            try:
                raise ValueError("incomplete collection")
            finally:
                self.helper.close_app()
        self.assertEqual(result.exception.code, 1)
        self.assertIn("ValueError: incomplete collection", stderr.getvalue())

    def test_explicit_exit_status_is_preserved(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            try:
                raise SystemExit(7)
            finally:
                self.helper.close_app()
        self.assertEqual(result.exception.code, 7)

    def test_interrupt_returns_130(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            try:
                raise KeyboardInterrupt()
            finally:
                self.helper.close_app()
        self.assertEqual(result.exception.code, 130)


if __name__ == "__main__":
    unittest.main()
