"""Execute the actual rollout entry point with CPU Isaac and HTTP substitutes."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests
import torch


class PredictionThread:
    """Deterministic prediction scheduling, including a result arriving at join."""

    instances = []
    finish_at_join = False
    never_finish = False

    def __init__(self, *, target, args, daemon):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.alive = False
        self.join_timeouts = []
        self.instances.append(self)

    def start(self):
        self.alive = True
        if not self.finish_at_join and not self.never_finish:
            self._finish()

    def _finish(self):
        self.target(*self.args)
        self.alive = False

    def join(self, timeout):
        self.join_timeouts.append(timeout)
        if self.alive and self.finish_at_join and not self.never_finish:
            self._finish()

    def is_alive(self):
        return self.alive


class RunVLATest(unittest.TestCase):
    def setUp(self):
        PredictionThread.instances = []
        PredictionThread.finish_at_join = False
        PredictionThread.never_finish = False
        self.path = Path(__file__).resolve().parents[1] / "scripts/run_vla.py"
        self.log = Mock()
        self.app = Mock()
        self.app.is_running.return_value = True
        self.close_app = Mock()
        self.controller = Mock()
        self.controller.ee_body_idx = 0
        pose = torch.tensor([[[0.30, 0.13, 1.35, 1, 0, 0, 0]]])
        self.camera = types.SimpleNamespace(
            data=types.SimpleNamespace(output={"rgb": torch.zeros((1, 8, 12, 3))})
        )
        self.runtime = Mock()
        self.runtime.start.return_value = self.runtime
        self.runtime.robot_controller = self.controller
        self.runtime.robot = types.SimpleNamespace(
            data=types.SimpleNamespace(body_link_pose_w=pose)
        )
        self.runtime.scene = {
            "camera_policy": self.camera,
            "red_mug": types.SimpleNamespace(
                data=types.SimpleNamespace(root_pos_w=torch.zeros((1, 3)))
            ),
        }
        self.runtime.sim_dt = 1 / 60
        self.runtime_factory = Mock(return_value=self.runtime)
        isaac_app = types.ModuleType("vla_sim.isaac_app")
        isaac_app.args_cli = types.SimpleNamespace(target="red_mug", device="cpu")
        isaac_app.boot_app = Mock(return_value=self.app)
        isaac_app.close_app = self.close_app
        isaac_app.log = self.log
        simulation = types.ModuleType("vla_sim.simulation")
        simulation.RuntimeOptions = types.SimpleNamespace
        simulation.SimulationRuntime = self.runtime_factory
        simulation.as_torch = torch.as_tensor
        simulation.pose_wxyz_from_sim = lambda value: value
        self.modules = {
            "vla_sim.isaac_app": isaac_app,
            "vla_sim.simulation": simulation,
        }
        self.health = self.response({"status": "ok"})
        self.predict_response = self.response({"action": [0] * 7})

    @staticmethod
    def response(payload):
        response = Mock()
        response.json.return_value = payload
        return response

    def run_script(self, *extra, post_effect=None):
        post_kwargs = (
            {"return_value": self.predict_response}
            if post_effect is None
            else {"side_effect": post_effect}
        )
        with (
            patch.dict(sys.modules, self.modules),
            patch.object(
                sys, "argv",
                [str(self.path), "--max-steps", "3", "--vla-timeout", "0.25", *extra],
            ),
            patch("threading.Thread", PredictionThread),
            patch("vla_sim.vla_client.requests.get", return_value=self.health),
            patch("vla_sim.vla_client.requests.post", **post_kwargs),
        ):
            return runpy.run_path(str(self.path), run_name="__main__")

    def assert_cleanup(self):
        self.close_app.assert_called_once_with()
        for thread in PredictionThread.instances:
            self.assertEqual(thread.join_timeouts, [1.25])

    def test_health_failure_exits_before_scene_creation_and_closes_app(self):
        self.health = self.response({"status": "loading"})
        with self.assertRaisesRegex(RuntimeError, "unavailable or not ready"):
            self.run_script()
        self.runtime_factory.assert_not_called()
        self.assert_cleanup()

    def test_valid_zero_action_is_consumed_and_rollout_succeeds(self):
        self.run_script()
        self.assertEqual(self.runtime.step.call_count, 123)
        self.assertEqual(self.controller.set_pose_target.call_count, 3)
        for call in self.controller.set_pose_target.call_args_list:
            pos, quat = call.args
            torch.testing.assert_close(pos, torch.tensor([0.30, 0.13, 1.35]))
            self.assertTrue(torch.isfinite(quat).all())
        self.assertTrue(any("consumed_actions=1" in str(call.args) for call in self.log.call_args_list))
        self.assert_cleanup()

    def test_invalid_response_and_timeout_make_bounded_rollout_fail(self):
        for effect in (
            [self.response({"action": []})],
            [self.response({"action": [float("nan"), 0, 0, 0, 0, 0, 0]})],
            [requests.Timeout("prediction timed out")],
        ):
            with self.subTest(effect=effect):
                self.setUp()
                with self.assertRaisesRegex(RuntimeError, "errors=1"):
                    self.run_script(post_effect=effect)
                for call in self.controller.set_pose_target.call_args_list:
                    self.assertTrue(torch.isfinite(call.args[0]).all())
                self.assert_cleanup()

    def test_background_exception_is_not_ignored_in_unlimited_mode(self):
        error = ValueError("model worker failed")
        with patch("vla_sim.vla_client.VLAClient.predict", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "prediction worker failed") as caught:
                self.run_script("--max-steps", "0")
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(self.runtime.step.call_count, 120)
        self.assert_cleanup()

    def test_missing_camera_produces_no_requests_and_fails(self):
        self.camera.data.output = {}
        with self.assertRaisesRegex(RuntimeError, "requests=0.*consumed_actions=0"):
            self.run_script()
        self.assertEqual(PredictionThread.instances, [])
        self.assert_cleanup()

    def test_response_only_at_join_does_not_claim_consumed_action(self):
        PredictionThread.finish_at_join = True
        with self.assertRaisesRegex(RuntimeError, "requests=1 errors=0 consumed_actions=0"):
            self.run_script("--max-steps", "1")
        self.assert_cleanup()

    def test_worker_still_running_after_join_fails(self):
        PredictionThread.never_finish = True
        with self.assertRaisesRegex(RuntimeError, "did not finish before shutdown"):
            self.run_script("--max-steps", "1")
        self.assert_cleanup()

    def test_app_stopping_before_requested_steps_is_incomplete(self):
        self.app.is_running.side_effect = [True, False]
        with self.assertRaisesRegex(RuntimeError, "stopped early: steps=1 expected=3"):
            self.run_script()
        self.assertEqual(self.runtime.step.call_count, 121)
        self.assert_cleanup()

    def test_keyboard_interrupt_joins_worker_and_closes_app(self):
        PredictionThread.finish_at_join = True
        self.app.is_running.side_effect = [True, KeyboardInterrupt()]
        self.run_script()
        self.assertTrue(any("Ctrl+C received" in str(call.args) for call in self.log.call_args_list))
        self.assert_cleanup()


if __name__ == "__main__":
    unittest.main()
