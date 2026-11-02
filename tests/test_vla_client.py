"""Verify inference failure handling without a server, Isaac, ROS, or GPU."""

from __future__ import annotations

import base64
import ast
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests
from PIL import Image

from vla_sim.vla_client import VLAClient


class VLAClientTest(unittest.TestCase):
    def setUp(self):
        self.client = VLAClient("http://mock/", timeout=0.25)
        self.rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 255
        self.log = Mock()

    @staticmethod
    def response(payload):
        response = Mock()
        response.json.return_value = payload
        return response

    def predict(self):
        return self.client.predict(
            self.rgb, "pick the mug", unnorm_key="test_dataset", log=self.log
        )

    def test_success_sends_jpeg_and_preserves_logical_gripper(self):
        for gripper in (0.0, 1.0):
            with self.subTest(gripper=gripper):
                action = [0.02, 0, 0, 0, 0, 0, gripper]
                with patch(
                    "vla_sim.vla_client.requests.post",
                    return_value=self.response({"action": action}),
                ) as post:
                    result = self.predict()
                np.testing.assert_allclose(result, action)
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(post.call_args.args, ("http://mock/predict",))
                self.assertEqual(post.call_args.kwargs["timeout"], 0.25)
                payload = post.call_args.kwargs["json"]
                self.assertEqual(payload["instruction"], "pick the mug")
                self.assertEqual(payload["unnorm_key"], "test_dataset")
                with Image.open(io.BytesIO(base64.b64decode(payload["image_b64"]))) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.size, (12, 8))
                    self.assertEqual(image.mode, "RGB")
        self.assertEqual(self.client.stats, {"requests": 2, "errors": 0, "error_rate": 0.0})

    def test_timeout_after_success_does_not_repeat_previous_delta(self):
        with patch(
            "vla_sim.vla_client.requests.post",
            side_effect=[
                self.response({"action": [0.02, 0, 0, 0, 0, 0, 1]}),
                requests.Timeout("inference timed out"),
                requests.Timeout("inference timed out again"),
            ],
        ):
            first = self.predict()
            first[0] = 9
            first[6] = 0
            failed = self.predict()
            held_grip = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            np.testing.assert_array_equal(failed, held_grip)
            failed[0] = 8
            failed[6] = 0
            np.testing.assert_array_equal(self.predict(), held_grip)
        self.assertAlmostEqual(float(self.client._last_action[0]), 0.02)
        self.assertEqual(float(self.client._last_action[6]), 1.0)
        self.assertEqual(self.client.stats["requests"], 3)
        self.assertEqual(self.client.stats["errors"], 2)
        self.assertAlmostEqual(self.client.stats["error_rate"], 2 / 3)

    def test_invalid_action_is_counted_and_returns_zero(self):
        payloads = (
            {},
            {"action": None},
            {"action": 0},
            {"action": [0, 0]},
            {"action": [0] * 8},
            {"action": [[0] * 7]},
            {"action": [float("nan"), 0, 0, 0, 0, 0, 0]},
            {"action": [0, 0, 0, float("inf"), 0, 0, 0]},
            {"action": [0, 0, 0, 0, 0, 0, float("-inf")]},
            {"action": ["invalid", 0, 0, 0, 0, 0, 0]},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                self.client = VLAClient("http://mock", timeout=0.25)
                with patch(
                    "vla_sim.vla_client.requests.post",
                    return_value=self.response(payload),
                ):
                    result = self.predict()
                np.testing.assert_array_equal(result, np.zeros(7, dtype=np.float32))
                self.assertEqual(self.client.stats, {"requests": 1, "errors": 1, "error_rate": 1.0})

    def test_invalid_response_preserves_last_valid_grip_and_cache(self):
        valid = [0.02, 0, 0, 0, 0, 0, 1]
        with patch(
            "vla_sim.vla_client.requests.post",
            side_effect=[
                self.response({"action": valid}),
                self.response({"action": [float("nan"), 0, 0, 0, 0, 0, 0]}),
            ],
        ):
            self.predict()
            np.testing.assert_array_equal(self.predict(), [0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(self.client._last_action, valid)
        self.assertEqual(self.client.stats, {"requests": 2, "errors": 1, "error_rate": 0.5})

    def test_http_and_json_failures_are_counted_once(self):
        http_error = self.response({})
        http_error.raise_for_status.side_effect = requests.HTTPError("500")
        invalid_json = self.response({})
        invalid_json.json.side_effect = ValueError("invalid JSON")
        with patch(
            "vla_sim.vla_client.requests.post",
            side_effect=[requests.ConnectionError("offline"), http_error, invalid_json],
        ):
            for _ in range(3):
                np.testing.assert_array_equal(self.predict(), np.zeros(7, dtype=np.float32))
        self.assertEqual(self.client.stats, {"requests": 3, "errors": 3, "error_rate": 1.0})

    def test_encoding_failure_is_a_failed_attempt_without_http(self):
        with (
            patch("vla_sim.vla_client.encode_rgb_jpeg_base64", side_effect=ValueError("bad image")),
            patch("vla_sim.vla_client.requests.post") as post,
        ):
            np.testing.assert_array_equal(self.predict(), np.zeros(7, dtype=np.float32))
        post.assert_not_called()
        self.assertEqual(self.client.stats, {"requests": 1, "errors": 1, "error_rate": 1.0})

    def test_valid_response_recovers_after_failure(self):
        expected = [0, 0.01, 0, 0, 0, 0, 1]
        with patch(
            "vla_sim.vla_client.requests.post",
            side_effect=[requests.Timeout("timeout"), self.response({"action": expected})],
        ):
            np.testing.assert_array_equal(self.predict(), np.zeros(7, dtype=np.float32))
            np.testing.assert_allclose(self.predict(), expected)
        self.assertEqual(self.client.stats, {"requests": 2, "errors": 1, "error_rate": 0.5})

    def test_tenth_success_logs_without_changing_counts(self):
        with patch(
            "vla_sim.vla_client.requests.post",
            return_value=self.response({"action": [0] * 7, "latency_ms": 12.5}),
        ):
            for _ in range(10):
                self.predict()
        self.assertEqual(self.client.stats, {"requests": 10, "errors": 0, "error_rate": 0.0})
        self.log.assert_called_once()
        self.assertIn("#10", self.log.call_args.args[0])

    def test_health_check_uses_configured_timeout_and_requires_ready(self):
        for status, ready in (("ok", True), ("loading", False)):
            with self.subTest(status=status), patch(
                "vla_sim.vla_client.requests.get",
                return_value=self.response({"status": status}),
            ) as get:
                self.assertEqual(self.client.health_check(log=self.log), ready)
            get.assert_called_once_with("http://mock/health", timeout=0.25)
        self.assertEqual(self.client.stats["requests"], 0)

    def test_health_timeout_returns_false_without_prediction_count(self):
        with patch("vla_sim.vla_client.requests.get", side_effect=requests.Timeout("timeout")):
            self.assertFalse(self.client.health_check(log=self.log))
        self.assertEqual(self.client.stats, {"requests": 0, "errors": 0, "error_rate": 0.0})

    def test_timeout_must_be_positive_and_finite(self):
        for timeout in (0, -1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "timeout"):
                VLAClient("http://mock", timeout=timeout)


class RealServoClientBoundaryTest(unittest.TestCase):
    """Exercise actual servo handlers with no ROS imports, node or publisher."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/real_vla_servo.py"
        tree = ast.parse(path.read_text())
        servo = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RealVLAServo")
        handlers = {"_query_vla", "_set_gripper", "_on_gripper_response"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_limit_norm"]
        nodes += [node for node in servo.body if isinstance(node, ast.FunctionDef) and node.name in handlers]
        namespace = {
            "np": np, "time": SimpleNamespace(monotonic=lambda: 1.0),
            "SetIO": SimpleNamespace(Request=type("Request", (), {"FUN_SET_DIGITAL_OUT": 1})),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        cls.handlers = namespace

    def setUp(self):
        self.log = Mock()
        self.node = SimpleNamespace(
            args=SimpleNamespace(
                duration=10, instruction="pick the mug", unnorm_key="test_dataset",
                linear_gain=1.0, angular_gain=1.0, max_linear_speed=0.05,
                max_angular_speed=0.05, smoothing_alpha=0.2,
                gripper_close_threshold=0.5, gripper_io_pin=0, enable_motion=True,
            ),
            latest_rgb=np.zeros((8, 12, 3), dtype=np.uint8), started_at=0.0,
            client=VLAClient("http://mock", timeout=0.25), query_count=0,
            last_cmd_linear=np.array([0.02, 0, 0], dtype=np.float32),
            last_cmd_angular=np.array([0, 0, 0.01], dtype=np.float32),
            last_action=np.array([0.02, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            get_logger=lambda: self.log, _send_zero=Mock(), _set_gripper=Mock(),
            last_gripper_sent=1.0, gripper_available=True, gripper_client=Mock(),
        )
        self.node.client._last_action = self.node.last_action.copy()
        self.node._on_gripper_response = lambda future, value: self.invoke("_on_gripper_response", future, value)

    def invoke(self, name, *args):
        return self.handlers[name](self.node, *args)

    def test_failed_prediction_clears_residual_smoothed_motion_without_releasing(self):
        for result in (requests.Timeout("timeout"), VLAClientTest.response({"action": [float("nan")] * 7})):
            with self.subTest(result=type(result).__name__):
                self.setUp()
                with patch("vla_sim.vla_client.requests.post", side_effect=[result]):
                    self.invoke("_query_vla")
                np.testing.assert_array_equal(self.node.last_cmd_linear, [0, 0, 0])
                np.testing.assert_array_equal(self.node.last_cmd_angular, [0, 0, 0])
                self.assertEqual(self.node.last_action[6], 1.0)
                self.node._set_gripper.assert_not_called()
                self.node._send_zero.assert_called_once()

    def test_valid_response_recovers_after_a_failed_query(self):
        with patch("vla_sim.vla_client.requests.post", side_effect=[
            requests.Timeout("timeout"),
            VLAClientTest.response({"action": [0.01, 0, 0, 0, 0, 0, 1]}),
        ]):
            self.invoke("_query_vla")
            self.invoke("_query_vla")
        np.testing.assert_allclose(self.node.last_cmd_linear, [0.002, 0, 0])
        self.node._set_gripper.assert_called_once_with(True)
        self.assertEqual(self.node.client.stats["errors"], 1)

    def test_success_preserves_existing_gain_limit_and_smoothing(self):
        with patch("vla_sim.vla_client.requests.post", return_value=VLAClientTest.response(
            {"action": [0.1, 0, 0, 0, 0, 0.1, 1]},
        )):
            self.invoke("_query_vla")
        np.testing.assert_allclose(self.node.last_cmd_linear, [0.026, 0, 0])
        np.testing.assert_allclose(self.node.last_cmd_angular, [0, 0, 0.018])
        self.node._send_zero.assert_not_called()

    def test_gripper_acknowledgement_is_required_for_success_log(self):
        self.invoke("_on_gripper_response", Mock(result=lambda: SimpleNamespace(success=True)), 1.0)
        self.log.info.assert_called_once()
        self.log.warning.assert_not_called()
        self.assertEqual(self.node.last_gripper_sent, 1.0)

    def test_rejected_gripper_request_can_be_sent_again(self):
        self.node.last_gripper_sent = None
        future = Mock(result=lambda: SimpleNamespace(success=False))
        self.node.gripper_client.call_async.return_value = future
        self.invoke("_set_gripper", True)
        self.invoke("_on_gripper_response", future, 1.0)
        self.invoke("_set_gripper", True)
        self.assertEqual(self.node.gripper_client.call_async.call_count, 2)
        self.log.info.assert_not_called()
        self.log.warning.assert_called_once()

    def test_failed_gripper_future_clears_only_matching_command_cache(self):
        future = Mock()
        future.result.side_effect = RuntimeError("service failed")
        self.invoke("_on_gripper_response", future, 1.0)
        self.assertIsNone(self.node.last_gripper_sent)
        self.log.info.assert_not_called()

    def test_old_gripper_failure_does_not_clear_a_newer_opposite_command(self):
        self.node.last_gripper_sent = 0.0
        self.invoke("_on_gripper_response", Mock(result=lambda: SimpleNamespace(success=False)), 1.0)
        self.assertEqual(self.node.last_gripper_sent, 0.0)

    def test_dry_run_never_dispatches_a_gripper_request(self):
        self.node.args.enable_motion = False
        self.invoke("_set_gripper", False)
        self.node.gripper_client.call_async.assert_not_called()


if __name__ == "__main__":
    unittest.main()
