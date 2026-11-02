"""Verify inference failure handling without a server, Isaac, ROS, or GPU."""

from __future__ import annotations

import base64
import ast
import io
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
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


class _PredictionThread:
    defer = False
    instances = []

    def __init__(self, *, target, daemon):
        self.target, self.daemon = target, daemon
        self.finished = False
        self.instances.append(self)

    def start(self):
        if not self.defer:
            self.finish()

    def finish(self):
        if not self.finished:
            self.target()
            self.finished = True


class _Twist:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id=None)
        self.twist = SimpleNamespace(
            linear=SimpleNamespace(x=0, y=0, z=0),
            angular=SimpleNamespace(x=0, y=0, z=0),
        )


class RealServoClientBoundaryTest(unittest.TestCase):
    """Exercise actual servo handlers with no ROS imports, node or publisher."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/real_vla_servo.py"
        tree = ast.parse(path.read_text())
        servo = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RealVLAServo")
        handlers = {
            "_query_vla", "_set_gripper", "_on_gripper_response", "_poll_prediction",
            "_hold_motion", "_finish_duration", "_publish_servo", "_send_zero",
        }
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_limit_norm", "_is_fresh", "parse_args", "main"}]
        nodes += [node for node in servo.body if isinstance(node, ast.FunctionDef) and node.name in handlers]
        namespace = {
            "np": np, "queue": queue, "argparse": __import__("argparse"),
            "threading": SimpleNamespace(Thread=_PredictionThread),
            "TwistStamped": _Twist,
            "SetIO": SimpleNamespace(Request=type("Request", (), {"FUN_SET_DIGITAL_OUT": 1})),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        cls.handlers = namespace

    def setUp(self):
        _PredictionThread.instances = []
        _PredictionThread.defer = False
        self.clock = SimpleNamespace(now=1.0)
        self.ros = SimpleNamespace(ok=Mock(return_value=True), shutdown=Mock(), init=Mock(), spin=Mock())
        self.handlers["time"] = SimpleNamespace(monotonic=lambda: self.clock.now)
        self.handlers["rclpy"] = self.ros
        self.log = Mock()
        self.node = SimpleNamespace(
            args=SimpleNamespace(
                duration=10, instruction="pick the mug", unnorm_key="test_dataset",
                linear_gain=1.0, angular_gain=1.0, max_linear_speed=0.05,
                max_angular_speed=0.05, smoothing_alpha=0.2,
                gripper_close_threshold=0.5, gripper_io_pin=0, enable_motion=True,
                frame_id="base_link", image_timeout=0.5, action_timeout=1.5,
            ),
            latest_rgb=np.zeros((8, 12, 3), dtype=np.uint8), started_at=0.0,
            last_image_time=0.9, last_action_time=0.9,
            client=VLAClient("http://mock", timeout=0.25), query_count=0,
            last_cmd_linear=np.array([0.02, 0, 0], dtype=np.float32),
            last_cmd_angular=np.array([0, 0, 0.01], dtype=np.float32),
            last_action=np.array([0.02, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            get_logger=lambda: self.log, _send_zero=Mock(), _set_gripper=Mock(),
            last_gripper_sent=1.0, gripper_available=True, gripper_client=Mock(),
            _prediction_result=queue.Queue(maxsize=1), _prediction_thread=None,
            _stopping=False, _hold_reason=None, publish_count=0, servo_pub=Mock(),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp")),
        )
        self.node.client._last_action = self.node.last_action.copy()
        self.node._on_gripper_response = lambda future, value: self.invoke("_on_gripper_response", future, value)
        for name in ("_poll_prediction", "_hold_motion", "_finish_duration"):
            setattr(self.node, name, lambda *args, _name=name: self.invoke(_name, *args))

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

    def test_missing_stale_future_or_nonfinite_image_prevents_query(self):
        for received_at in (None, -5, 2, float("nan"), float("inf")):
            with self.subTest(received_at=received_at):
                self.setUp()
                self.node.last_image_time = received_at
                with patch("vla_sim.vla_client.requests.post") as post:
                    self.invoke("_query_vla")
                post.assert_not_called()
                self.assertIsNone(self.node._prediction_thread)
                self.node._set_gripper.assert_not_called()
                np.testing.assert_array_equal(self.node.last_cmd_linear, [0, 0, 0])

    def test_pending_request_is_bounded_and_old_action_stops_with_fresh_camera(self):
        _PredictionThread.defer = True
        self.invoke("_query_vla")
        self.clock.now = 3.0
        self.node.last_image_time = 3.0
        self.invoke("_publish_servo")
        self.invoke("_query_vla")
        self.assertEqual(len(_PredictionThread.instances), 1)
        np.testing.assert_array_equal(self.node.last_cmd_linear, [0, 0, 0])
        self.node.servo_pub.publish.assert_not_called()
        self.node._set_gripper.assert_not_called()

    def test_slow_http_worker_does_not_block_main_stop_checks(self):
        started, release = threading.Event(), threading.Event()

        def predict(*args, **kwargs):
            started.set()
            if not release.wait(2):
                raise RuntimeError("test worker was not released")
            return np.array([0.01, 0, 0, 0, 0, 0, 1], dtype=np.float32)

        with patch.dict(self.handlers, {"threading": threading}), patch.object(self.node.client, "predict", side_effect=predict):
            self.invoke("_query_vla")
            worker = self.node._prediction_thread
            try:
                self.assertTrue(started.wait(1))
                self.assertTrue(worker.daemon)
                self.clock.now = 3.0
                self.node.last_image_time = 3.0
                self.invoke("_publish_servo")
                self.invoke("_query_vla")
                self.assertIs(self.node._prediction_thread, worker)
                self.assertTrue(worker.is_alive())
                np.testing.assert_array_equal(self.node.last_cmd_linear, [0, 0, 0])
                self.node._set_gripper.assert_not_called()
            finally:
                release.set()
                worker.join(timeout=1)
            self.assertFalse(worker.is_alive())

    def test_prediction_after_duration_never_applies_action_or_gripper(self):
        _PredictionThread.defer = True
        self.node.args.duration = 3.0
        before = self.node.last_action.copy()
        self.invoke("_query_vla")
        self.clock.now = 10.0
        self.node.last_image_time = 10.0
        with patch("vla_sim.vla_client.requests.post", return_value=VLAClientTest.response({"action": [0, 0, 0.02, 0, 0, 0, 0]})):
            _PredictionThread.instances[0].finish()
        self.invoke("_publish_servo")
        np.testing.assert_array_equal(self.node.last_action, before)
        self.node._set_gripper.assert_not_called()
        self.node.servo_pub.publish.assert_not_called()
        self.assertTrue(self.node._stopping)
        self.ros.shutdown.assert_called_once()

    def test_fresh_camera_cannot_refresh_an_old_predictions_action_age(self):
        _PredictionThread.defer = True
        before = self.node.last_action.copy()
        self.invoke("_query_vla")
        self.clock.now = 3.0
        self.node.last_image_time = 3.0
        with patch("vla_sim.vla_client.requests.post", return_value=VLAClientTest.response({"action": [0.01, 0, 0, 0, 0, 0, 0]})):
            _PredictionThread.instances[0].finish()
        self.invoke("_poll_prediction")
        np.testing.assert_array_equal(self.node.last_action, before)
        self.assertIsNone(self.node.last_action_time)
        self.node._set_gripper.assert_not_called()

    def test_camera_expiring_during_inference_discards_result(self):
        _PredictionThread.defer = True
        self.invoke("_query_vla")
        self.clock.now = 1.6
        with patch("vla_sim.vla_client.requests.post", return_value=VLAClientTest.response({"action": [0.01, 0, 0, 0, 0, 0, 1]})):
            _PredictionThread.instances[0].finish()
        self.invoke("_poll_prediction")
        self.node._set_gripper.assert_not_called()
        np.testing.assert_array_equal(self.node.last_cmd_linear, [0, 0, 0])

    def test_fresh_prediction_recovers_after_a_camera_hold(self):
        self.node.last_image_time = -5
        self.invoke("_query_vla")
        self.node.last_image_time = self.clock.now
        with patch("vla_sim.vla_client.requests.post", return_value=VLAClientTest.response({"action": [0.01, 0, 0, 0, 0, 0, 1]})):
            self.invoke("_query_vla")
        np.testing.assert_allclose(self.node.last_cmd_linear, [0.002, 0, 0])
        self.assertEqual(self.node.last_action_time, 1.0)
        self.node._set_gripper.assert_called_once_with(True)

    def test_late_result_after_shutdown_is_discarded(self):
        self.node._stopping = True
        self.node._prediction_result.put((1.0, np.zeros(7), None, []))
        before = self.node.last_action.copy()
        self.invoke("_poll_prediction")
        np.testing.assert_array_equal(self.node.last_action, before)
        self.node._set_gripper.assert_not_called()
        self.node.servo_pub.publish.assert_not_called()

    def test_worker_exception_stops_velocity_without_gripper_request(self):
        with patch.object(self.node.client, "predict", side_effect=RuntimeError("worker failed")):
            self.invoke("_query_vla")
        np.testing.assert_array_equal(self.node.last_cmd_linear, [0, 0, 0])
        self.node._set_gripper.assert_not_called()
        self.node._send_zero.assert_called_once()

    def test_publisher_rejects_expired_future_or_nonfinite_action_time(self):
        for received_at in (None, -5, 2, float("nan"), float("inf")):
            with self.subTest(received_at=received_at):
                self.setUp()
                self.node.last_action_time = received_at
                self.invoke("_publish_servo")
                self.node.servo_pub.publish.assert_not_called()
                np.testing.assert_array_equal(self.node.last_cmd_linear, [0, 0, 0])

    def test_valid_publisher_preserves_velocity_and_dry_run_sends_nothing(self):
        self.invoke("_publish_servo")
        message = self.node.servo_pub.publish.call_args.args[0]
        self.assertAlmostEqual(message.twist.linear.x, 0.02)
        self.node.servo_pub.reset_mock()
        self.node.args.enable_motion = False
        self.invoke("_publish_servo")
        self.invoke("_send_zero")
        self.node.servo_pub.publish.assert_not_called()

    def test_main_stops_before_destroy_even_after_an_unexpected_exception(self):
        for error in (None, KeyboardInterrupt(), RuntimeError("spin failed")):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                order = []
                def zero():
                    self.assertTrue(self.node._stopping)
                    order.append("zero")
                self.node._send_zero.side_effect = zero
                self.node.destroy_node = lambda: order.append("destroy")
                self.ros.spin.side_effect = error
                self.ros.shutdown.side_effect = lambda: order.append("shutdown")
                with patch.dict(self.handlers, {"parse_args": lambda: self.node.args, "RealVLAServo": lambda args: self.node}):
                    if isinstance(error, RuntimeError):
                        with self.assertRaisesRegex(RuntimeError, "spin failed"):
                            self.handlers["main"]()
                    else:
                        self.handlers["main"]()
                self.assertEqual(order, ["zero", "destroy", "shutdown"])

    def test_invalid_numeric_options_are_rejected_without_ros(self):
        for flag, values in (
            ("--image-timeout", ("0", "-1", "nan", "inf")),
            ("--action-timeout", ("0", "-1", "nan", "inf")),
            ("--duration", ("0", "nan")),
            ("--linear-gain", ("nan", "inf")),
            ("--smoothing-alpha", ("1.1", "nan")),
        ):
            for value in values:
                with self.subTest(flag=flag, value=value), patch.object(sys, "argv", ["servo", flag, value]), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        self.handlers["parse_args"]()
                    self.assertEqual(caught.exception.code, 2)

    def test_default_rates_and_explicit_signed_gains_remain_supported(self):
        with patch.object(sys, "argv", ["servo", "--linear-gain", "-1", "--angular-gain", "-0.5"]):
            args = self.handlers["parse_args"]()
        self.assertEqual((args.query_rate_hz, args.publish_rate_hz), (1.0, 10.0))
        self.assertEqual((args.linear_gain, args.angular_gain), (-1.0, -0.5))
        self.assertFalse(args.enable_motion)
        self.assertEqual((args.image_timeout, args.action_timeout), (0.5, 1.5))

    def test_constructor_failure_still_closes_ros_context_without_motion(self):
        factory = Mock(side_effect=RuntimeError("startup failed"))
        with patch.dict(self.handlers, {"parse_args": lambda: self.node.args, "RealVLAServo": factory}):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                self.handlers["main"]()
        self.ros.spin.assert_not_called()
        self.node._send_zero.assert_not_called()
        self.ros.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
