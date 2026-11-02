#!/usr/bin/env python3
"""Send small OpenVLA velocity commands to MoveIt Servo from a real camera."""

import argparse
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from ur_msgs.srv import SetIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vla_sim.vla_client import VLAClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-topic", default="/image_raw")
    parser.add_argument("--servo-topic", default="/servo_node/delta_twist_cmds")
    parser.add_argument("--controller-command-topic", default="/forward_velocity_controller/commands")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--instruction", default="pick up the red mug")
    parser.add_argument("--vla-server", default="http://localhost:8000")
    parser.add_argument("--unnorm-key", default="ur3e_vla_dataset")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--query-rate-hz", type=float, default=1.0)
    parser.add_argument("--publish-rate-hz", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--image-timeout", type=float, default=0.5,
                        help="Maximum age of the latest received camera frame, in seconds.")
    parser.add_argument("--action-timeout", type=float, default=1.5,
                        help="Maximum action age measured from its input frame reception, in seconds.")
    parser.add_argument("--linear-gain", type=float, default=1.0)
    parser.add_argument("--angular-gain", type=float, default=0.0)
    parser.add_argument("--max-linear-speed", type=float, default=0.01)
    parser.add_argument("--max-angular-speed", type=float, default=0.03)
    parser.add_argument("--allow-high-speed", action="store_true",
                        help="Compatibility flag; speed is limited by --max-linear-speed.")
    parser.add_argument("--smoothing-alpha", type=float, default=1.0,
                        help="Low-pass command update. 1.0 disables smoothing; 0.2 is conservative.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Print image/controller/joint-state health while running.")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--gripper-io-service", default="/io_and_status_controller/set_io",
                        help="ROS 2 SetIO service name exposed by ur_robot_driver.")
    parser.add_argument("--gripper-io-pin", type=int, default=0,
                        help="UR digital output pin number driving the gripper.")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5,
                        help="VLA action[6] value above which the gripper is commanded closed.")
    args = parser.parse_args()
    for name in ("query_rate_hz", "publish_rate_hz", "duration", "timeout", "image_timeout", "action_timeout"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("linear_gain", "angular_gain"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    for name in ("max_linear_speed", "max_angular_speed"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    for name in ("smoothing_alpha", "gripper_close_threshold"):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between zero and one")
    if args.gripper_io_pin < 0:
        parser.error("--gripper-io-pin must be nonnegative")
    return args


def _is_fresh(received_at, now, timeout):
    return (
        received_at is not None
        and np.isfinite(received_at)
        and np.isfinite(now)
        and 0 <= now - received_at <= timeout
    )


def _limit_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).copy()
    if max_norm <= 0.0:
        return np.zeros_like(vec)
    norm = float(np.linalg.norm(vec))
    if norm > max_norm:
        vec *= max_norm / (norm + 1e-9)
    return vec


class RealVLAServo(Node):
    def __init__(self, args):
        super().__init__("real_vla_servo")
        self.args = args
        self.client = VLAClient(args.vla_server, timeout=args.timeout)
        self.latest_rgb = None
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_action_time = None
        self.last_cmd_linear = np.zeros(3, dtype=np.float32)
        self.last_cmd_angular = np.zeros(3, dtype=np.float32)
        self.last_image_time = None
        self.last_controller_cmd = None
        self.last_controller_time = None
        self.initial_joint_pos = None
        self.last_joint_pos = None
        self.last_joint_time = None
        self.started_at = time.monotonic()
        self.query_count = 0
        self.publish_count = 0
        self.last_gripper_sent = None
        self._prediction_result = queue.Queue(maxsize=1)
        self._prediction_thread = None
        self._stopping = False
        self._hold_reason = None

        self.gripper_client = self.create_client(SetIO, args.gripper_io_service)
        self.gripper_available = self.gripper_client.wait_for_service(timeout_sec=3.0)
        if self.gripper_available:
            self.get_logger().info(f"Gripper IO connected: {args.gripper_io_service}")
        else:
            self.get_logger().warning(
                f"Gripper IO service {args.gripper_io_service} not available after 3.0s; "
                "gripper commands will be skipped."
            )

        self.create_subscription(Image, args.image_topic, self._on_image, qos_profile_sensor_data)
        if args.diagnostics:
            self.create_subscription(
                Float64MultiArray,
                args.controller_command_topic,
                self._on_controller_cmd,
                10,
            )
            self.create_subscription(JointState, args.joint_states_topic, self._on_joint_state, 10)
        self.servo_pub = self.create_publisher(TwistStamped, args.servo_topic, 10)
        self.create_timer(1.0 / max(args.query_rate_hz, 0.1), self._query_vla)
        self.create_timer(1.0 / max(args.publish_rate_hz, 1.0), self._publish_servo)
        if args.diagnostics:
            self.create_timer(1.0, self._print_diagnostics)

        mode = "MOTION ENABLED" if args.enable_motion else "DRY RUN ONLY"
        self.get_logger().warning(f"Mode: {mode}")
        self.get_logger().info(f"Image topic: {args.image_topic}")
        self.get_logger().info(f"Servo topic: {args.servo_topic}")
        self.get_logger().info(f"Instruction: {args.instruction!r}")
        self.get_logger().info(
            f"Limits: linear <= {args.max_linear_speed:.4f} m/s, "
            f"angular <= {args.max_angular_speed:.4f} rad/s, "
            f"duration={args.duration:.1f}s"
        )

        if not self.client.health_check(log=lambda msg: self.get_logger().info(msg)):
            self.get_logger().warning("VLA server health check failed; continuing.")
        # Duration measures the control session, not IO/model startup waits.
        self.started_at = time.monotonic()

    def _on_image(self, msg: Image):
        try:
            encoding = msg.encoding.lower()
            if encoding not in {"rgb8", "bgr8"}:
                raise ValueError(f"Unsupported image encoding: {msg.encoding}")
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if encoding == "bgr8":
                rgb = rgb[:, :, ::-1]
            self.latest_rgb = np.ascontiguousarray(rgb)
            self.last_image_time = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc}")

    def _set_gripper(self, is_closed: bool):
        """Send the DO pin state if it differs from the last value sent. Non-blocking."""
        if not self.gripper_available or not self.args.enable_motion:
            return
        value = 1.0 if is_closed else 0.0
        if value == self.last_gripper_sent:
            return
        self.last_gripper_sent = value

        req = SetIO.Request()
        req.fun = SetIO.Request.FUN_SET_DIGITAL_OUT
        req.pin = self.args.gripper_io_pin
        req.state = value

        future = self.gripper_client.call_async(req)
        future.add_done_callback(lambda f: self._on_gripper_response(f, value))

    def _on_gripper_response(self, future, value):
        try:
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError("SetIO rejected the gripper request")
            self.get_logger().info(f"[Gripper] DO_{self.args.gripper_io_pin} set to {value:.0f}")
        except Exception as exc:
            if self.last_gripper_sent == value:
                self.last_gripper_sent = None
            self.get_logger().warning(f"[Gripper] SetIO failed: {type(exc).__name__}: {exc}")

    def _on_controller_cmd(self, msg: Float64MultiArray):
        self.last_controller_cmd = np.asarray(msg.data, dtype=np.float32)
        self.last_controller_time = time.monotonic()

    def _on_joint_state(self, msg: JointState):
        pos = np.asarray(msg.position, dtype=np.float32)
        if pos.size == 0:
            return
        if self.initial_joint_pos is None:
            self.initial_joint_pos = pos.copy()
        self.last_joint_pos = pos
        self.last_joint_time = time.monotonic()

    def _hold_motion(self, reason):
        self.last_cmd_linear = np.zeros(3, dtype=np.float32)
        self.last_cmd_angular = np.zeros(3, dtype=np.float32)
        self.last_action_time = None
        self._send_zero()
        if reason != self._hold_reason:
            self.get_logger().warning(reason)
        self._hold_reason = reason

    def _finish_duration(self):
        if not self._stopping:
            self._stopping = True
            self._hold_motion("Reached duration; held gripper and stopped motion.")
            if rclpy.ok():
                rclpy.shutdown()

    def _poll_prediction(self):
        try:
            observed_at, action, error, messages = self._prediction_result.get_nowait()
        except queue.Empty:
            return
        self._prediction_thread = None
        if self._stopping:
            return
        now = time.monotonic()
        if now - self.started_at >= self.args.duration:
            self._finish_duration()
            return
        if not _is_fresh(self.last_image_time, now, self.args.image_timeout):
            self._hold_motion("Camera frame expired; discarded prediction and held gripper.")
            return
        if not _is_fresh(observed_at, now, self.args.action_timeout):
            self._hold_motion("Prediction expired; held gripper and stopped motion.")
            return
        for message in messages:
            self.get_logger().info(message)
        if error is not None:
            self._hold_motion(f"VLA query failed; cleared motion and held gripper: {error}")
            return
        self.last_action = action.copy()
        self.last_action_time = observed_at
        self._hold_reason = None

        linear = _limit_norm(action[:3] * self.args.linear_gain, self.args.max_linear_speed)
        angular = _limit_norm(action[3:6] * self.args.angular_gain, self.args.max_angular_speed)
        alpha = float(np.clip(self.args.smoothing_alpha, 0.0, 1.0))
        linear = (1.0 - alpha) * self.last_cmd_linear + alpha * linear
        angular = (1.0 - alpha) * self.last_cmd_angular + alpha * angular
        self.last_cmd_linear = linear.astype(np.float32)
        self.last_cmd_angular = angular.astype(np.float32)
        self.query_count += 1

        gripper_closed = bool(action[6] > self.args.gripper_close_threshold)
        self._set_gripper(gripper_closed)

        self.get_logger().info(
            f"vla#{self.query_count} action={np.round(action, 5).tolist()} "
            f"cmd_linear={np.round(linear, 5).tolist()} "
            f"cmd_angular={np.round(angular, 5).tolist()} "
            f"gripper={'closed' if gripper_closed else 'open'} ({action[6]:.3f})"
        )

    def _query_vla(self):
        if self._stopping:
            return
        self._poll_prediction()
        now = time.monotonic()
        if self._stopping:
            return
        if now - self.started_at >= self.args.duration:
            self._finish_duration()
            return
        if self.latest_rgb is None or not _is_fresh(self.last_image_time, now, self.args.image_timeout):
            self._hold_motion("Waiting for a fresh camera frame; held gripper.")
            return
        if self._prediction_thread is not None:
            return

        # Snapshot all worker inputs. Only the main callbacks apply commands;
        # slow HTTP must not block the publisher's freshness/deadline checks.
        rgb = self.latest_rgb.copy()
        observed_at = self.last_image_time
        client = self.client
        instruction, key = self.args.instruction, self.args.unnorm_key
        results = self._prediction_result

        def predict():
            messages = []
            try:
                errors_before = client.stats["errors"]
                action = client.predict(rgb, instruction, unnorm_key=key, log=messages.append)
                error = "failed response" if client.stats["errors"] > errors_before else None
            except Exception as exc:
                action, error = None, f"{type(exc).__name__}: {exc}"
            results.put_nowait((observed_at, action, error, messages))

        self._prediction_thread = threading.Thread(target=predict, daemon=True)
        self._prediction_thread.start()
        self._poll_prediction()


    def _print_diagnostics(self):
        now = time.monotonic()
        image_age = None if self.last_image_time is None else now - self.last_image_time
        controller_age = None if self.last_controller_time is None else now - self.last_controller_time
        joint_age = None if self.last_joint_time is None else now - self.last_joint_time

        if self.last_controller_cmd is None:
            controller_text = "none"
        else:
            finite = bool(np.all(np.isfinite(self.last_controller_cmd)))
            norm = float(np.linalg.norm(self.last_controller_cmd))
            controller_text = (
                f"finite={finite} norm={norm:.5f} "
                f"cmd={np.round(self.last_controller_cmd, 5).tolist()}"
            )

        if self.initial_joint_pos is None or self.last_joint_pos is None:
            joint_text = "none"
        else:
            joint_delta = self.last_joint_pos - self.initial_joint_pos
            joint_text = f"delta_norm={float(np.linalg.norm(joint_delta)):.5f}"

        self.get_logger().info(
            "[DIAG] "
            f"image_age={image_age if image_age is not None else 'none'} "
            f"controller_age={controller_age if controller_age is not None else 'none'} "
            f"controller={controller_text} "
            f"joint_age={joint_age if joint_age is not None else 'none'} "
            f"joint={joint_text} "
            f"published_twist={self.publish_count}"
        )

    def _publish_servo(self):
        if self._stopping:
            return
        self._poll_prediction()
        now = time.monotonic()
        if self._stopping:
            return
        if now - self.started_at >= self.args.duration:
            self._finish_duration()
            return
        if not _is_fresh(self.last_image_time, now, self.args.image_timeout):
            self._hold_motion("Camera frame expired; held gripper and stopped motion.")
            return
        if not _is_fresh(self.last_action_time, now, self.args.action_timeout):
            self._hold_motion("Action expired; held gripper and stopped motion.")
            return
        if not self.args.enable_motion:
            return

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.twist.linear.x = float(self.last_cmd_linear[0])
        msg.twist.linear.y = float(self.last_cmd_linear[1])
        msg.twist.linear.z = float(self.last_cmd_linear[2])
        msg.twist.angular.x = float(self.last_cmd_angular[0])
        msg.twist.angular.y = float(self.last_cmd_angular[1])
        msg.twist.angular.z = float(self.last_cmd_angular[2])
        self.servo_pub.publish(msg)
        self.publish_count += 1

    def _send_zero(self):
        if not self.args.enable_motion or not rclpy.ok():
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        for _ in range(5):
            try:
                self.servo_pub.publish(msg)
            except Exception as exc:
                self.get_logger().warning(f"Failed to publish zero twist: {exc}")
                break


def main():
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = RealVLAServo(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node._stopping = True
                try:
                    node._send_zero()
                finally:
                    node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
