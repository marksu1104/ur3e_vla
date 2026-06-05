#!/usr/bin/env python3
"""Send small OpenVLA velocity commands to MoveIt Servo from a real camera."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

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
    parser.add_argument("--linear-gain", type=float, default=1.0)
    parser.add_argument("--angular-gain", type=float, default=0.0)
    parser.add_argument("--max-linear-speed", type=float, default=0.01)
    parser.add_argument("--max-angular-speed", type=float, default=0.03)
    parser.add_argument("--allow-high-speed", action="store_true",
                        help="Allow max-linear-speed above the conservative safety cap.")
    parser.add_argument("--smoothing-alpha", type=float, default=1.0,
                        help="Low-pass command update. 1.0 disables smoothing; 0.2 is conservative.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Print image/controller/joint-state health while running.")
    parser.add_argument("--enable-motion", action="store_true")
    return parser.parse_args()


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

    def _query_vla(self):
        if self.latest_rgb is None:
            self.get_logger().info("Waiting for image...")
            return
        if time.monotonic() - self.started_at > self.args.duration:
            return

        action = self.client.predict(
            self.latest_rgb,
            self.args.instruction,
            unnorm_key=self.args.unnorm_key,
            log=lambda msg: self.get_logger().info(msg),
        )
        self.last_action = action

        linear = _limit_norm(action[:3] * self.args.linear_gain, self.args.max_linear_speed)
        angular = _limit_norm(action[3:6] * self.args.angular_gain, self.args.max_angular_speed)
        alpha = float(np.clip(self.args.smoothing_alpha, 0.0, 1.0))
        linear = (1.0 - alpha) * self.last_cmd_linear + alpha * linear
        angular = (1.0 - alpha) * self.last_cmd_angular + alpha * angular
        self.last_cmd_linear = linear.astype(np.float32)
        self.last_cmd_angular = angular.astype(np.float32)
        self.query_count += 1

        self.get_logger().info(
            f"vla#{self.query_count} action={np.round(action, 5).tolist()} "
            f"cmd_linear={np.round(linear, 5).tolist()} "
            f"cmd_angular={np.round(angular, 5).tolist()} "
            f"gripper_ignored={action[6]:.3f}"
        )


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
        elapsed = time.monotonic() - self.started_at
        if elapsed > self.args.duration:
            self._send_zero()
            self.get_logger().info("Reached duration; sent zero command and shutting down.")
            rclpy.shutdown()
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
    # if args.max_linear_speed > 0.05 and not args.allow_high_speed:
    #     print(
    #         f"[SAFETY] Requested --max-linear-speed {args.max_linear_speed:.3f} m/s is high. "
    #         "Clamping to 0.05 m/s. Pass --allow-high-speed only after confirming safety."
    #     )
    #     args.max_linear_speed = 0.05
    rclpy.init()
    node = RealVLAServo(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._send_zero()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
