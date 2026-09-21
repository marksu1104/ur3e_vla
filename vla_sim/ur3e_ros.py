"""ROS publishers/clients for commanding the real UR3e, used by sim-to-real.

Deliberately mirrors what ``scripts/real_vla_servo.py`` already does on real
hardware -- ``TwistStamped`` to MoveIt Servo, ``ur_msgs/SetIO`` for the
gripper -- so a scripted trajectory exercises the exact same delivery path a
VLA policy uses, rather than a parallel one that could succeed or fail
differently.

Unlike the read-only joint input in ``vla_sim.sim_real_sync``, this module
*does* publish. Nothing here is constructed unless sim-to-real is selected,
and every command remains gated by ``motion_enabled``.
"""

from __future__ import annotations


class RealArmCommander:
    """Publish servo twists and drive the gripper DO, with DI readback."""

    def __init__(
        self,
        servo_topic: str,
        gripper_io_service: str,
        io_states_topic: str,
        gripper_io_pin: int,
        frame_id: str,
        motion_enabled: bool = False,
        connect_timeout: float = 3.0,
        log=print,
    ):
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from rclpy.node import Node
        from ur_msgs.msg import IOStates
        from ur_msgs.srv import SetIO

        self._rclpy = rclpy
        self._twist_type = TwistStamped
        self._setio_type = SetIO
        self._log = log
        self._frame_id = frame_id
        # Enforced here rather than at each call site: a caller that forgets
        # the check moves a real robot during what was meant to be a dry run.
        self._motion_enabled = bool(motion_enabled)
        self._gripper_io_pin = int(gripper_io_pin)
        self._last_gripper_sent: float | None = None
        self._digital_inputs: dict[int, bool] = {}

        self._node = Node("ur3e_sim_to_real")
        self._twist_pub = self._node.create_publisher(TwistStamped, servo_topic, 10)
        self._node.create_subscription(IOStates, io_states_topic, self._on_io_states, 10)
        self._gripper_client = self._node.create_client(SetIO, gripper_io_service)
        self.gripper_available = self._gripper_client.wait_for_service(
            timeout_sec=connect_timeout
        )
        if self.gripper_available:
            log(f"[RealArm] Gripper IO connected: {gripper_io_service}")
        else:
            log(
                f"[RealArm] WARNING: {gripper_io_service} unavailable after "
                f"{connect_timeout}s; gripper commands will be skipped."
            )

    @property
    def node(self):
        """Expose the node so callers can drive it from a shared executor.

        See the matching note on ``ROSJointStateSubscriber.node``: spinning
        this node and the joint-state node with separate ``rclpy.spin_once``
        calls starves both, because each call churns the global executor's
        node set.
        """
        return self._node

    @property
    def digital_inputs(self) -> dict[int, bool]:
        """Latest digital-input readings, keyed by pin number."""
        return dict(self._digital_inputs)

    def _on_io_states(self, message) -> None:
        readings: dict[int, bool] = {}
        for entry in getattr(message, "digital_in_states", []):
            try:
                readings[int(entry.pin)] = bool(entry.state)
            except (AttributeError, TypeError, ValueError):
                continue
        self._digital_inputs = readings

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def publish_twist(self, linear, angular) -> None:
        """Send one Cartesian velocity command to MoveIt Servo."""
        if not self._motion_enabled:
            return
        message = self._twist_type()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        message.twist.linear.x = float(linear[0])
        message.twist.linear.y = float(linear[1])
        message.twist.linear.z = float(linear[2])
        message.twist.angular.x = float(angular[0])
        message.twist.angular.y = float(angular[1])
        message.twist.angular.z = float(angular[2])
        self._twist_pub.publish(message)

    def send_zero_twist(self, repeat: int = 5) -> None:
        """Command a full stop. Safe to call during shutdown."""
        for _ in range(repeat):
            try:
                self.publish_twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            except Exception as exc:
                self._log(f"[RealArm] failed to publish zero twist: {exc}")
                break

    def set_gripper(self, closed: bool) -> None:
        """Drive the gripper DO pin, edge-triggered so repeats are not resent."""
        if not self._motion_enabled or not self.gripper_available:
            return
        value = 1.0 if closed else 0.0
        if value == self._last_gripper_sent:
            return
        self._last_gripper_sent = value

        request = self._setio_type.Request()
        request.fun = self._setio_type.Request.FUN_SET_DIGITAL_OUT
        request.pin = self._gripper_io_pin
        request.state = value
        future = self._gripper_client.call_async(request)
        future.add_done_callback(lambda f: self._on_gripper_response(f, value))

    def _on_gripper_response(self, future, value: float) -> None:
        try:
            future.result()
            self._log(f"[RealArm] DO_{self._gripper_io_pin} set to {value:.0f}")
        except Exception as exc:
            self._log(f"[RealArm] SetIO failed: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        self._node.destroy_node()
