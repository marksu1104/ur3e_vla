"""Read-only ROS 2 JointState subscription for the virtual mirror."""

from __future__ import annotations

from vla_sim.joint_state_mirror import LatestJointState


class ROSJointStateSubscriber:
    """Keep only the newest ROS JointState and never create publishers."""

    def __init__(self, topic: str, latest: LatestJointState):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        self._latest = latest
        self._node = Node("ur3e_real_mirror")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self._node.create_subscription(JointState, topic, self._on_joint_state, qos)

    def _on_joint_state(self, message) -> None:
        self._latest.update(message.name, message.position)

    def spin_once(self) -> None:
        """Process at most one newest queued ROS sample per Isaac frame."""
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        self._node.destroy_node()
