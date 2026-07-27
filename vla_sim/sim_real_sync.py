"""Real/Isaac synchronization state, control decisions, and ROS joint input.

This module owns the behavior unique to ``scripts/sync_sim_real.py``.  The
only code allowed to publish commands to the physical robot remains isolated
in ``vla_sim.real_arm_io``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from time import monotonic
from typing import Iterable

import numpy as np

from vla_sim.actions import clamp_action, compute_action_from_ee_poses
from vla_sim.config import (
    EE_ORIENT_DOWN,
    HOME_POS,
    PLACE_POSITIONS,
    ROBOT_BASE_ROT,
    TARGETS,
)
from vla_sim.planning import build_pick_place_trajectory
from vla_sim.runtime import ExternalStateBackend


SUPPORTED_TASK_PAIR = (1, 2)
UR3E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class SyncOptions:
    """Parameters for the currently validated sim-to-real behavior."""

    enable_motion: bool
    gripper_timeout: float
    waypoint_tolerance: float
    waypoint_timeout: float
    linear_gain: float
    max_linear_speed: float
    yaw_gain: float
    max_yaw_speed: float
    yaw_tolerance: float
    grasp_z_offset: float
    place_z_offset: float
    compensate_frame_offset: bool


@dataclass(frozen=True)
class JointStateSnapshot:
    """One mapped state with a clear live/HOLD decision."""

    positions: tuple[float, ...] | None
    received_at: float | None
    age_seconds: float | None
    state: str
    detail: str

    @property
    def is_live(self) -> bool:
        return self.state == "live"


class LatestJointState:
    """Map named JointState samples and retain only the latest valid sample."""

    def __init__(
        self,
        joint_names: tuple[str, ...] = UR3E_ARM_JOINT_NAMES,
        *,
        stale_timeout: float = 0.5,
    ):
        if stale_timeout <= 0.0:
            raise ValueError("stale_timeout must be positive")
        self.joint_names = tuple(joint_names)
        self.stale_timeout = float(stale_timeout)
        self._positions: tuple[float, ...] | None = None
        self._received_at: float | None = None
        self._last_error = "awaiting_joint_state"

    def update(
        self,
        names: Iterable[str],
        positions: Iterable[float],
        *,
        received_at: float | None = None,
    ) -> bool:
        names = tuple(str(name) for name in names)
        positions = tuple(float(position) for position in positions)
        if len(names) != len(positions):
            self._last_error = "name_position_length_mismatch"
            return False
        if len(set(names)) != len(names):
            self._last_error = "duplicate_joint_name"
            return False
        by_name = dict(zip(names, positions, strict=True))
        missing = [name for name in self.joint_names if name not in by_name]
        if missing:
            self._last_error = f"missing_joint:{','.join(missing)}"
            return False
        mapped = tuple(by_name[name] for name in self.joint_names)
        if not all(math.isfinite(position) for position in mapped):
            self._last_error = "nonfinite_joint_position"
            return False
        self._positions = mapped
        self._received_at = monotonic() if received_at is None else float(received_at)
        self._last_error = ""
        return True

    def snapshot(self, *, now: float | None = None) -> JointStateSnapshot:
        if self._positions is None or self._received_at is None:
            return JointStateSnapshot(
                positions=None,
                received_at=None,
                age_seconds=None,
                state="hold",
                detail=self._last_error,
            )
        current = monotonic() if now is None else float(now)
        age = max(0.0, current - self._received_at)
        if age > self.stale_timeout:
            return JointStateSnapshot(
                positions=self._positions,
                received_at=self._received_at,
                age_seconds=age,
                state="hold",
                detail="stale_joint_state",
            )
        return JointStateSnapshot(
            positions=self._positions,
            received_at=self._received_at,
            age_seconds=age,
            state="live",
            detail="",
        )


class ROSJointStateSubscriber:
    """Keep only the newest ROS JointState; this class never publishes."""

    def __init__(self, topic: str, latest: LatestJointState):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        self._latest = latest
        self._node = Node("ur3e_joint_sync")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self._node.create_subscription(JointState, topic, self._on_joint_state, qos)

    @property
    def node(self):
        return self._node

    def _on_joint_state(self, message) -> None:
        self._latest.update(message.name, message.position)

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        self._node.destroy_node()


class JointSyncBackend(ExternalStateBackend):
    """Write only the latest valid external arm sample into Isaac."""

    def __init__(self, source: LatestJointState):
        self.source = source
        self.last_snapshot = source.snapshot()
        self.paused = False

    @property
    def status(self) -> dict:
        snapshot = self.last_snapshot
        return {
            "state": "paused" if self.paused else snapshot.state,
            "detail": "paused" if self.paused else snapshot.detail,
            "age_seconds": snapshot.age_seconds,
            "received_at": snapshot.received_at,
        }

    def apply(self, controller, dt: float) -> None:
        del dt
        self.last_snapshot = self.source.snapshot()
        if (
            not self.paused
            and self.last_snapshot.is_live
            and self.last_snapshot.positions is not None
        ):
            controller.write_measured_arm_state(self.last_snapshot.positions)
        controller.apply_gripper_target()


@dataclass(frozen=True)
class GripperStatus:
    state: str
    want_closed: bool | None
    elapsed: float | None

    @property
    def failed(self) -> bool:
        return self.state == "timeout"


class GripperFeedbackTracker:
    """Track one gripper request until the expected digital input confirms it."""

    def __init__(self, timeout_seconds: float):
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self._want_closed: bool | None = None
        self._requested_at: float | None = None
        self._settled: str | None = None

    def request(self, closed: bool, now: float) -> None:
        self._want_closed = bool(closed)
        self._requested_at = float(now)
        self._settled = None

    def update(self, digital_inputs, now: float) -> GripperStatus:
        if self._want_closed is None or self._requested_at is None:
            return GripperStatus("idle", None, None)
        elapsed = max(0.0, float(now) - self._requested_at)
        if self._settled is None:
            expected_pin = 2 if self._want_closed else 0
            if bool(digital_inputs.get(expected_pin, False)):
                self._settled = "confirmed"
            elif elapsed >= self.timeout_seconds:
                self._settled = "timeout"
        return GripperStatus(
            self._settled or "waiting", self._want_closed, elapsed
        )

    def clear(self) -> None:
        self._want_closed = None
        self._requested_at = None
        self._settled = None


@dataclass(frozen=True)
class FollowerState:
    target_pos: tuple[float, float, float]
    target_quat: tuple[float, float, float, float]
    gripper: float
    waypoint_index: int
    total_waypoints: int
    distance: float
    at_waypoint: bool
    gripper_change: bool
    done: bool


class ProgressTrajectoryFollower:
    """Advance only after the measured arm reaches the current waypoint."""

    def __init__(self, trajectory, position_tolerance: float):
        if not trajectory:
            raise ValueError("trajectory must contain at least one waypoint")
        if position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be positive")
        self.waypoints = [
            (
                tuple(float(v) for v in pos),
                tuple(float(v) for v in quat),
                float(grip),
            )
            for _duration, pos, quat, grip in trajectory
        ]
        self.position_tolerance = float(position_tolerance)
        self._index = 0

    def advance(self) -> None:
        if self._index < len(self.waypoints):
            self._index += 1

    def state(self, current_pos) -> FollowerState:
        if self._index >= len(self.waypoints):
            pos, quat, grip = self.waypoints[-1]
            return FollowerState(
                pos,
                quat,
                grip,
                len(self.waypoints),
                len(self.waypoints),
                0.0,
                True,
                False,
                True,
            )
        pos, quat, grip = self.waypoints[self._index]
        distance = math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(current_pos, pos))
        )
        previous_grip = (
            self.waypoints[self._index - 1][2] if self._index > 0 else grip
        )
        return FollowerState(
            pos,
            quat,
            grip,
            self._index,
            len(self.waypoints),
            distance,
            distance <= self.position_tolerance,
            grip != previous_grip,
            False,
        )


DRIVE = "drive"
ACTUATE_GRIPPER = "actuate_gripper"
ADVANCE = "advance"
COMPLETE = "complete"
ABORT = "abort"


@dataclass(frozen=True)
class TrialDecision:
    kind: str
    reason: str = ""


def decide(
    *,
    joint_state_live: bool,
    follow: FollowerState,
    yaw_error_rad: float,
    yaw_tolerance: float,
    elapsed_at_waypoint: float,
    waypoint_timeout: float,
    gripper_status: GripperStatus,
) -> TrialDecision:
    if not joint_state_live:
        return TrialDecision(ABORT, "joint_state_not_live")
    if follow.done:
        return TrialDecision(COMPLETE, "trajectory_complete")
    if follow.gripper_change and follow.at_waypoint:
        if gripper_status.failed:
            return TrialDecision(ABORT, "gripper_not_confirmed")
        if abs(yaw_error_rad) <= yaw_tolerance:
            return TrialDecision(ACTUATE_GRIPPER)
    elif follow.at_waypoint:
        return TrialDecision(ADVANCE)
    if elapsed_at_waypoint > waypoint_timeout:
        return TrialDecision(ABORT, "waypoint_timeout")
    return TrialDecision(DRIVE)


def _quat_to_yaw(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_error(current_quat, target_quat) -> float:
    target = _quat_to_yaw(target_quat)
    current = _quat_to_yaw(current_quat)
    return (target - current + math.pi) % (2.0 * math.pi) - math.pi


def _rotate_xy(vector_xy, angle: float) -> tuple[float, float]:
    x, y = vector_xy
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


ISAAC_BASE_YAW_RAD = _quat_to_yaw(ROBOT_BASE_ROT)


class SimToRealTrial:
    """Execute the currently validated bridge task against the real arm."""

    def __init__(
        self,
        runtime,
        controller,
        bridge,
        commander,
        backend,
        seed: int,
        options: SyncOptions,
        log,
    ):
        self.runtime = runtime
        self.controller = controller
        self.bridge = bridge
        self.commander = commander
        self.backend = backend
        self.seed = seed
        self.options = options
        self.log = log
        self.state = "waiting"
        self.gripper = GripperFeedbackTracker(options.gripper_timeout)
        self.follower = None
        self.trial = None
        self.waypoint_started_at = 0.0

    def handle_command(self, command, now: float) -> None:
        if command["type"] == "task" and self.state == "waiting":
            pair = (command["task_index"], command["position_index"])
            if pair != SUPPORTED_TASK_PAIR:
                raise RuntimeError(f"unsupported sync task: obj={pair[0]} dest={pair[1]}")
            self._start_task(command, now)
        elif command["type"] == "control" and command["action"] == "reset":
            self._reset_and_home(command, now)

    def _start_task(self, command, now: float) -> None:
        name = command["object"]
        target = self.runtime.scene[name]
        trajectory = build_pick_place_trajectory(
            TARGETS[name],
            target.data.root_pos_w[0].cpu().numpy(),
            target.data.root_state_w[0, 3:7].cpu().numpy(),
            PLACE_POSITIONS[command["position_index"]],
            grasp_z_offset=self.options.grasp_z_offset,
            place_z_offset=self.options.place_z_offset,
        )
        self.log(
            f"[sim-to-real] trajectory for {name} -> dest "
            f"{command['position_index']}: grasp z {trajectory[3][1][2]:.4f} "
            f"(+{self.options.grasp_z_offset:.3f}), place z "
            f"{trajectory[7][1][2]:.4f} (+{self.options.place_z_offset:.3f})"
        )
        self._begin(trajectory, command, now)
        self.bridge.set_state(
            "running",
            trial_id=command["trial_id"],
            object=name,
            task_index=command["task_index"],
            position_index=command["position_index"],
            seed=self.seed,
        )

    def _reset_and_home(self, command, now: float) -> None:
        """Preserve the currently validated reset behavior, including homing."""
        self.seed = self.seed if command["seed"] is None else command["seed"]
        self.commander.send_zero_twist()
        self.runtime.reset_targets()
        self.commander.set_gripper(False)
        self.controller.write_commanded_gripper_state(0.0)
        self._begin([(0.0, HOME_POS, EE_ORIENT_DOWN, 0.0)], None, now)
        self.log(
            "[sim-to-real] reset: gripper opened, homing to "
            f"{HOME_POS} (straight line from the current pose)"
        )
        self.bridge.set_state("resetting", seed=self.seed)

    def _begin(self, trajectory, trial, now: float) -> None:
        self.follower = ProgressTrajectoryFollower(
            trajectory, self.options.waypoint_tolerance
        )
        self.gripper.clear()
        self.trial = trial
        self.state = "running"
        self.waypoint_started_at = now

    def step(self, now: float, step_index: int) -> None:
        if self.state != "running" or self.follower is None:
            return
        pos, quat = self._ee_pose()
        follow = self.follower.state(pos)
        yaw_error = _yaw_error(quat, follow.target_quat)
        yaw_speed = float(
            np.clip(
                yaw_error * self.options.yaw_gain,
                -self.options.max_yaw_speed,
                self.options.max_yaw_speed,
            )
        )
        angular = np.array([0.0, 0.0, yaw_speed], dtype=np.float32)
        decision = decide(
            joint_state_live=self.backend.last_snapshot.is_live,
            follow=follow,
            yaw_error_rad=yaw_error,
            yaw_tolerance=self.options.yaw_tolerance,
            elapsed_at_waypoint=now - self.waypoint_started_at,
            waypoint_timeout=self.options.waypoint_timeout,
            gripper_status=self.gripper.update(self.commander.digital_inputs, now),
        )
        if decision.kind == ABORT:
            self._finish(follow, False, decision.reason)
        elif decision.kind == COMPLETE:
            self._finish(follow, True, decision.reason)
        elif decision.kind == ACTUATE_GRIPPER:
            self._actuate_gripper(follow, now)
        elif decision.kind == ADVANCE:
            self._advance(now)
        else:
            self._drive(follow, pos, quat, yaw_error, angular, now, step_index)

    def _ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pose = self.controller.robot.data.body_state_w[
            0, self.controller.ee_body_idx, :7
        ].cpu().numpy()
        return pose[:3], pose[3:7]

    def _linear_command(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if self.options.compensate_frame_offset:
            action[0], action[1] = _rotate_xy(
                (action[0], action[1]), -ISAAC_BASE_YAW_RAD
            )
        linear = action[:3] * self.options.linear_gain
        norm = float(np.linalg.norm(linear))
        if self.options.max_linear_speed <= 0.0:
            return np.zeros(3, dtype=np.float32)
        if norm > self.options.max_linear_speed:
            linear *= self.options.max_linear_speed / (norm + 1e-9)
        return linear

    def _drive(
        self, follow, pos, quat, yaw_error, angular, now: float, step_index: int
    ) -> None:
        action = clamp_action(
            compute_action_from_ee_poses(
                pos,
                quat,
                follow.target_pos,
                follow.target_quat,
                gripper_target=follow.gripper,
            )
        )
        linear = self._linear_command(action)
        if self.options.enable_motion:
            self.commander.publish_twist(linear, angular)
            if step_index % 30 == 0:
                self.log(
                    f"[sim-to-real] wp={follow.waypoint_index}/"
                    f"{follow.total_waypoints} dist={follow.distance:.4f} "
                    f"yaw_err={yaw_error:+.4f} "
                    f"linear={np.round(linear, 5).tolist()} wz={angular[2]:+.4f}"
                )
            return
        self.log(
            f"[sim-to-real] DRY RUN wp={follow.waypoint_index}/"
            f"{follow.total_waypoints} dist={follow.distance:.4f} "
            f"yaw_err={yaw_error:+.4f} "
            f"would send linear={np.round(linear, 5).tolist()} "
            f"wz={angular[2]:+.4f} "
            f"target={tuple(round(v, 4) for v in follow.target_pos)}"
        )
        self._advance(now)

    def _actuate_gripper(self, follow, now: float) -> None:
        want_closed = follow.gripper >= 0.5
        if not self.options.enable_motion:
            self.log(
                f"[sim-to-real] DRY RUN: would set gripper "
                f"{'closed' if want_closed else 'open'} at waypoint "
                f"{follow.waypoint_index}"
            )
            self.controller.write_commanded_gripper_state(
                follow.gripper, self.runtime.sim_dt
            )
            self._advance(now)
            return
        self.commander.send_zero_twist(repeat=1)
        if self.gripper.update(self.commander.digital_inputs, now).state == "idle":
            self.commander.set_gripper(want_closed)
            self.gripper.request(want_closed, now)
        status = self.gripper.update(self.commander.digital_inputs, now)
        self.controller.write_commanded_gripper_state(
            follow.gripper, self.runtime.sim_dt
        )
        if status.state == "confirmed":
            self.gripper.clear()
            self._advance(now)

    def _advance(self, now: float) -> None:
        self.follower.advance()
        self.waypoint_started_at = now

    def _finish(self, follow, success: bool, reason: str) -> None:
        self.commander.send_zero_twist()
        if success:
            self.log(
                "[sim-to-real] "
                + ("homing complete" if self.trial is None else "trajectory complete")
            )
        else:
            snapshot = self.backend.last_snapshot
            self.log(
                f"[sim-to-real] ABORT ({reason}) at waypoint "
                f"{follow.waypoint_index}/{follow.total_waypoints} "
                f"dist={follow.distance:.4f} joint_sync={snapshot.state} "
                f"age={snapshot.age_seconds}"
            )
        self.state = "waiting"
        if self.trial is None:
            self.bridge.set_state("waiting", seed=self.seed)
        else:
            self.bridge.finish_trial(
                trial_id=self.trial["trial_id"],
                result={"success": success, "reason": reason},
                progress={
                    "waypoint": follow.waypoint_index,
                    "total": follow.total_waypoints,
                    "distance": round(follow.distance, 5),
                },
            )
        self.follower = None
        self.trial = None


def run_sim_to_real(
    *,
    app,
    runtime,
    controller,
    bridge,
    commander,
    subscriber,
    backend,
    seed: int,
    stream_every: int,
    options: SyncOptions,
    log,
) -> None:
    """Run the persistent bridge/ROS loop for the supported physical task."""
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor()
    executor.add_node(subscriber.node)
    executor.add_node(commander.node)
    trial = SimToRealTrial(
        runtime, controller, bridge, commander, backend, seed, options, log
    )
    step = 0
    last_heartbeat = 0.0
    bridge.set_state("waiting", seed=seed)
    log(
        "sim-to-real active. "
        f"motion={'ENABLED' if options.enable_motion else 'DRY RUN (publishes nothing)'} "
        f"supported_task=obj{SUPPORTED_TASK_PAIR[0]}/dest{SUPPORTED_TASK_PAIR[1]} "
        f"frame_compensation={options.compensate_frame_offset} "
        f"max_linear={options.max_linear_speed} m/s"
    )
    try:
        while app.is_running():
            for _ in range(8):
                executor.spin_once(timeout_sec=0.0)
            now = time.monotonic()
            if now - last_heartbeat >= 2.0:
                snapshot = backend.last_snapshot
                pos, _ = trial._ee_pose()
                log(
                    f"[sim-to-real] state={trial.state} "
                    f"joint_sync={snapshot.state} age={snapshot.age_seconds} "
                    f"detail={snapshot.detail!r} "
                    f"ee={np.round(pos, 4).tolist()} di={commander.digital_inputs}"
                )
                last_heartbeat = now
            command = bridge.poll_command()
            if command is not None:
                bridge.command_applied(command)
                trial.handle_command(command, now)
            trial.step(now, step)
            runtime.step()
            if step % stream_every == 0:
                rgb = runtime.latest_yolo_rgb()
                if rgb is not None:
                    bridge.publish_frame(rgb[0].cpu().numpy().astype(np.uint8))
            step += 1
    finally:
        executor.remove_node(subscriber.node)
        executor.remove_node(commander.node)
        executor.shutdown()
