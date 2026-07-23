"""State-application backends for the canonical simulation runtime.

The current runtime is physics-driven.  The abstract external backend is kept
small on purpose: a later real-robot mirror can provide measured joint states
without teaching scripts about ROS transport details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vla_sim.joint_state_mirror import JointStateSnapshot, LatestJointState

if TYPE_CHECKING:
    from vla_sim.runtime import RobotController


class StateBackend(ABC):
    """Apply one authoritative robot state before a simulation step."""

    @abstractmethod
    def apply(self, controller: "RobotController", dt: float) -> None:
        """Apply state for the next simulation frame."""


class PhysicsDriveBackend(StateBackend):
    """Use the controller's IK and joint targets to drive PhysX."""

    def apply(self, controller: "RobotController", dt: float) -> None:
        del dt
        controller.apply_physics_targets()


class ExternalStateBackend(StateBackend, ABC):
    """Reserved interface for a future measured-state source.

    Implementations must update only the virtual articulation.  This interface
    intentionally contains no ROS publisher or robot-motion API.
    """


class JointStateMirrorBackend(ExternalStateBackend):
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

    def apply(self, controller: "RobotController", dt: float) -> None:
        del dt
        self.last_snapshot = self.source.snapshot()
        if self.paused or not self.last_snapshot.is_live:
            return
        positions = self.last_snapshot.positions
        if positions is None:
            return
        controller.write_measured_arm_state(positions)
