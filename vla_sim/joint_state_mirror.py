"""Latest-only UR3e joint-state mapping used by the read-only mirror."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Iterable

import math


UR3E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


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
        """Accept one sample by name, rejecting malformed samples atomically."""
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
        """Return the latest valid sample or a non-moving HOLD state."""
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
