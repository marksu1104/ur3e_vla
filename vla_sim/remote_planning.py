"""Compatibility exports for the canonical scripted planning helpers."""

from vla_sim.demo_planning import (
    ARM_SPEED_MPS,
    GRIPPER_CLOSED,
    GRIPPER_SPEED_RAD_S,
    build_canonical_pick_place_trajectory,
    move_duration,
)


# Keep the remote entry point stable while the planner has one implementation.
build_pick_place_trajectory = build_canonical_pick_place_trajectory


__all__ = [
    "ARM_SPEED_MPS",
    "GRIPPER_CLOSED",
    "GRIPPER_SPEED_RAD_S",
    "build_pick_place_trajectory",
    "move_duration",
]
