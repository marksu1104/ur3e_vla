"""Unit tests for safe, latest-only UR3e JointState mapping."""

from __future__ import annotations

import unittest

from vla_sim.config import ARM_JOINT_NAMES
from vla_sim.ur3e_ros import LatestJointState


class LatestJointStateTest(unittest.TestCase):
    def test_maps_a_shuffled_sample_by_joint_name(self) -> None:
        sample_names = tuple(reversed(ARM_JOINT_NAMES))
        sample_positions = tuple(float(index) for index in range(len(sample_names)))
        latest = LatestJointState(stale_timeout=0.5)

        self.assertTrue(latest.update(sample_names, sample_positions, received_at=10.0))
        snapshot = latest.snapshot(now=10.1)

        expected = tuple(reversed(sample_positions))
        self.assertEqual(snapshot.positions, expected)
        self.assertEqual(snapshot.state, "live")

    def test_rejects_missing_joint_without_a_valid_sample(self) -> None:
        latest = LatestJointState(stale_timeout=0.5)
        self.assertFalse(latest.update(ARM_JOINT_NAMES[:-1], (0.0,) * 5, received_at=1.0))

        snapshot = latest.snapshot(now=1.0)
        self.assertEqual(snapshot.state, "hold")
        self.assertIn(ARM_JOINT_NAMES[-1], snapshot.detail)

    def test_rejects_duplicate_names(self) -> None:
        latest = LatestJointState(stale_timeout=0.5)
        names = (*ARM_JOINT_NAMES[:-1], ARM_JOINT_NAMES[0])
        self.assertFalse(latest.update(names, (0.0,) * 6, received_at=1.0))
        self.assertEqual(latest.snapshot(now=1.0).detail, "duplicate_joint_name")

    def test_holds_the_last_valid_pose_after_timeout(self) -> None:
        latest = LatestJointState(stale_timeout=0.5)
        positions = tuple(float(index) for index in range(6))
        self.assertTrue(latest.update(ARM_JOINT_NAMES, positions, received_at=10.0))

        snapshot = latest.snapshot(now=10.51)
        self.assertEqual(snapshot.state, "hold")
        self.assertEqual(snapshot.detail, "stale_joint_state")
        self.assertEqual(snapshot.positions, positions)


if __name__ == "__main__":
    unittest.main()
