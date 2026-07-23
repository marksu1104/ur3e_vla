"""Pure unit tests for real-to-sim joint-state mapping and stale HOLD."""

from __future__ import annotations

import unittest

from vla_sim.joint_state_mirror import LatestJointState, UR3E_ARM_JOINT_NAMES


class LatestJointStateTest(unittest.TestCase):
    def test_maps_joint_names_even_when_message_is_shuffled(self):
        source = LatestJointState(stale_timeout=1.0)
        names = tuple(reversed(UR3E_ARM_JOINT_NAMES))
        values = tuple(float(index) for index in range(len(names)))
        self.assertTrue(source.update(names, values, received_at=10.0))
        snapshot = source.snapshot(now=10.1)
        self.assertTrue(snapshot.is_live)
        self.assertEqual(snapshot.positions, tuple(reversed(values)))

    def test_missing_joint_is_rejected_without_overwriting_latest_sample(self):
        source = LatestJointState(stale_timeout=1.0)
        self.assertTrue(
            source.update(UR3E_ARM_JOINT_NAMES, range(6), received_at=10.0)
        )
        self.assertFalse(source.update(UR3E_ARM_JOINT_NAMES[:-1], range(5), received_at=10.1))
        snapshot = source.snapshot(now=10.2)
        self.assertTrue(snapshot.is_live)
        self.assertEqual(snapshot.positions, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))

    def test_stale_sample_holds_last_position(self):
        source = LatestJointState(stale_timeout=0.5)
        self.assertTrue(
            source.update(UR3E_ARM_JOINT_NAMES, range(6), received_at=10.0)
        )
        self.assertTrue(source.snapshot(now=10.49).is_live)
        snapshot = source.snapshot(now=10.51)
        self.assertEqual(snapshot.state, "hold")
        self.assertEqual(snapshot.detail, "stale_joint_state")
        self.assertEqual(snapshot.positions, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))

    def test_latest_valid_message_replaces_older_message(self):
        source = LatestJointState(stale_timeout=1.0)
        self.assertTrue(source.update(UR3E_ARM_JOINT_NAMES, range(6), received_at=10.0))
        self.assertTrue(source.update(UR3E_ARM_JOINT_NAMES, range(10, 16), received_at=10.1))
        self.assertEqual(
            source.snapshot(now=10.2).positions,
            (10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        )


if __name__ == "__main__":
    unittest.main()
