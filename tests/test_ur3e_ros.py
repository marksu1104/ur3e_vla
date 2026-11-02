"""CPU-only checks for joint mapping, gripper transport and fresh feedback."""

from __future__ import annotations

import ast
from concurrent.futures import Future
from dataclasses import dataclass
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vla_sim.config import ARM_JOINT_NAMES
from vla_sim.ur3e_ros import LatestJointState, RealArmCommander


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


def _commander():
    """Construct only ordinary state: no ROS import, node or publisher."""
    commander = RealArmCommander.__new__(RealArmCommander)
    commander._motion_enabled = True
    commander.gripper_available = True
    commander._last_gripper_sent = None
    commander._last_gripper_accepted = None
    commander._gripper_request_id = 0
    commander._gripper_io_pin = 0
    commander._digital_inputs = {}
    commander._digital_inputs_received_at = None
    commander._setio_type = SimpleNamespace(
        Request=type("Request", (), {"FUN_SET_DIGITAL_OUT": 1})
    )
    commander._gripper_client = Mock()
    commander._log = Mock()
    commander.send_zero_twist = Mock()
    commander.futures = []

    def dispatch(request):
        del request
        future = Future()
        commander.futures.append(future)
        return future

    commander._gripper_client.call_async.side_effect = dispatch
    return commander


class GripperTransportTest(unittest.TestCase):
    def setUp(self):
        self.commander = _commander()

    def test_io_callback_records_reception_time_and_returns_a_copy(self):
        message = SimpleNamespace(digital_in_states=[SimpleNamespace(pin=2, state=True)])
        with patch("vla_sim.ur3e_ros.monotonic", return_value=12.0):
            self.commander._on_io_states(message)
        self.assertEqual(self.commander.digital_inputs_received_at, 12.0)
        values = self.commander.digital_inputs
        values[2] = False
        self.assertEqual(self.commander.digital_inputs, {2: True})

    def test_accepted_response_is_distinct_from_an_inflight_request(self):
        self.commander.set_gripper(True)
        self.assertFalse(self.commander.gripper_command_accepted(True))
        self.commander.futures[0].set_result(SimpleNamespace(success=True))
        self.assertTrue(self.commander.gripper_command_accepted(True))
        self.assertFalse(self.commander.gripper_command_accepted(False))
        self.commander.set_gripper(True)
        self.assertEqual(len(self.commander.futures), 1)

    def test_rejected_response_clears_the_cache_and_allows_retry(self):
        self.commander.set_gripper(True)
        self.commander.futures[0].set_result(SimpleNamespace(success=False))
        self.assertIsNone(self.commander._last_gripper_sent)
        self.assertFalse(self.commander.gripper_command_accepted(True))
        self.assertFalse(any("accepted" in str(call) for call in self.commander._log.call_args_list))
        self.commander.set_gripper(True)
        self.assertEqual(len(self.commander.futures), 2)

    def test_future_exception_clears_the_cache_and_allows_retry(self):
        self.commander.set_gripper(False)
        self.commander.futures[0].set_exception(RuntimeError("IO failed"))
        self.assertIsNone(self.commander._last_gripper_sent)
        self.commander.set_gripper(False)
        self.assertEqual(len(self.commander.futures), 2)

    def test_dispatch_exception_does_not_leave_a_cached_command(self):
        self.commander._gripper_client.call_async.side_effect = RuntimeError("client closed")
        self.commander.set_gripper(True)
        self.assertIsNone(self.commander._last_gripper_sent)
        self.assertFalse(self.commander.gripper_command_accepted(True))

    def test_late_rejection_does_not_clear_newer_opposite_request(self):
        self.commander.set_gripper(True)
        self.commander.set_gripper(False)
        self.commander.futures[0].set_result(SimpleNamespace(success=False))
        self.assertEqual(self.commander._last_gripper_sent, 0.0)
        self.commander.futures[1].set_result(SimpleNamespace(success=True))
        self.assertTrue(self.commander.gripper_command_accepted(False))

    def test_late_acceptance_cannot_authenticate_a_newer_request(self):
        self.commander.set_gripper(True)
        self.commander.set_gripper(False)
        self.commander.futures[0].set_result(SimpleNamespace(success=True))
        self.assertFalse(self.commander.gripper_command_accepted(True))
        self.assertFalse(self.commander.gripper_command_accepted(False))

    def test_late_rejection_does_not_clear_newer_same_value_request(self):
        self.commander.set_gripper(True)
        self.commander.set_gripper(False)
        self.commander.set_gripper(True)
        self.commander.futures[0].set_result(SimpleNamespace(success=False))
        self.assertEqual(self.commander._last_gripper_sent, 1.0)
        self.commander.futures[2].set_result(SimpleNamespace(success=True))
        self.assertTrue(self.commander.gripper_command_accepted(True))

    def test_empty_response_is_a_failure(self):
        self.commander.set_gripper(True)
        self.commander.futures[0].set_result(None)
        self.assertIsNone(self.commander._last_gripper_sent)
        self.assertFalse(self.commander.gripper_command_accepted(True))

    def test_dry_run_and_unavailable_io_never_dispatch(self):
        for enabled, available in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, available=available):
                self.commander._motion_enabled = enabled
                self.commander.gripper_available = available
                self.commander.set_gripper(True)
        self.commander._gripper_client.call_async.assert_not_called()


class GripperFeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "vla_sim/sim_real_sync.py"
        tree = ast.parse(path.read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {"GripperStatus", "GripperFeedbackTracker"}]
        trial = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimToRealTrial")
        methods = [n for n in trial.body if isinstance(n, ast.FunctionDef) and n.name in {"_gripper_status", "_actuate_gripper"}]
        namespace = {"__name__": __name__, "dataclass": dataclass, "math": math}
        exec(compile(ast.Module(body=classes + methods, type_ignores=[]), str(path), "exec"), namespace)
        cls.namespace = namespace
        cls.Tracker = namespace["GripperFeedbackTracker"]

    def setUp(self):
        self.tracker = self.Tracker(3.0)
        self.tracker.request(True, 10.0)

    def status(self, received_at, now=10.1, accepted=True, values=None):
        return self.tracker.update(
            {2: True} if values is None else values, now,
            received_at=received_at, command_accepted=accepted,
        )

    def test_pre_request_sample_cannot_confirm(self):
        self.assertEqual(self.status(9.9).state, "waiting")

    def test_missing_timestamp_cannot_confirm(self):
        self.assertEqual(self.status(None).state, "waiting")

    def test_old_post_request_sample_cannot_confirm(self):
        self.assertEqual(self.status(10.1, now=10.7).state, "waiting")

    def test_fresh_post_request_sample_confirms_an_accepted_command(self):
        self.assertEqual(self.status(10.05).state, "confirmed")

    def test_fresh_di_waits_until_transport_acceptance(self):
        self.assertEqual(self.status(10.05, accepted=False).state, "waiting")
        self.assertEqual(self.status(10.05, now=10.2).state, "confirmed")

    def test_late_fresh_feedback_does_not_override_the_deadline(self):
        self.assertEqual(self.status(13.05, now=13.1).state, "timeout")

    def test_future_or_nonfinite_sample_time_cannot_confirm(self):
        for received_at in (10.2, float("nan"), float("inf")):
            with self.subTest(received_at=received_at):
                self.assertEqual(self.status(received_at).state, "waiting")

    def test_open_requires_the_open_input_not_the_grasp_input(self):
        self.tracker.request(False, 10.0)
        self.assertEqual(self.status(10.05).state, "waiting")
        self.assertEqual(self.status(10.05, values={0: True, 2: False}).state, "confirmed")

    def test_cleared_request_does_not_reuse_an_old_confirmation(self):
        self.assertEqual(self.status(10.05).state, "confirmed")
        self.tracker.clear()
        self.assertEqual(self.status(10.05).state, "idle")
        self.tracker.request(True, 11.0)
        self.assertEqual(self.status(10.05, now=11.1).state, "waiting")

    def trial(self):
        commander = _commander()
        trial = SimpleNamespace(
            commander=commander, gripper=self.Tracker(3.0),
            options=SimpleNamespace(enable_motion=True),
            controller=Mock(), runtime=SimpleNamespace(sim_dt=1 / 60),
            _advance=Mock(), log=Mock(),
        )
        trial._gripper_status = MethodType(self.namespace["_gripper_status"], trial)
        trial._actuate_gripper = MethodType(self.namespace["_actuate_gripper"], trial)
        return trial

    def test_trial_advances_only_after_ack_and_a_new_di_sample(self):
        trial = self.trial()
        trial.commander._digital_inputs = {2: True}
        trial.commander._digital_inputs_received_at = 9.9
        follow = SimpleNamespace(gripper=1.0)
        trial._actuate_gripper(follow, 10.0)
        trial.commander.futures[0].set_result(SimpleNamespace(success=True))
        trial._actuate_gripper(follow, 10.1)
        trial._advance.assert_not_called()
        with patch("vla_sim.ur3e_ros.monotonic", return_value=10.15):
            trial.commander._on_io_states(SimpleNamespace(digital_in_states=[SimpleNamespace(pin=2, state=True)]))
        trial._actuate_gripper(follow, 10.2)
        trial._advance.assert_called_once_with(10.2)
        self.assertEqual(len(trial.commander.futures), 1)

    def test_rejected_transport_cannot_be_confirmed_by_fresh_di(self):
        trial = self.trial()
        follow = SimpleNamespace(gripper=1.0)
        trial._actuate_gripper(follow, 10.0)
        trial.commander.futures[0].set_result(SimpleNamespace(success=False))
        with patch("vla_sim.ur3e_ros.monotonic", return_value=10.1):
            trial.commander._on_io_states(SimpleNamespace(digital_in_states=[SimpleNamespace(pin=2, state=True)]))
        trial._actuate_gripper(follow, 10.2)
        trial._advance.assert_not_called()
        self.assertEqual(trial._gripper_status(13.0).state, "timeout")


if __name__ == "__main__":
    unittest.main()
