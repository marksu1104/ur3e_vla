"""Exercise actual remote command handling with controlled HTTP interleavings."""

from __future__ import annotations

import ast
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

import numpy as np

from vla_sim.remote_bridge import BridgeServer


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, item):
        return _Array(self.value[item])

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class RemoteCommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[1] / "scripts/run_remote_pick_place.py"
        tree = ast.parse(cls.source.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        command_block = next(
            node for node in ast.walk(main)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "command is not None"
        )
        module = ast.Module(body=[command_block], type_ignores=[])
        cls.command_code = compile(ast.fix_missing_locations(module), str(cls.source), "exec")
        cls.main_code = compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(cls.source), "exec")

    def setUp(self):
        self.bridge = BridgeServer({0: "red_mug", 1: "blue_cup"}, 2)
        self.bridge.set_state("waiting")
        target = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=_Array([[0.1, 0.2, 0.3]]),
            root_link_pose_w=_Array([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]]),
        ))
        self.runtime = SimpleNamespace(reset_targets=lambda: None)
        self.controller = SimpleNamespace(reset_home=lambda: None)
        self.namespace = {
            "bridge": self.bridge, "state": "waiting", "runtime": self.runtime,
            "controller": self.controller, "scene": {"red_mug": target, "blue_cup": target},
            "as_torch": lambda value: value, "pose_wxyz_from_sim": lambda value: value,
            "TARGETS": {"red_mug": {}, "blue_cup": {}}, "PLACE_POSITIONS": [(0.0, 0.0), (0.2, 0.2)],
            "GRIPPER_OPEN": 0.0, "build_pick_place_trajectory": lambda *args, **kwargs: object(),
            "PoseTrajectoryPlayer": lambda *args, **kwargs: SimpleNamespace(
                sample=lambda value: ((0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0), 0.0, False),
                total_time=10.0,
            ),
            "device": "cpu", "seed": 42, "HOME_POS": (0.0, 0.0, 0.5),
            "EE_ORIENT_DOWN": (1.0, 0.0, 0.0, 0.0), "run_t": 1.0,
            "player": SimpleNamespace(total_time=10.0), "paused_from_state": "running",
        }

    def post(self, path, payload):
        async def request():
            responses = []
            consumed = False

            async def receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
                return {"type": "http.disconnect"}

            async def send(message):
                responses.append(message)

            await self.bridge._app(
                {"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 1),
                 "server": ("mock", 80)}, receive, send,
            )
            status = next(message["status"] for message in responses if message["type"] == "http.response.start")
            body = b"".join(message.get("body", b"") for message in responses if message["type"] == "http.response.body")
            return status, json.loads(body)

        return asyncio.run(request())

    def apply(self):
        self.namespace["command"] = self.bridge.poll_command()
        self.assertIsNotNone(self.namespace["command"])
        exec(self.command_code, self.namespace)

    def assert_rejected_during_transition(self, path, payload):
        status, response = self.post(path, payload)
        self.assertEqual(status, 409)
        self.assertIn("pending", response["detail"])

    def test_task_reservation_prevents_accepted_then_dropped_second_task(self):
        self.assertEqual(self.post("/pickplace", {"obj": 0, "dest": 1})[0], 202)

        def build(*args, **kwargs):
            self.assertEqual(self.bridge.status_snapshot()["state"], "waiting")
            self.assert_rejected_during_transition("/pickplace", {"obj": 1, "dest": 0})
            return object()

        self.namespace["build_pick_place_trajectory"] = build
        self.apply()
        self.assertEqual(self.namespace["state"], "running")
        self.assertEqual(self.bridge.status_snapshot()["trial_id"], 1)
        self.assertIsNone(self.bridge.poll_command())
        self.assertIsNone(self.bridge._pending_command)
        self.bridge.set_state("waiting")
        status, response = self.post("/pickplace", {"obj": 1, "dest": 0})
        self.assertEqual(status, 202)
        self.assertEqual(response["trial_id"], 2)

    def test_reset_reservation_blocks_task_while_fixed_objects_are_restored(self):
        self.assertEqual(self.post("/control", {"action": "reset", "seed": 17})[0], 200)
        self.runtime.reset_targets = lambda: self.assert_rejected_during_transition(
            "/pickplace", {"obj": 1, "dest": 0}
        )
        self.apply()
        self.assertEqual(self.namespace["state"], "resetting")
        self.assertEqual(self.bridge.status_snapshot()["state"], "resetting")
        self.assertEqual(self.bridge.status_snapshot()["seed"], 17)
        self.assertIsNone(self.bridge._pending_command)
        self.assertIsNone(self.bridge.poll_command())
        self.bridge.set_state("waiting")
        self.assertEqual(self.post("/pickplace", {"obj": 1, "dest": 0})[1]["trial_id"], 1)

    def transition_with_interleaved_control(self, action, initial_state, next_state, competing_action):
        self.bridge.set_state(initial_state)
        self.namespace["state"] = initial_state
        self.assertEqual(self.post("/control", {"action": action})[0], 200)
        original_set_state = self.bridge.set_state

        def publish(state, **fields):
            self.assert_rejected_during_transition("/control", {"action": competing_action})
            original_set_state(state, **fields)

        self.bridge.set_state = publish
        self.apply()
        self.assertEqual(self.bridge.status_snapshot()["state"], next_state)
        self.assertIsNone(self.bridge.poll_command())
        self.assertIsNone(self.bridge._pending_command)

    def test_pause_keeps_reservation_until_paused_is_published(self):
        self.transition_with_interleaved_control("pause", "running", "paused", "reset")

    def test_resume_keeps_reservation_until_running_is_published(self):
        self.namespace["paused_from_state"] = "settling"
        self.transition_with_interleaved_control("resume", "paused", "running", "reset")
        self.assertEqual(self.namespace["state"], "settling")

    def test_failed_trajectory_keeps_reservation_before_error_shutdown(self):
        self.assertEqual(self.post("/pickplace", {"obj": 0, "dest": 1})[0], 202)

        def fail(*args, **kwargs):
            raise RuntimeError("trajectory construction failed")

        self.namespace["build_pick_place_trajectory"] = fail
        with self.assertRaisesRegex(RuntimeError, "trajectory construction failed"):
            self.apply()
        self.assertIsNotNone(self.bridge._pending_command)
        self.assertIsNone(self.bridge.poll_command())
        self.assert_rejected_during_transition("/pickplace", {"obj": 1, "dest": 0})
        self.assert_rejected_during_transition("/control", {"action": "reset"})

    def test_actual_main_stops_bridge_after_failed_trajectory(self):
        self.bridge.start = Mock()
        self.bridge.stop = Mock()
        self.controller.finger_limits = (0.0, 1.0)
        self.controller.finger_position = 0.0
        self.controller.physical_gripper_command = 0.0
        self.controller.set_pose_target = Mock()
        self.runtime.scene = self.namespace["scene"]
        self.runtime.scene["camera_yolo"] = object()
        self.runtime.robot_controller = self.controller
        self.runtime.sim_dt = 1 / 60
        self.runtime.device = "cpu"
        self.runtime.start = lambda: self.runtime
        self.runtime.latest_yolo_rgb = lambda: None
        steps = 0
        iterations = 0

        def step():
            nonlocal steps
            steps += 1
            if steps == 120:
                self.assertEqual(self.bridge.status_snapshot()["state"], "waiting")
                self.assertEqual(self.post("/pickplace", {"obj": 0, "dest": 1})[0], 202)

        def is_running():
            nonlocal iterations
            iterations += 1
            return iterations <= 123

        def fail(*args, **kwargs):
            raise RuntimeError("trajectory construction failed")

        self.runtime.step = step
        namespace = {
            **self.namespace, "time": time, "app": SimpleNamespace(is_running=is_running),
            "_extra_args": SimpleNamespace(
                bridge_port=None, stream_every_n_steps=None, stream_quality=None,
                max_trial_seconds=None, stream_width=None, stream_height=None, show_markers=False, seed=42,
            ),
            "args_cli": SimpleNamespace(device="cpu"), "BRIDGE_PORT": 8100,
            "BRIDGE_HOST": "127.0.0.1", "STREAM_EVERY_N_STEPS": 2,
            "STREAM_JPEG_QUALITY": 85, "MAX_TRIAL_SECONDS": 60,
            "TASK_INDEX_MAP": {0: "red_mug", 1: "blue_cup"}, "TARGET_KEYS": ["red_mug"],
            "BridgeServer": lambda *args, **kwargs: self.bridge,
            "SimulationRuntime": lambda options: self.runtime, "RuntimeOptions": SimpleNamespace,
            "_run_with_progress": lambda label, operation: operation(),
            "object_visibility_report": lambda camera: {}, "log": lambda message: None,
            "build_pick_place_trajectory": fail,
        }
        exec(self.main_code, namespace)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "trajectory construction failed"):
            namespace["main"]()
        self.bridge.stop.assert_called_once_with()
        self.assertIsNotNone(self.bridge._pending_command)
        self.assert_rejected_during_transition("/pickplace", {"obj": 1, "dest": 0})


if __name__ == "__main__":
    unittest.main()
