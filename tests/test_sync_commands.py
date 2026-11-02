"""CPU-only regressions for synchronization command reservation ordering."""

from __future__ import annotations

import ast
import types
import unittest
from pathlib import Path

from fastapi import HTTPException

from vla_sim.remote_bridge import BridgeServer


ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = ROOT / "scripts" / "sync_sim_real.py"
SYNC_MODULE = ROOT / "vla_sim" / "sim_real_sync.py"
SUPPORTED_PAIR = (1, 2)


def _call_names(node: ast.AST) -> set[str]:
    names = set()
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
        elif isinstance(call.func, ast.Name):
            names.add(call.func.id)
    return names


def _command_block(path: Path, required_calls: set[str]) -> ast.If:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and required_calls <= _call_names(node)
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one command block, found {len(matches)}")
    return matches[0]


def _compile_block(node: ast.If, path: Path):
    module = ast.fix_missing_locations(
        ast.Module(body=[node], type_ignores=[])
    )
    return compile(module, str(path), "exec")


def _compile_trial_handler():
    tree = ast.parse(SYNC_MODULE.read_text(encoding="utf-8"), filename=str(SYNC_MODULE))
    trial_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SimToRealTrial"
    )
    handler = next(
        node
        for node in trial_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "handle_command"
    )
    namespace = {"SUPPORTED_TASK_PAIR": SUPPORTED_PAIR}
    module = ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[]))
    exec(compile(module, str(SYNC_MODULE), "exec"), namespace)
    return namespace["handle_command"]


class SyncCommandOrderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim_node = _command_block(
            SYNC_MODULE, {"handle_command", "command_applied"}
        )
        cls.real_node = _command_block(
            SYNC_SCRIPT, {"_apply_real_to_sim_control", "command_applied"}
        )
        cls.sim_code = _compile_block(cls.sim_node, SYNC_MODULE)
        cls.real_code = _compile_block(cls.real_node, SYNC_SCRIPT)
        cls.trial_handler = staticmethod(_compile_trial_handler())

    def new_bridge(self):
        bridge = BridgeServer(
            {1: "red_mug"},
            num_positions=3,
            allowed_task_pairs={SUPPORTED_PAIR},
        )
        bridge.set_state("waiting")
        return bridge

    def new_trial(self, bridge, start_task):
        class Trial:
            pass

        trial = Trial()
        trial.state = "waiting"
        trial.handle_command = types.MethodType(self.trial_handler, trial)
        trial._start_task = types.MethodType(start_task, trial)
        trial._reset_and_home = types.MethodType(
            lambda instance, command, now: None, trial
        )
        trial.bridge = bridge
        return trial

    def test_sim_to_real_holds_task_until_running_is_published(self):
        bridge = self.new_bridge()
        first = bridge._reserve_task(*SUPPORTED_PAIR)
        command = bridge.poll_command()
        self.assertEqual(command, first)
        competing = []

        def start_task(trial, accepted, now):
            del now
            try:
                bridge._reserve_task(*SUPPORTED_PAIR)
            except HTTPException as exc:
                competing.append((exc.status_code, exc.detail))
            trial.state = "running"
            bridge.set_state("running", trial_id=accepted["trial_id"])

        trial = self.new_trial(bridge, start_task)
        exec(
            self.sim_code,
            {"command": command, "bridge": bridge, "trial": trial, "now": 1.0},
        )

        self.assertEqual(competing, [(409, "a command is already pending")])
        self.assertEqual(bridge.status_snapshot()["trial_id"], 1)
        self.assertIsNone(bridge._pending_command)
        self.assertIsNone(bridge.poll_command())

    def test_real_to_sim_holds_control_until_state_is_published(self):
        bridge = BridgeServer({}, 0, allow_tasks=False)
        bridge.set_state("running")
        bridge._reserve_control("pause", None)
        command = bridge.poll_command()
        competing = []

        def apply_control(command, runtime, backend, target, state, seed):
            del command, runtime, backend, state
            try:
                target._reserve_control("pause", None)
            except HTTPException as exc:
                competing.append((exc.status_code, exc.detail))
            target.set_state("paused")
            return "paused", seed

        namespace = {
            "command": command,
            "bridge": bridge,
            "runtime": None,
            "backend": None,
            "state": "running",
            "seed": 42,
            "_apply_real_to_sim_control": apply_control,
        }
        exec(self.real_code, namespace)

        self.assertEqual(
            competing,
            [(409, "wait for the pending command to be applied")],
        )
        self.assertEqual(namespace["state"], "paused")
        self.assertEqual(bridge.status_snapshot()["state"], "paused")
        self.assertIsNone(bridge._pending_command)

    def test_sim_to_real_handler_failure_retains_reservation(self):
        bridge = self.new_bridge()
        command = bridge._reserve_task(*SUPPORTED_PAIR)
        self.assertEqual(bridge.poll_command(), command)

        def fail(trial, command, now):
            del trial, command, now
            raise RuntimeError("task transition failed")

        trial = self.new_trial(bridge, fail)
        with self.assertRaisesRegex(RuntimeError, "task transition failed"):
            exec(
                self.sim_code,
                {
                    "command": command,
                    "bridge": bridge,
                    "trial": trial,
                    "now": 1.0,
                },
            )

        self.assertEqual(bridge._pending_command, command)
        with self.assertRaises(HTTPException) as task_error:
            bridge._reserve_task(*SUPPORTED_PAIR)
        self.assertEqual(task_error.exception.status_code, 409)
        with self.assertRaises(HTTPException) as control_error:
            bridge._reserve_control("reset", None)
        self.assertEqual(control_error.exception.status_code, 409)

    def test_real_to_sim_handler_failure_retains_reservation(self):
        bridge = BridgeServer({}, 0, allow_tasks=False)
        bridge.set_state("running")
        bridge._reserve_control("reset", None)
        command = bridge.poll_command()

        def fail(*args):
            raise RuntimeError("control transition failed")

        with self.assertRaisesRegex(RuntimeError, "control transition failed"):
            exec(
                self.real_code,
                {
                    "command": command,
                    "bridge": bridge,
                    "runtime": None,
                    "backend": None,
                    "state": "running",
                    "seed": 42,
                    "_apply_real_to_sim_control": fail,
                },
            )

        self.assertEqual(bridge._pending_command, command)
        with self.assertRaises(HTTPException) as error:
            bridge._reserve_control("reset", None)
        self.assertEqual(error.exception.status_code, 409)

    def test_sync_runner_restricts_only_sim_to_real_controls(self):
        tree = ast.parse(
            SYNC_SCRIPT.read_text(encoding="utf-8"), filename=str(SYNC_SCRIPT)
        )
        bridge_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "BridgeServer"
        )
        keyword = next(
            item for item in bridge_call.keywords if item.arg == "allowed_controls"
        )
        self.assertIsInstance(keyword.value, ast.IfExp)
        self.assertEqual(ast.literal_eval(keyword.value.body), {"reset"})
        self.assertIsNone(ast.literal_eval(keyword.value.orelse))


if __name__ == "__main__":
    unittest.main()
