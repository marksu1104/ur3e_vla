"""Exercise bridge HTTP commands through ASGI without sockets or Isaac."""

from __future__ import annotations

import json
import unittest

from vla_sim.remote_bridge import BridgeServer


class BridgeApiTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bridge = BridgeServer({0: "red_mug", 1: "blue_cup"}, num_positions=2)
        self.bridge.set_state("waiting")

    async def post(self, path, payload, bridge=None):
        bridge = self.bridge if bridge is None else bridge
        messages = []
        consumed = False

        async def receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {
                    "type": "http.request",
                    "body": json.dumps(payload).encode(),
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1),
            "server": ("mock", 80),
        }
        await bridge._app(scope, receive, send)
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
        return status, json.loads(body)

    def apply_control(self, expected_action, next_state):
        command = self.bridge.poll_command()
        self.assertEqual(command["type"], "control")
        self.assertEqual(command["action"], expected_action)
        self.bridge.set_state(next_state)
        self.bridge.command_applied(command)
        return command

    async def test_nested_and_other_invalid_actions_return_400_without_commands(self):
        for action in ([], {}, ["pause"], {"action": "reset"}, None, True, 1, "invalid"):
            with self.subTest(action=action):
                status, response = await self.post("/control", {"action": action})
                self.assertEqual(status, 400)
                self.assertIn("action must be", response["detail"])
                self.assertIsNone(self.bridge.poll_command())

    async def test_invalid_seed_is_rejected_before_reserving_reset(self):
        for seed in (True, 1.5, "42", [], {}):
            with self.subTest(seed=seed):
                status, response = await self.post("/control", {"action": "reset", "seed": seed})
                self.assertEqual(status, 400)
                self.assertIn("seed must be", response["detail"])
                self.assertIsNone(self.bridge.poll_command())

    async def test_task_pause_resume_reset_preserve_reservation_semantics(self):
        status, response = await self.post("/pickplace", {"obj": 0, "dest": 1})
        self.assertEqual(status, 202)
        self.assertEqual(response, {"accepted": True, "trial_id": 1, "obj": 0, "dest": 1})
        self.assertEqual((await self.post("/control", {"action": "reset"}))[0], 409)
        task = self.bridge.poll_command()
        self.assertEqual(task["type"], "task")
        self.assertEqual(task["object"], "red_mug")
        self.bridge.set_state("running", trial_id=task["trial_id"])
        self.bridge.command_applied(task)

        self.assertEqual(await self.post("/control", {"action": "pause"}), (200, {"accepted": True}))
        self.assertEqual(self.bridge.status_snapshot()["state"], "running")
        self.assertEqual((await self.post("/control", {"action": "pause"}))[0], 409)
        self.apply_control("pause", "paused")

        self.assertEqual(await self.post("/control", {"action": "resume"}), (200, {"accepted": True}))
        self.apply_control("resume", "running")

        self.assertEqual(await self.post("/control", {"action": "reset", "seed": 42}), (200, {"accepted": True}))
        reset = self.apply_control("reset", "waiting")
        self.assertEqual(reset["seed"], 42)
        self.assertEqual(self.bridge.status_snapshot()["trial_id"], None)

    async def test_pause_resume_require_matching_runtime_state(self):
        self.assertEqual((await self.post("/control", {"action": "pause"}))[0], 409)
        self.assertEqual((await self.post("/control", {"action": "resume"}))[0], 409)
        self.assertIsNone(self.bridge.poll_command())


    async def test_default_controls_remain_pause_resume_reset(self):
        self.assertEqual(
            self.bridge.allowed_controls,
            frozenset({"pause", "resume", "reset"}),
        )
        self.bridge.set_state("running")
        self.assertEqual(
            await self.post("/control", {"action": "pause"}),
            (200, {"accepted": True}),
        )

    async def test_restricted_controls_return_409_without_queueing(self):
        bridge = BridgeServer(
            {0: "red_mug"},
            num_positions=1,
            allowed_controls={"reset"},
        )
        bridge.set_state("running")
        for action in ("pause", "resume"):
            with self.subTest(action=action):
                status, response = await self.post(
                    "/control", {"action": action}, bridge=bridge
                )
                self.assertEqual(status, 409)
                self.assertIn("unavailable", response["detail"])
                self.assertIsNone(bridge.poll_command())
                self.assertIsNone(bridge._pending_command)

        self.assertEqual(
            await self.post("/control", {"action": "reset"}, bridge=bridge),
            (200, {"accepted": True}),
        )
        self.assertEqual(bridge.poll_command()["action"], "reset")

if __name__ == "__main__":
    unittest.main()
