"""Thread-safe HTTP and WebSocket transport for remote pick-and-place."""

from __future__ import annotations

import asyncio
import copy
import json
import queue
import threading
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

PROTOCOL_VERSION = 1


@dataclass
class _ClientQueues:
    """Outbound queues owned by the uvicorn event loop.

    Frames are intentionally bounded and latest-wins.  Events are separate so
    important state/result messages can never be evicted by video backpressure.
    """

    frames: asyncio.Queue
    events: asyncio.Queue
    last_completed_trial_id: int | None = None


class BridgeServer:
    """Serve commands, status, events, and JPEG frames to LAN clients.

    HTTP handlers never touch Isaac APIs. They reserve commands under a lock;
    the simulation main loop applies them and publishes state, frames, and the
    final result.
    """

    def __init__(
        self,
        task_names: dict[int, str],
        num_positions: int,
        host: str = "127.0.0.1",
        port: int = 8100,
        jpeg_quality: int = 85,
        allow_tasks: bool = True,
        allowed_task_pairs: set[tuple[int, int]] | None = None,
    ):
        self.task_names = {
            int(key): str(value) for key, value in task_names.items()
        }
        self.num_positions = int(num_positions)
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.allow_tasks = bool(allow_tasks)
        self.allowed_task_pairs = (
            None
            if allowed_task_pairs is None
            else {
                (int(task_index), int(position_index))
                for task_index, position_index in allowed_task_pairs
            }
        )

        self._status_lock = threading.Lock()
        self._status: dict[str, Any] = {
            "state": "starting",
            "trial_id": None,
            "object": None,
            "task_index": None,
            "position_index": None,
            "progress": None,
            "result": None,
            "object_poses": None,
            "seed": None,
            "frames_sent": 0,
            "yolo_visibility": None,
            "goal_visibility": None,
            "supported_tasks": (
                None
                if self.allowed_task_pairs is None
                else [
                    {"obj": obj, "dest": dest}
                    for obj, dest in sorted(self.allowed_task_pairs)
                ]
            ),
        }
        self._next_trial_id = 1
        self._pending_command: dict[str, Any] | None = None
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_event: asyncio.Event | None = None
        self._clients: dict[int, _ClientQueues] = {}
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._app = self._build_app()

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            self._loop = asyncio.get_running_loop()
            self._frame_event = asyncio.Event()
            self._ready.set()
            pump_task = asyncio.create_task(
                self._pump_frames(), name="remote-bridge-frame-pump"
            )
            try:
                yield
            finally:
                pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    pass
                self._clients.clear()
                self._frame_event = None
                self._loop = None

        app = FastAPI(
            title="UR3e Remote Bridge",
            version=str(PROTOCOL_VERSION),
            lifespan=lifespan,
        )

        @app.get("/health")
        async def health():
            snapshot = self.status_snapshot()
            return {
                "status": "ok",
                "state": snapshot["state"],
                "protocol": PROTOCOL_VERSION,
            }

        @app.get("/status")
        async def status():
            return self.status_snapshot()

        @app.post("/pickplace", status_code=202)
        async def pickplace(payload: dict[str, Any] = Body(...)):
            if not self.allow_tasks:
                raise HTTPException(
                    status_code=409,
                    detail="pickplace is unavailable in real-to-sim sync mode",
                )
            obj = self._require_index(payload, "obj", self.task_names)
            dest = self._require_index(payload, "dest", range(self.num_positions))
            if (
                self.allowed_task_pairs is not None
                and (obj, dest) not in self.allowed_task_pairs
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"unsupported task; allowed pairs are "
                        f"{sorted(self.allowed_task_pairs)}"
                    ),
                )
            command = self._reserve_task(obj, dest)
            return {
                "accepted": True,
                "trial_id": command["trial_id"],
                "obj": obj,
                "dest": dest,
            }

        @app.post("/control")
        async def control(payload: dict[str, Any] = Body(...)):
            action = payload.get("action")
            if action not in {"pause", "resume", "reset"}:
                raise HTTPException(
                    status_code=400,
                    detail="action must be pause, resume, or reset",
                )
            seed = payload.get("seed")
            if seed is not None and (
                isinstance(seed, bool) or not isinstance(seed, int)
            ):
                raise HTTPException(
                    status_code=400,
                    detail="seed must be an integer or null",
                )
            self._reserve_control(str(action), seed)
            return {"accepted": True}

        @app.websocket("/unity")
        async def unity(websocket: WebSocket):
            await self._serve_websocket(websocket)

        return app

    async def _serve_websocket(self, websocket: WebSocket) -> None:
        """Serve one latest-frame stream without coupling it to HTTP commands."""
        await websocket.accept()
        client_id = id(websocket)
        # Exactly one pending frame: a slow viewer skips stale video instead
        # of accumulating visible latency.
        client = _ClientQueues(asyncio.Queue(maxsize=1), asyncio.Queue())
        self._clients[client_id] = client
        disconnect_task = asyncio.create_task(
            self._wait_for_disconnect(websocket),
            name=f"remote-bridge-disconnect-{client_id}",
        )
        try:
            snapshot = self.status_snapshot()
            complete = self._complete_from_status(snapshot)
            if complete is not None:
                # Suppress a live completion racing this handshake replay.
                client.last_completed_trial_id = int(complete["trial_id"])
            await websocket.send_text(
                json.dumps(self._state_from_status(snapshot))
            )
            if complete is not None:
                await websocket.send_text(json.dumps(complete))

            while not disconnect_task.done():
                # Events have priority. A short frame wait keeps the sender
                # responsive when the client is idle without spinning.
                try:
                    event = client.events.get_nowait()
                except asyncio.QueueEmpty:
                    try:
                        frame = await asyncio.wait_for(
                            client.frames.get(), timeout=0.05
                        )
                    except asyncio.TimeoutError:
                        continue
                    await websocket.send_bytes(frame)
                else:
                    await websocket.send_text(json.dumps(event))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
            self._clients.pop(client_id, None)

    @staticmethod
    async def _wait_for_disconnect(websocket: WebSocket) -> None:
        """Consume client messages only to observe a clean disconnect."""
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

    @staticmethod
    def _state_from_status(status: dict[str, Any]) -> dict[str, Any]:
        event = {
            "type": "state",
            "state": status["state"],
            "trial_id": status["trial_id"] or 0,
        }
        if status["state"] == "waiting":
            for field in (
                "object_poses",
                "seed",
                "yolo_visibility",
                "goal_visibility",
                "supported_tasks",
            ):
                event[field] = copy.deepcopy(status.get(field))
        return event

    @staticmethod
    def _complete_from_status(
        status: dict[str, Any],
    ) -> dict[str, Any] | None:
        result = status.get("result")
        if status.get("state") != "done" or not isinstance(result, dict):
            return None
        return {
            "type": "complete",
            "trial_id": status.get("trial_id") or 0,
            "success": bool(result.get("success")),
            "detail": copy.deepcopy(result.get("detail") or {}),
        }

    @staticmethod
    def _require_index(payload: dict[str, Any], name: str, valid) -> int:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value not in valid:
            raise HTTPException(
                status_code=400, detail=f"{name} is out of range"
            )
        return int(value)

    def _reserve_task(self, task_index: int, position_index: int) -> dict[str, Any]:
        with self._status_lock:
            if self._status["state"] != "waiting":
                raise HTTPException(
                    status_code=409,
                    detail="send reset before submitting another task",
                )
            if self._pending_command is not None:
                raise HTTPException(
                    status_code=409, detail="a command is already pending"
                )
            command = {
                "type": "task",
                "trial_id": self._next_trial_id,
                "task_index": task_index,
                "position_index": position_index,
                "object": self.task_names[task_index],
            }
            self._next_trial_id += 1
            self._pending_command = command
            self._commands.put(command)
            return command.copy()

    def _reserve_control(self, action: str, seed: int | None) -> None:
        with self._status_lock:
            state = self._status["state"]
            if self._pending_command is not None:
                raise HTTPException(
                    status_code=409,
                    detail="wait for the pending command to be applied",
                )
            if action == "pause" and state != "running":
                raise HTTPException(
                    status_code=409,
                    detail="pause is only valid while running",
                )
            if action == "resume" and state != "paused":
                raise HTTPException(
                    status_code=409,
                    detail="resume is only valid while paused",
                )
            if action == "reset" and state == "starting":
                raise HTTPException(
                    status_code=409,
                    detail="reset is unavailable while starting",
                )
            command = {"type": "control", "action": action, "seed": seed}
            self._pending_command = command
            self._commands.put(command)

    def start(self) -> None:
        """Start uvicorn in a daemon thread and wait until its socket is ready."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._startup_error = None
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        def run_server():
            try:
                self._server.run()
            except BaseException as exc:  # surfaced to the caller below
                self._startup_error = exc

        self._thread = threading.Thread(
            target=run_server,
            name="remote-bridge-uvicorn",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._startup_error is not None:
                raise RuntimeError("remote bridge server failed to start") from self._startup_error
            if self._server.started and self._ready.is_set():
                return
            if not self._thread.is_alive():
                raise RuntimeError(
                    "remote bridge server stopped before startup "
                    f"on port {self.port}"
                )
            time.sleep(0.02)
        self.stop()
        raise TimeoutError(f"remote bridge did not start within 10 seconds on port {self.port}")

    def stop(self) -> None:
        """Stop uvicorn and verify that its daemon thread has exited."""
        if self._server is not None:
            self._server.should_exit = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        if thread is not None and thread.is_alive():
            raise TimeoutError("remote bridge server did not stop within 10 seconds")
        self._server = None
        self._thread = None

    def poll_command(self) -> dict[str, Any] | None:
        """Return one accepted command without blocking the simulation loop."""
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def command_applied(self, command: dict[str, Any]) -> None:
        """Clear the command reservation once the main loop has consumed it."""
        with self._status_lock:
            if self._pending_command == command:
                self._pending_command = None

    def publish_frame(self, rgb_uint8_hw3) -> None:
        """Copy an RGB(A) frame into the latest-wins JPEG slot."""
        rgb = np.asarray(rgb_uint8_hw3)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"expected RGB image shaped (H, W, C>=3), got {rgb.shape}")
        rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8).copy()
        with self._frame_lock:
            self._latest_frame = rgb
        loop = self._loop
        frame_event = self._frame_event
        if loop is not None and frame_event is not None:
            try:
                loop.call_soon_threadsafe(frame_event.set)
            except RuntimeError:
                pass

    def set_state(self, state: str, **fields) -> None:
        """Transition public state and push exactly one matching state event."""
        if state not in {
            "starting",
            "resetting",
            "waiting",
            "running",
            "paused",
            "hold",
            "done",
        }:
            raise ValueError(f"unknown remote bridge state: {state}")
        with self._status_lock:
            previous = self._status["state"]
            if state == "waiting":
                self._status.update(
                    trial_id=None,
                    object=None,
                    task_index=None,
                    position_index=None,
                    progress=None,
                    result=None,
                )
            elif state == "resetting":
                self._status.update(
                    progress=None,
                    result=None,
                    object_poses=None,
                    yolo_visibility=None,
                    goal_visibility=None,
                )
            elif state in {"running", "paused"}:
                self._status.update(
                    result=None,
                    object_poses=None,
                    yolo_visibility=None,
                    goal_visibility=None,
                )
            self._status["state"] = state
            self._status.update(copy.deepcopy(fields))
            event = self._state_from_status(self._status)
        if previous != state:
            self.publish_event(event)

    def update_status(self, **fields) -> None:
        """Update live fields such as progress without emitting a state event."""
        with self._status_lock:
            self._status.update(copy.deepcopy(fields))

    def finish_trial(
        self,
        *,
        trial_id: int,
        result: dict[str, Any],
        progress: dict[str, Any],
    ) -> None:
        """Atomically commit a result, then emit complete before state=done.

        Committing status before scheduling either event closes the reconnect
        window where a new client could see running after complete had already
        been fanned out. Unity connections also deduplicate this event by
        trial_id when their handshake replays the committed result.
        """
        with self._status_lock:
            self._status["state"] = "done"
            self._status["trial_id"] = int(trial_id)
            self._status["result"] = copy.deepcopy(result)
            self._status["progress"] = copy.deepcopy(progress)
            state_event = {
                "type": "state",
                "state": "done",
                "trial_id": int(trial_id),
            }
        self.publish_event(
            {
                "type": "complete",
                "trial_id": int(trial_id),
                "success": bool(result.get("success")),
                "detail": copy.deepcopy(result.get("detail") or {}),
            }
        )
        self.publish_event(state_event)

    def publish_event(self, payload: dict) -> None:
        """Schedule an event fan-out on uvicorn's asyncio loop."""
        event = copy.deepcopy(payload)
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._publish_event_on_loop, event)
            except RuntimeError:
                pass

    def _publish_event_on_loop(self, event: dict) -> None:
        for client in tuple(self._clients.values()):
            if event.get("type") == "complete":
                trial_id = int(event.get("trial_id") or 0)
                if trial_id == client.last_completed_trial_id:
                    continue
                client.last_completed_trial_id = trial_id
            client.events.put_nowait(event)

    def status_snapshot(self) -> dict:
        with self._status_lock:
            return copy.deepcopy(self._status)

    async def _pump_frames(self) -> None:
        assert self._frame_event is not None
        while True:
            await self._frame_event.wait()
            self._frame_event.clear()
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                continue
            jpeg = await asyncio.to_thread(_encode_jpeg, frame, self.jpeg_quality)
            with self._status_lock:
                self._status["frames_sent"] += 1
            for client in tuple(self._clients.values()):
                if client.frames.full():
                    try:
                        client.frames.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                client.frames.put_nowait(jpeg)


def _encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """Encode RGB numpy data to JPEG, preferring OpenCV over Pillow."""
    try:
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            raise RuntimeError("cv2.imencode returned false")
        return encoded.tobytes()
    except ImportError:
        from PIL import Image
        import io

        buffer = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
