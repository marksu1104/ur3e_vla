"""Small local stand-in for the future Unity C# bridge client.

The real application should call the same HTTP endpoints and keep the same
WebSocket open. This script is only for manual integration checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from websockets.sync.client import connect


class ProtocolFailure(RuntimeError):
    """Raised when the server violates the Unity bridge contract."""


def _ws_url(server: str) -> str:
    # The HTTP server and WebSocket stream use the same host and port.
    server = server.rstrip("/")
    if server.startswith("https://"):
        return "wss://" + server.removeprefix("https://") + "/unity"
    if server.startswith("http://"):
        return "ws://" + server.removeprefix("http://") + "/unity"
    raise ProtocolFailure("server must start with http:// or https://")


@dataclass
class BridgeClient:
    """HTTP commands matching the future C# client."""

    server: str
    _http: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self.server = self.server.rstrip("/")

    def close(self) -> None:
        self._http.close()

    def status(self) -> dict:
        # Keep one HTTP session so repeated commands can reuse the connection.
        response = self._http.get(f"{self.server}/status", timeout=5)
        response.raise_for_status()
        return response.json()

    def pickplace(self, obj: int, dest: int) -> requests.Response:
        return self._http.post(
            f"{self.server}/pickplace",
            json={"obj": obj, "dest": dest},
            timeout=5,
        )

    def control(self, action: str, seed: int | None = None) -> requests.Response:
        payload: dict[str, Any] = {"action": action}
        if seed is not None:
            payload["seed"] = seed
        return self._http.post(f"{self.server}/control", json=payload, timeout=5)


class UnityStream:
    """Persistent raw-JPEG stream matching the future C# video source."""

    def __init__(self, server: str):
        self._ws = connect(_ws_url(server), open_timeout=10, close_timeout=2)
        # The server always sends its current state before frames and events.
        try:
            kind, payload = self.receive(timeout=10)
            if kind != "event" or payload.get("type") != "state":
                raise ProtocolFailure(
                    "first /unity message was not a state event"
                )
        except Exception:
            self.close()
            raise
        self.initial_state = payload

    def __enter__(self) -> "UnityStream":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._ws.close()

    def receive(
        self, timeout: float = 1.0
    ) -> tuple[str, bytes | dict] | tuple[None, None]:
        try:
            message = self._ws.recv(timeout=timeout)
        except TimeoutError:
            return None, None
        # Text messages are JSON events; binary messages are complete JPEGs.
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError as exc:
                raise ProtocolFailure(f"invalid JSON event: {exc}") from exc
            if not isinstance(payload, dict):
                raise ProtocolFailure("JSON event must be an object")
            return "event", payload
        if not isinstance(message, bytes):
            raise ProtocolFailure(
                f"unexpected WebSocket message: {type(message).__name__}"
            )
        if not (message.startswith(b"\xff\xd8") and message.endswith(b"\xff\xd9")):
            raise ProtocolFailure("binary /unity message was not a complete JPEG")
        return "frame", message


def watch(server: str) -> None:
    """Display the persistent C#-compatible stream; q/Esc closes only this viewer."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("--watch requires opencv-python and numpy") from exc
    if "GUI:                           NONE" in cv2.getBuildInformation():
        raise RuntimeError(
            "OpenCV has no GUI support; install opencv-python, not "
            "opencv-python-headless"
        )

    state = "connecting"
    trial_id = 0
    result = "-"
    frame_count = 0
    last_frame_at = time.monotonic()
    fps = 0.0

    try:
        # One WebSocket stays open while tasks and control commands use HTTP.
        with UnityStream(server) as stream:
            state = str(stream.initial_state.get("state"))
            trial_id = int(stream.initial_state.get("trial_id") or 0)
            print(f"EVENT state={state} trial={trial_id}", flush=True)
            while True:
                kind, payload = stream.receive(timeout=0.25)
                if kind == "event":
                    event_type = payload.get("type")
                    if event_type == "state":
                        state = str(payload.get("state"))
                        trial_id = int(payload.get("trial_id") or 0)
                        print(f"EVENT state={state} trial={trial_id}", flush=True)
                    elif event_type == "complete":
                        result = str(bool(payload.get("success")))
                        print(
                            f"EVENT complete trial={payload.get('trial_id')} "
                            f"success={payload.get('success')}",
                            flush=True,
                        )
                    continue
                if kind != "frame":
                    continue

                image = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise ProtocolFailure("could not decode JPEG")
                now = time.monotonic()
                if frame_count:
                    fps = 1.0 / max(now - last_frame_at, 1e-6)
                frame_count += 1
                last_frame_at = now
                overlay = (
                    f"{state} trial={trial_id} result={result} "
                    f"frame={frame_count} {fps:.1f}fps"
                )
                cv2.putText(
                    image,
                    overlay,
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("UR3e Remote Bridge", image)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    return
    finally:
        cv2.destroyAllWindows()


def _print_response(response: requests.Response) -> None:
    print(f"HTTP {response.status_code} {response.text}")
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default=os.environ.get("UR3E_BRIDGE_SERVER", "http://127.0.0.1:8100"),
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--watch", action="store_true")
    operation.add_argument("--obj", type=int, choices=range(3))
    operation.add_argument("--control", choices=("pause", "resume", "reset"))
    parser.add_argument("--dest", type=int, choices=range(3))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.dest is not None and args.obj is None:
        parser.error("--dest requires --obj")
    if args.obj is not None and args.dest is None:
        parser.error("--obj requires --dest")
    if args.seed is not None and args.control != "reset":
        parser.error("--seed is only valid with --control reset")

    if args.watch:
        watch(args.server)
        return

    client = BridgeClient(args.server)
    try:
        if args.obj is not None:
            _print_response(client.pickplace(args.obj, args.dest))
        elif args.control:
            _print_response(client.control(args.control, args.seed))
        else:
            print(json.dumps(client.status(), indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (ProtocolFailure, requests.RequestException, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
