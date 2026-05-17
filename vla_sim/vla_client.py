"""HTTP client for the OpenVLA inference server."""

import base64
import io
import time

import numpy as np
import requests
from PIL import Image


class VLAClient:
    """Small client used by the Isaac Lab simulation process."""

    def __init__(self, server_url: str, timeout: float = 2.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._last_action = np.zeros(7, dtype=np.float32)
        self._request_count = 0
        self._error_count = 0

    def health_check(self, log=print) -> bool:
        """Return True when the server is reachable and the model is ready."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            log(f"[VLA] Server status: {data['status']}")
            log(f"[VLA] Model: {data.get('model_path', '?')}")
            log(f"[VLA] Quantization: {data.get('quantization', '?')}")
            return data["status"] == "ok"
        except Exception as exc:
            log(f"[VLA] Health check failed: {exc}")
            return False

    def predict(
        self,
        rgb: np.ndarray,
        instruction: str,
        unnorm_key: str = "ur3e_vla_dataset",
        log=print,
    ) -> np.ndarray:
        """Send an RGB image and instruction, then return a 7D action."""
        self._request_count += 1
        t0 = time.monotonic()

        try:
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            response = requests.post(
                f"{self.server_url}/predict",
                json={
                    "image_b64": img_b64,
                    "instruction": instruction,
                    "unnorm_key": unnorm_key,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            action = np.array(data["action"], dtype=np.float32)
            self._last_action = action
            latency = data.get("latency_ms", (time.monotonic() - t0) * 1000)

            if self._request_count % 10 == 0:
                rounded = action.round(3).tolist()
                log(
                    f"[VLA] #{self._request_count}: action={rounded} "
                    f"latency={latency:.0f}ms err={self._error_count}"
                )

            return action

        except Exception as exc:
            self._error_count += 1
            if self._error_count <= 5 or self._error_count % 20 == 0:
                log(f"[VLA] predict failed (#{self._error_count}): {exc}")
            return self._last_action.copy()

    @property
    def stats(self):
        return {
            "requests": self._request_count,
            "errors": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
        }
