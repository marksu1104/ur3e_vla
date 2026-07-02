"""Video helpers for trajectory inspection."""

from pathlib import Path

import numpy as np


class VideoRecorder:
    """Small OpenCV MP4 recorder for Isaac camera RGB frames."""

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.writer = None
        self.frame_count = 0

    def write_rgb(self, rgb: np.ndarray):
        if rgb is None:
            return

        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return

        frame = frame[:, :, :3].astype(np.uint8)
        if self.writer is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("opencv-python is required for video recording") from exc

            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {self.path}")

        import cv2

        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
