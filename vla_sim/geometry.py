"""Geometry helpers using IsaacLab wxyz quaternions."""

from __future__ import annotations

import numpy as np


def quat_mul(q1, q2) -> tuple[float, float, float, float]:
    """Hamilton product for quaternions in IsaacLab wxyz order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def yaw_from_quat_wxyz(q) -> float:
    """Return world-Z yaw from a wxyz quaternion."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def angle_diff(a: float, b: float) -> float:
    """Return signed shortest angular difference a - b in radians."""
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)
