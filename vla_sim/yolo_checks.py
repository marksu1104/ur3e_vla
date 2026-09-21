"""Lightweight CameraYolo frame-coverage diagnostics."""

from __future__ import annotations

import numpy as np

from vla_sim.config import (
    PLACE_MARKER_COLORS,
    PLACE_POSITIONS,
    TARGET_KEYS,
    TARGETS,
    YOLO_GOAL_ROIS,
    YOLO_MIN_VISIBLE_PIXELS,
    YOLO_TARGET_ROIS,
    YOLO_VISIBILITY_MARGIN_PX,
)


def _rgb_features(camera):
    output = camera.data.output.get("rgb")
    if output is None:
        return None
    rgb = output[0, ..., :3].cpu().numpy().astype(np.float32) / 255.0
    norm = np.linalg.norm(rgb, axis=-1, keepdims=True)
    unit_rgb = rgb / np.maximum(norm, 1e-6)
    saturation = (rgb.max(axis=-1) - rgb.min(axis=-1)) / np.maximum(
        rgb.max(axis=-1), 1e-6
    )
    return rgb, unit_rgb, saturation


def _roi_mask(height: int, width: int, bounds):
    x0f, y0f, x1f, y1f = bounds
    x0, x1 = int(x0f * width), min(width, int(x1f * width))
    y0, y1 = int(y0f * height), min(height, int(y1f * height))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask, [x0, y0, x1 - 1, y1 - 1]


def _coverage(mask, width: int, height: int, roi_xyxy: list[int]) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"visible": False, "complete": False, "pixels": 0}
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    margins = [bbox[0], bbox[1], width - 1 - bbox[2], height - 1 - bbox[3]]
    return {
        "visible": True,
        "complete": (
            len(xs) >= YOLO_MIN_VISIBLE_PIXELS
            and min(margins) >= YOLO_VISIBILITY_MARGIN_PX
        ),
        "pixels": int(len(xs)),
        "bbox_xyxy": bbox,
        "frame_margin_px": int(min(margins)),
        "analysis_roi_xyxy": roi_xyxy,
    }


def object_visibility_report(camera) -> dict:
    """Estimate complete object coverage from distinct 3D-print colors."""
    features = _rgb_features(camera)
    if features is None:
        return {"all_in_frame": False, "error": "rgb_unavailable"}
    rgb, unit_rgb, saturation = features
    height, width, _ = rgb.shape
    colors = np.asarray(
        [TARGETS[name]["color"] for name in TARGET_KEYS],
        dtype=np.float32,
    )
    colors /= np.linalg.norm(colors, axis=1, keepdims=True)
    scores = unit_rgb @ colors.T
    assignments = scores.argmax(axis=-1)
    luminance = rgb.mean(axis=-1)

    objects = {}
    for index, target_key in enumerate(TARGET_KEYS):
        roi, roi_xyxy = _roi_mask(
            height, width, YOLO_TARGET_ROIS[target_key]
        )
        color_saturation = float(colors[index].max() - colors[index].min())
        if color_saturation < 0.10:
            mask = (luminance >= 0.65) & (saturation <= 0.22) & roi
        else:
            mask = (
                (assignments == index)
                & (scores[..., index] >= 0.94)
                & (saturation >= 0.20)
                & roi
            )
        objects[target_key] = _coverage(mask, width, height, roi_xyxy)

    return {
        "camera": "camera_yolo",
        "method": "rgb_solid_color_frame_coverage",
        "resolution": [int(width), int(height)],
        "all_in_frame": all(item["complete"] for item in objects.values()),
        "objects": objects,
    }


def goal_visibility_report(camera) -> dict:
    """Verify that every colored placement disk is completely visible."""
    features = _rgb_features(camera)
    if features is None:
        return {"all_in_frame": False, "error": "rgb_unavailable"}
    _rgb, unit_rgb, saturation = features
    height, width, _ = unit_rgb.shape
    colors = np.asarray(PLACE_MARKER_COLORS, dtype=np.float32)
    colors /= np.linalg.norm(colors, axis=1, keepdims=True)
    scores = unit_rgb @ colors.T
    assignments = scores.argmax(axis=-1)

    goals = {}
    for index, place_pos in enumerate(PLACE_POSITIONS):
        roi, roi_xyxy = _roi_mask(height, width, YOLO_GOAL_ROIS[index])
        mask = (
            (assignments == index)
            & (scores[..., index] >= 0.85)
            & (saturation >= 0.20)
            & roi
        )
        coverage = _coverage(mask, width, height, roi_xyxy)
        coverage["world_xy"] = list(map(float, place_pos))
        goals[f"p{index}"] = coverage

    return {
        "camera": "camera_yolo",
        "method": "rgb_marker_color_frame_coverage",
        "resolution": [int(width), int(height)],
        "all_in_frame": all(item["complete"] for item in goals.values()),
        "goals": goals,
    }
