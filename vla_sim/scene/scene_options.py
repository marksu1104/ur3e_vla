"""Named feature selections for the shared base scene.

Profiles select cameras and presentation-only scene additions.  They never
duplicate the robot, table, or movable-object definitions in ``base_scene``.
This module intentionally does not import simulator APIs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LightingSettings:
    """Renderer-independent lighting values for one scene option."""

    dome_intensity: float = 650.0
    dome_color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    fill_intensity: float = 6500.0
    fill_color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    fill_position: tuple[float, float, float] = (0.13, 0.84, 1.80)
    fill_radius: float = 0.75
    exposure_iso: float = 85.0


@dataclass(frozen=True)
class SceneOptions:
    """Feature selection for one workflow sharing ``BaseSceneCfg``."""

    name: str
    cameras: tuple[str, ...]
    show_destination_fixtures: bool = False
    show_markers: bool = False
    viewport_camera: str | None = None
    lighting: LightingSettings = LightingSettings()


SCENE_OPTIONS = {
    "canonical": SceneOptions(
        name="canonical",
        cameras=("camera_yolo", "camera_policy", "camera_wrist"),
        viewport_camera="camera_yolo",
    ),
    "remote": SceneOptions(
        name="remote",
        cameras=("camera_yolo",),
        show_destination_fixtures=True,
        viewport_camera="camera_yolo",
    ),
    "vla": SceneOptions(
        name="vla",
        cameras=("camera_policy",),
        viewport_camera="camera_policy",
    ),
    "collection": SceneOptions(
        name="collection",
        cameras=("camera_policy", "camera_wrist"),
        viewport_camera="camera_policy",
    ),
    "sync": SceneOptions(
        name="sync",
        cameras=("camera_yolo",),
        viewport_camera="camera_yolo",
    ),
    "printable": SceneOptions(
        name="printable",
        cameras=(),
        show_destination_fixtures=True,
    ),
}


def get_scene_options(options: str | SceneOptions) -> SceneOptions:
    """Resolve one named option set and fail early on typos."""
    if isinstance(options, SceneOptions):
        return options
    try:
        return SCENE_OPTIONS[options]
    except KeyError as exc:
        choices = ", ".join(SCENE_OPTIONS)
        raise ValueError(f"unknown scene options {options!r}; choose from {choices}") from exc
