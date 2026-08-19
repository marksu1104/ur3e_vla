"""Thin runtime specialization for the canonical multi-camera scene."""

from __future__ import annotations

from vla_sim.config import UR3E_MOUNT_ABS
from vla_sim.isaac_app import log
from vla_sim.runtime import SimulationRuntime
from vla_sim.scene import set_plastic_material
from vla_sim.multiview.gripper_marker import attach_gripper_marker
from vla_sim.multiview.scene_multiview import make_multiview_scene_cfg


class MultiviewSimulationRuntime(SimulationRuntime):
    """Canonical runtime with three additional recording cameras."""

    def make_scene_cfg(self):
        return make_multiview_scene_cfg(
            num_envs=1,
            env_spacing=2.0,
            stream_width=self.options.stream_width,
            stream_height=self.options.stream_height,
        )

    def prepare_scene_extras(self, stage) -> None:
        color = (0.769, 0.698, 0.541)
        for path in ("/World/BackdropBack", "/World/BackdropSide"):
            set_plastic_material(stage, path, color)

        marker_path = attach_gripper_marker(stage, UR3E_MOUNT_ABS)
        log(f"Gripper marker attached at: {marker_path}")
