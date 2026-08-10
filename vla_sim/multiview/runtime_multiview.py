"""Runtime variant that boots the multiview scene instead of the base one.

Does not modify vla_sim/runtime.py. ``SimulationRuntime.start()`` calls
``make_scene_cfg(...)`` inline (it isn't a separately overridable hook),
so the only way to swap in ``MultiviewSceneCfg`` without editing that
file is to override ``start()`` in a subclass. The body below is a
straight copy of the base ``start()`` with the scene-cfg line swapped
for ``make_multiview_scene_cfg(...)`` -- keep this in sync if
``vla_sim/runtime.py`` changes.
"""

from __future__ import annotations

import omni.kit.app
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from vla_sim.config import GRIPPER_USD_RELATIVE, PHYSICS_DT, PLACE_MARKER_COLORS
from vla_sim.isaac_app import log
from vla_sim.runtime import RobotController, RuntimeOptions, SimulationRuntime
from vla_sim.scene import (
    bind_gripper_pad_visuals,
    configure_gripper_pads,
    enable_extensions,
    hide_markers,
    prepare_target_visuals,
    set_marker_material,
    set_plastic_material,
    spawn_raw_and_assemble,
)

from vla_sim.multiview.scene_multiview import make_multiview_scene_cfg


class MultiviewSimulationRuntime(SimulationRuntime):
    """SimulationRuntime that builds MultiviewSceneCfg instead of SceneCfg."""

    def start(self) -> "MultiviewSimulationRuntime":
        """Create the canonical scene plus the two extra cameras and start sim."""
        enable_extensions()
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        disabled = extension_manager.set_extension_enabled_immediate(
            "isaacsim.core.throttling", False
        )
        log(f"isaacsim.core.throttling disabled = {disabled}")
        spawn_raw_and_assemble(gripper_usd_relative=GRIPPER_USD_RELATIVE)
        sim_utils.modify_articulation_root_properties(
            "/World/Robot",
            sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
            ),
        )
        stage = omni.usd.get_context().get_stage()
        configure_gripper_pads(stage)
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(device=self.options.device, dt=PHYSICS_DT)
        )
        # --- the one line that differs from vla_sim.runtime.SimulationRuntime.start():
        self.scene = InteractiveScene(
            make_multiview_scene_cfg(
                num_envs=1,
                env_spacing=2.0,
                stream_width=self.options.stream_width,
                stream_height=self.options.stream_height,
            )
        )
        # ---
        stage = omni.usd.get_context().get_stage()
        log("Applying final scene materials before camera initialization...")
        for path, color in {
            "/World/MatA": (0.08, 0.08, 0.08),
            "/World/MatB": (0.08, 0.08, 0.08),
            "/World/BackdropBack": (0.769, 0.698, 0.541),
            "/World/BackdropSide": (0.769, 0.698, 0.541),
        }.items():
            set_plastic_material(stage, path, color)
        prepare_target_visuals(stage)
        paths = bind_gripper_pad_visuals(stage)
        log(f"Robotiq finger-pad setup applied at: {paths}")
        for index, color in enumerate(PLACE_MARKER_COLORS):
            set_marker_material(stage, f"/World/MarkerP{index}", color)
        if not self.options.show_markers:
            hide_markers(stage)
        log("Final materials applied.")
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera("/World/CameraYolo")
                log("GUI viewport camera = /World/CameraYolo")
        except Exception as exc:
            log(f"GUI viewport camera unchanged: {exc}")
        self.sim.reset()
        self.sim.play()
        self.sim_dt = self.sim.get_physics_dt()
        self.robot_controller = RobotController(self.scene["robot"], str(self.sim.device))
        self.robot_controller.reset_home()
        log(
            "Robotiq finger_joint "
            f"id={self.robot_controller.finger_joint_id} "
            f"soft_limits={self.robot_controller.finger_limits}"
        )
        return self