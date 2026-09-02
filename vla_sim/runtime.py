"""Shared canonical UR3e simulation runtime and normalized robot controller."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

import omni.kit.app
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene

from vla_sim.config import (
    ARM_JOINT_NAMES,
    EE_BODY_NAME,
    EE_ORIENT_DOWN,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOME_POS,
    HOME_Q,
    PHYSICS_DT,
    PLACE_MARKER_COLORS,
    ROBOT_BASE_POS,
    ROBOT_BASE_ROT,
    get_scene_profile,
)
from vla_sim.isaac_app import log
from vla_sim.fixtures import prepare_destination_fixtures
from vla_sim.planning import GRIPPER_SPEED_RAD_S
from vla_sim.scene import (
    make_scene_cfg,
    quat_isaac_to_wxyz,
    quat_wxyz_to_isaac,
    spawn_assembled_robot,
)
from vla_sim.visuals import (
    bind_gripper_pad_visuals,
    configure_gripper_pads,
    hide_markers,
    prepare_target_visuals,
    set_marker_material,
    set_plastic_material,
)


def as_torch(array) -> torch.Tensor:
    """Return a zero-copy Torch view for old tensors and 3.0 array proxies."""
    if isinstance(array, torch.Tensor):
        return array
    tensor = getattr(array, "torch", None)
    if tensor is not None:
        return tensor
    try:
        import warp as wp

        if isinstance(array, wp.array):
            return wp.to_torch(array)
    except ImportError:
        pass
    raise TypeError(f"Unsupported simulation array type: {type(array)!r}")


def pose_wxyz_from_sim(pose) -> torch.Tensor:
    """Return a simulation pose in the project's stable xyz+wxyz layout."""
    pose_tensor = as_torch(pose)
    return torch.cat(
        (pose_tensor[..., :3], quat_isaac_to_wxyz(pose_tensor[..., 3:7])), dim=-1
    )

class StateBackend(ABC):
    """Apply one authoritative robot state before each simulation step."""

    @abstractmethod
    def apply(self, controller: "RobotController", dt: float) -> None:
        pass


class PhysicsDriveBackend(StateBackend):
    def apply(self, controller: "RobotController", dt: float) -> None:
        del dt
        controller.apply_physics_targets()


class ExternalStateBackend(StateBackend, ABC):
    """Read-only interface reserved for measured robot state."""


@dataclass(frozen=True)
class RuntimeOptions:
    """Scene options shared by all canonical-scene entry points."""

    scene_profile: str = "canonical"
    stream_width: int | None = None
    stream_height: int | None = None
    show_markers: bool | None = None
    show_destination_fixtures: bool | None = None
    device: str = "cuda:0"


class RobotController:
    """Control the canonical UR3e with a normalized 0..1 gripper command."""

    def __init__(self, robot, device: str):
        self.robot = robot
        self.device = device
        self.arm_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES))
        if len(self.arm_ids) != len(ARM_JOINT_NAMES):
            raise RuntimeError(f"Expected six UR3e arm joints, found {self.arm_ids}")
        self.arm_ids_t = torch.tensor(self.arm_ids, dtype=torch.int32, device=device)

        finger_ids, _ = robot.find_joints(["finger_joint"])
        if len(finger_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one Robotiq drive joint, found {finger_ids}"
            )
        self.finger_joint_id = finger_ids[0]
        self.finger_ids_t = torch.tensor(finger_ids, dtype=torch.int32, device=device)
        self.finger_limits = (
            as_torch(robot.data.soft_joint_pos_limits)[
                0, self.finger_joint_id
            ].cpu().tolist()
        )

        ee_ids, _ = robot.find_bodies([EE_BODY_NAME])
        if len(ee_ids) != 1:
            raise RuntimeError(f"Expected end-effector body {EE_BODY_NAME!r}")
        self.ee_body_idx = ee_ids[0]
        self.ee_jac_idx = robot.body_names.index(EE_BODY_NAME) - 1
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=1,
            device=device,
        )
        self.home_q = torch.tensor([HOME_Q], device=device, dtype=torch.float32)
        self._target_pos = torch.tensor(HOME_POS, device=device, dtype=torch.float32)
        self._target_quat = torch.tensor(
            quat_wxyz_to_isaac(EE_ORIENT_DOWN), device=device, dtype=torch.float32
        )
        self._logical_gripper_command = 0.0
        self._physical_gripper_command = float(GRIPPER_OPEN)

    @property
    def logical_gripper_command(self) -> float:
        """Requested gripper state: 0.0 open, 1.0 closed."""
        return self._logical_gripper_command

    @property
    def physical_gripper_command(self) -> float:
        """Current target in the official Robotiq ``finger_joint`` radians."""
        return self._physical_gripper_command

    @property
    def finger_position(self) -> float:
        return float(
            as_torch(self.robot.data.joint_pos)[0, self.finger_joint_id].item()
        )

    @property
    def ee_position(self) -> np.ndarray:
        return (
            as_torch(self.robot.data.body_state_w)[0, self.ee_body_idx, :3].cpu().numpy()
        )

    def reset_home(self) -> None:
        """Reset only the virtual robot to its existing home configuration."""
        root_pose = torch.tensor(
            [[*ROBOT_BASE_POS, *quat_wxyz_to_isaac(ROBOT_BASE_ROT)]],
            device=self.device,
            dtype=torch.float32,
        )
        self.robot.write_root_pose_to_sim(root_pose)
        self.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))
        self.robot.write_joint_state_to_sim(
            self.home_q,
            torch.zeros_like(self.home_q),
            joint_ids=self.arm_ids_t,
        )
        self.set_pose_target(HOME_POS, EE_ORIENT_DOWN)
        self.set_gripper_command(0.0, rate_limit=False)

    def set_pose_target(self, position, quaternion) -> None:
        """Set the world-frame pose target applied at the next physics step."""
        self._target_pos = torch.as_tensor(
            position, device=self.device, dtype=torch.float32
        )
        target_quat = torch.as_tensor(
            quaternion, device=self.device, dtype=torch.float32
        )
        self._target_quat = quat_wxyz_to_isaac(target_quat)

    def set_gripper_command(
        self, command: float, dt: float | None = None, *, rate_limit: bool = True
    ) -> None:
        """Map a logical 0..1 request to the physical finger joint.

        The physical rate limit exactly preserves the remote scripted behavior.
        """
        logical = float(np.clip(command, 0.0, 1.0))
        target = float(
            GRIPPER_OPEN + logical * (GRIPPER_CLOSE - GRIPPER_OPEN)
        )
        if rate_limit:
            if dt is None:
                raise ValueError("dt is required when rate_limit=True")
            step = GRIPPER_SPEED_RAD_S * float(dt)
            target = float(
                np.clip(
                    self._physical_gripper_command
                    + np.clip(target - self._physical_gripper_command, -step, step),
                    GRIPPER_OPEN,
                    GRIPPER_CLOSE,
                )
            )
        self._logical_gripper_command = logical
        self._physical_gripper_command = target

    def write_measured_arm_state(self, positions) -> None:
        """Write six measured arm joints without touching the virtual gripper."""
        values = torch.as_tensor(positions, device=self.device, dtype=torch.float32)
        if values.numel() != len(ARM_JOINT_NAMES):
            raise ValueError("expected exactly six measured UR3e arm joint values")
        values = values.reshape(1, len(ARM_JOINT_NAMES))
        self.robot.write_joint_state_to_sim(
            values,
            torch.zeros_like(values),
            joint_ids=self.arm_ids_t,
        )
        self.robot.set_joint_position_target(values, joint_ids=self.arm_ids_t)

    def apply_gripper_target(self) -> None:
        """Re-assert only the finger target, leaving the arm alone.

        External-state backends drive the arm from measured joint values and
        never call ``apply_physics_targets``, so without this the virtual
        gripper's target is set once and then not maintained.
        """
        finger_target = torch.full(
            (1, 1),
            self._physical_gripper_command,
            dtype=torch.float32,
            device=self.device,
        )
        self.robot.set_joint_position_target(finger_target, joint_ids=self.finger_ids_t)

    def write_commanded_gripper_state(self, logical: float, dt: float | None = None) -> None:
        """Track a commanded 0..1 gripper state on the virtual robot.

        Used by real-to-sim style mirroring, where the arm's pose comes from
        measured joint states but the gripper has none: the real Robotiq is
        driven over a digital output and reports only open/grasped digital
        inputs, never a joint position. So the mirror can only reflect what was
        commanded, not what was measured.

        Pass ``dt`` to close at the simulation's own finger speed. The real
        gripper is smaller and snaps shut far faster than the simulated one;
        writing that end state directly teleports the virtual fingers through
        the object, which resolves as an explosive contact instead of a grasp.
        Ramping keeps the virtual contact physically meaningful even though the
        real gripper has already finished moving.
        """
        self.set_gripper_command(logical, dt, rate_limit=dt is not None)
        self.apply_gripper_target()

    def apply_physics_targets(self) -> None:
        """Apply IK arm targets and the single official Robotiq drive joint."""
        root_pos = as_torch(self.robot.data.root_link_pose_w)[:, :3]
        ee_pose_w = as_torch(self.robot.data.body_state_w)[:, self.ee_body_idx, :7]
        ee_pos_b = ee_pose_w[:, :3] - root_pos
        ee_quat_b = ee_pose_w[:, 3:]
        self.ik.set_command(
            torch.cat(
                [self._target_pos.unsqueeze(0) - root_pos, self._target_quat.unsqueeze(0)],
                dim=-1,
            )
        )
        jac_full = as_torch(self.robot.root_physx_view.get_jacobians())
        jac = jac_full[:, self.ee_jac_idx, :, :][:, :, self.arm_ids]
        q_target = self.ik.compute(
            ee_pos_b,
            ee_quat_b,
            jac,
            as_torch(self.robot.data.joint_pos)[:, self.arm_ids],
        )
        self.robot.set_joint_position_target(q_target, joint_ids=self.arm_ids_t)
        finger_target = torch.full(
            (1, 1),
            self._physical_gripper_command,
            dtype=torch.float32,
            device=self.device,
        )
        self.robot.set_joint_position_target(finger_target, joint_ids=self.finger_ids_t)


class SimulationRuntime:
    """Own the canonical scene lifecycle without bridge or task-policy logic."""

    def __init__(
        self,
        options: RuntimeOptions | None = None,
        *,
        state_backend: StateBackend | None = None,
    ):
        self.options = options or RuntimeOptions()
        self.profile = get_scene_profile(self.options.scene_profile)
        self.state_backend = state_backend or PhysicsDriveBackend()
        self.sim = None
        self.scene = None
        self.robot_controller = None
        self.sim_dt = PHYSICS_DT

    @property
    def robot(self):
        if self.robot_controller is None:
            raise RuntimeError("SimulationRuntime.start() has not completed")
        return self.robot_controller.robot

    @property
    def device(self) -> str:
        if self.sim is None:
            raise RuntimeError("SimulationRuntime.start() has not completed")
        return str(self.sim.device)

    def make_scene_cfg(self):
        """Build the scene configuration used by this runtime."""
        return make_scene_cfg(
            num_envs=1,
            env_spacing=2.0,
            stream_width=self.options.stream_width,
            stream_height=self.options.stream_height,
            profile=self.profile,
        )

    def prepare_scene_extras(self, stage) -> None:
        """Hook for visual-only additions in specialized runtimes."""
        del stage

    def start(self) -> "SimulationRuntime":
        """Create the exact canonical remote scene and start simulation."""
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(device=self.options.device, dt=PHYSICS_DT)
        )
        spawn_assembled_robot()
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
        self.scene = InteractiveScene(self.make_scene_cfg())
        stage = omni.usd.get_context().get_stage()
        log("Applying final scene materials before camera initialization...")
        for path, color in {
            "/World/MatA": (0.08, 0.08, 0.08),
            "/World/MatB": (0.08, 0.08, 0.08),
            "/World/BackdropBack": (0.005, 0.005, 0.005),
            "/World/BackdropSide": (0.005, 0.005, 0.005),
        }.items():
            set_plastic_material(stage, path, color)
        prepare_target_visuals(stage)
        show_fixtures = (
            self.profile.show_destination_fixtures
            if self.options.show_destination_fixtures is None
            else self.options.show_destination_fixtures
        )
        if show_fixtures:
            prepare_destination_fixtures(stage)
        paths = bind_gripper_pad_visuals(stage)
        log(f"Robotiq finger-pad setup applied at: {paths}")
        self.prepare_scene_extras(stage)
        from pxr import UsdGeom

        if hasattr(UsdGeom.Camera, "GetExposureIsoAttr"):
            camera_iso = self.profile.lighting.exposure_iso
            cameras = [
                UsdGeom.Camera(prim)
                for prim in stage.Traverse()
                if prim.IsA(UsdGeom.Camera)
            ]
            for camera in cameras:
                camera.GetExposureIsoAttr().Set(camera_iso)

        for index, color in enumerate(PLACE_MARKER_COLORS):
            set_marker_material(stage, f"/World/MarkerP{index}", color)
        show_markers = (
            self.profile.show_markers
            if self.options.show_markers is None
            else self.options.show_markers
        )
        if not show_markers:
            hide_markers(stage)
        log("Final materials applied.")
        self.sim.reset()
        self.sim.play()
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None and self.profile.viewport_camera is not None:
                camera_key = self.profile.viewport_camera
                camera_prim = self.scene[camera_key].cfg.prim_path
                viewport.set_active_camera(camera_prim)
                log(f"GUI viewport camera = {camera_prim}")
        except Exception as exc:
            log(f"GUI viewport camera unchanged: {exc}")
        self.sim_dt = self.sim.get_physics_dt()
        self.robot_controller = RobotController(self.scene["robot"], str(self.sim.device))
        self.robot_controller.reset_home()
        log(
            f"Scene profile={self.profile.name} "
            f"cameras={list(self.profile.cameras)} fixtures={show_fixtures}"
        )
        log(
            "Robotiq finger_joint "
            f"id={self.robot_controller.finger_joint_id} "
            f"soft_limits={self.robot_controller.finger_limits}"
        )
        return self

    def reset_targets(self) -> None:
        """Restore the fixed canonical object composition."""
        if self.scene is None:
            raise RuntimeError("SimulationRuntime.start() has not completed")
        from vla_sim.config import TARGET_KEYS, TARGETS

        for name in TARGET_KEYS:
            pos = TARGETS[name]["spawn_pos"]
            rot = TARGETS[name]["spawn_rot"]
            obj = self.scene[name]
            obj.write_root_pose_to_sim(
                torch.tensor([[*pos, *quat_wxyz_to_isaac(rot)]], device=self.device)
            )
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))

    def step(self) -> None:
        """Apply the authoritative state and advance exactly one sim frame."""
        if self.sim is None or self.scene is None or self.robot_controller is None:
            raise RuntimeError("SimulationRuntime.start() has not completed")
        self.state_backend.apply(self.robot_controller, self.sim_dt)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim_dt)

    def latest_yolo_rgb(self):
        """Return the current RGB tensor used by the bridge stream."""
        if self.scene is None:
            raise RuntimeError("SimulationRuntime.start() has not completed")
        return self.scene["camera_yolo"].data.output.get("rgb")
