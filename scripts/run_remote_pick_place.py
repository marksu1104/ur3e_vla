"""Serve a scripted three-object UR3e pick-and-place simulation over HTTP/WS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument("--bridge-port", type=int, default=None)
_extra.add_argument("--stream-width", type=int, default=None)
_extra.add_argument("--stream-height", type=int, default=None)
_extra.add_argument("--stream-quality", type=int, default=None)
_extra.add_argument("--stream-every-n-steps", type=int, default=None)
_extra.add_argument("--max-trial-seconds", type=float, default=None)
_extra.add_argument("--show-markers", action="store_true")
_extra.add_argument("--seed", type=int, default=42)
_extra_args, _ = _extra.parse_known_args()

from vla_sim.isaac_app import boot_app, close_app, log

app = boot_app()

import numpy as np
import torch

import omni.usd
import omni.kit.app
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass

from vla_sim.actions import PoseTrajectoryPlayer
from vla_sim.config import (
    BACKDROP_BACK_POS,
    BACKDROP_BACK_SIZE,
    BACKDROP_SIDE_POS,
    BACKDROP_SIDE_SIZE,
    EE_BODY_NAME,
    EE_ORIENT_DOWN,
    GRIPPER_OPEN,
    HOME_POS,
    HOME_Q,
    PHYSICS_DT,
    ROBOT_BASE_POS,
    ROBOT_BASE_ROT,
    TABLE_A_POS,
    TABLE_B_POS,
    TABLE_MAT_A_POS,
    TABLE_MAT_B_POS,
    TABLE_MAT_SIZE,
)
from vla_sim.remote_config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    MAX_TRIAL_SECONDS,
    PLACE_MARKER_COLORS,
    PLACE_MARKER_RADIUS,
    PLACE_MARKER_THICKNESS,
    PLACE_POSITIONS,
    REMOTE_GRIPPER_CLOSE,
    REMOTE_GRIPPER_USD_RELATIVE,
    REMOTE_TARGET_KEYS,
    REMOTE_TARGETS,
    STREAM_EVERY_N_STEPS,
    STREAM_HEIGHT,
    STREAM_JPEG_QUALITY,
    STREAM_WIDTH,
    TASK_INDEX_MAP,
    YOLO_CAMERA_FOCAL,
    YOLO_CAMERA_POS,
    YOLO_CAMERA_ROT,
)
from vla_sim.demo_planning import detect_success
from vla_sim.remote_bridge import RemoteBridge
from vla_sim.remote_planning import GRIPPER_SPEED_RAD_S, build_pick_place_trajectory
from vla_sim.remote_scene import (
    bind_gripper_pad_visuals,
    configure_gripper_pads,
    prepare_target_visuals,
    set_marker_material,
    set_plastic_material,
)
from vla_sim.remote_visibility import goal_visibility_report, object_visibility_report
from vla_sim.scene import (
    enable_extensions,
    make_static_cuboid_cfg,
    make_table_cfg,
    make_target_cfg,
    spawn_raw_and_assemble,
)

def _marker_cfg(index: int) -> AssetBaseCfg:
    """Create a collision-free, smooth visual disk flush with the table mat."""
    x, y = PLACE_POSITIONS[index]
    table_surface_z = TABLE_MAT_A_POS[2] + 0.5 * TABLE_MAT_SIZE[2]
    marker_center_z = table_surface_z + 0.5 * PLACE_MARKER_THICKNESS
    return AssetBaseCfg(
        prim_path=f"/World/MarkerP{index}",
        spawn=sim_utils.MeshCylinderCfg(
            radius=PLACE_MARKER_RADIUS,
            height=PLACE_MARKER_THICKNESS,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, marker_center_z)),
    )


def _remote_target_cfg(name: str):
    return make_target_cfg(name, REMOTE_TARGETS[name])


@configclass
class RemoteSceneCfg(InteractiveSceneCfg):
    """The normal single-arm scene with only the remote-bridge task objects."""

    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS,
            rot=ROBOT_BASE_ROT,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                stiffness=10000.0,
                damping=500.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
            ),
            # Match Isaac Lab's official UR10e + Robotiq 2F-140 actuator
            # configuration. Only finger_joint is actively commanded; the
            # remaining closed-loop linkage joints must stay compliant/passive.
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=11.25,
                damping=0.1,
                effort_limit_sim=10.0,
                velocity_limit_sim=1.0,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                stiffness=0.2,
                damping=0.001,
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_inner_finger_pad_joint",
                    ".*_outer_finger_joint",
                    "right_outer_knuckle_joint",
                ],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
            ),
        },
    )
    table_a = make_table_cfg("/World/TableA", TABLE_A_POS)
    table_b = make_table_cfg("/World/TableB", TABLE_B_POS)
    mat_a = make_static_cuboid_cfg("/World/MatA", TABLE_MAT_SIZE, TABLE_MAT_A_POS)
    mat_b = make_static_cuboid_cfg("/World/MatB", TABLE_MAT_SIZE, TABLE_MAT_B_POS)
    backdrop_back = make_static_cuboid_cfg(
        "/World/BackdropBack", BACKDROP_BACK_SIZE, BACKDROP_BACK_POS
    )
    backdrop_side = make_static_cuboid_cfg(
        "/World/BackdropSide", BACKDROP_SIDE_SIZE, BACKDROP_SIDE_POS
    )
    red_mug = _remote_target_cfg("red_mug")
    spoon = _remote_target_cfg("spoon")
    bowl = _remote_target_cfg("bowl")
    marker_p0 = _marker_cfg(0)
    marker_p1 = _marker_cfg(1)
    marker_p2 = _marker_cfg(2)

    camera_yolo = CameraCfg(
        prim_path="/World/CameraYolo",
        update_period=0.0,
        height=(
            STREAM_HEIGHT
            if _extra_args.stream_height is None
            else _extra_args.stream_height
        ),
        width=(
            STREAM_WIDTH
            if _extra_args.stream_width is None
            else _extra_args.stream_width
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=YOLO_CAMERA_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=YOLO_CAMERA_POS, rot=YOLO_CAMERA_ROT, convention="opengl"
        ),
    )


def _hide_markers(stage) -> None:
    """Keep placement markers hidden in every bridge state."""
    from pxr import UsdGeom

    for index in range(len(PLACE_POSITIONS)):
        prim = stage.GetPrimAtPath(f"/World/MarkerP{index}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                UsdGeom.Tokens.invisible
            )


def _reset_remote_targets(scene, device: str) -> None:
    """Restore the fixed remote YOLO composition."""
    for name in REMOTE_TARGET_KEYS:
        pos = REMOTE_TARGETS[name]["spawn_pos"]
        rot = REMOTE_TARGETS[name]["spawn_rot"]
        obj = scene[name]
        obj.write_root_pose_to_sim(
            torch.tensor([[*pos, *rot]], device=device, dtype=torch.float32)
        )
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))


def _apply_ik(robot, ik, arm_ids_t, ee_body_idx, ee_jac_idx, target_pos, target_quat):
    root_pos = robot.data.root_state_w[:, :3]
    ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
    ee_pos_b = ee_pose_w[:, :3] - root_pos
    ee_quat_b = ee_pose_w[:, 3:]
    ik.set_command(
        torch.cat(
            [target_pos.unsqueeze(0) - root_pos, target_quat.unsqueeze(0)],
            dim=-1,
        )
    )
    jac_full = robot.root_physx_view.get_jacobians()
    jac = jac_full[:, ee_jac_idx, :, :][:, :, arm_ids_t]
    q_target = ik.compute(ee_pos_b, ee_quat_b, jac, robot.data.joint_pos[:, arm_ids_t])
    robot.set_joint_position_target(q_target, joint_ids=arm_ids_t)


def main() -> None:
    port = BRIDGE_PORT if _extra_args.bridge_port is None else _extra_args.bridge_port
    stream_every = (
        STREAM_EVERY_N_STEPS
        if _extra_args.stream_every_n_steps is None
        else _extra_args.stream_every_n_steps
    )
    quality = (
        STREAM_JPEG_QUALITY
        if _extra_args.stream_quality is None
        else _extra_args.stream_quality
    )
    max_trial_seconds = (
        MAX_TRIAL_SECONDS
        if _extra_args.max_trial_seconds is None
        else _extra_args.max_trial_seconds
    )
    if stream_every < 1:
        raise ValueError("--stream-every-n-steps must be at least 1")

    bridge = RemoteBridge(
        TASK_INDEX_MAP,
        len(PLACE_POSITIONS),
        host=BRIDGE_HOST,
        port=port,
        jpeg_quality=quality,
    )
    bridge.start()
    bridge.set_state("starting")
    try:
        enable_extensions()
        # The GUI throttling extension's timeline-play callback can recurse through
        # carb.settings/omni.usd during sim.reset() in Isaac Sim 6.0. Scripted
        # simulation owns its update loop, so the extension is not needed here.
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        disabled = extension_manager.set_extension_enabled_immediate(
            "isaacsim.core.throttling", False
        )
        log(f"isaacsim.core.throttling disabled = {disabled}")
        spawn_raw_and_assemble(gripper_usd_relative=REMOTE_GRIPPER_USD_RELATIVE)
        # NVIDIA's Robotiq setup guide recommends 64/4 solver iterations for
        # the 2F-140 closed linkage. This prevents the loop constraints from
        # visibly folding or stretching under contact.
        sim_utils.modify_articulation_root_properties(
            "/World/Robot",
            sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
            ),
        )
        configure_gripper_pads(omni.usd.get_context().get_stage())
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT)
        )
        scene = InteractiveScene(RemoteSceneCfg(num_envs=1, env_spacing=2.0))
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
        # Scene construction can recompose referenced instance proxies, so
        # reapply only their render binding before the first reset.
        gripper_visual_paths = bind_gripper_pad_visuals(stage)
        log(f"Robotiq finger-pad setup applied at: {gripper_visual_paths}")
        for index, color in enumerate(PLACE_MARKER_COLORS):
            set_marker_material(stage, f"/World/MarkerP{index}", color)
        if not _extra_args.show_markers:
            _hide_markers(stage)
        log("Final materials applied.")
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera("/World/CameraYolo")
                log("GUI viewport camera = /World/CameraYolo")
        except Exception as exc:
            log(f"GUI viewport camera unchanged: {exc}")
        sim.reset()
        sim.play()
        sim_dt = sim.get_physics_dt()

        robot = scene["robot"]
        device = str(sim.device)
        root_pose = torch.tensor(
            [[*ROBOT_BASE_POS, *ROBOT_BASE_ROT]],
            device=device,
            dtype=torch.float32,
        )
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        arm_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        arm_ids, _ = robot.find_joints(arm_names)
        arm_ids_t = torch.tensor(arm_ids, dtype=torch.long, device=device)
        finger_ids, _ = robot.find_joints(["finger_joint"])
        if len(finger_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one Robotiq drive joint, found {finger_ids}"
            )
        finger_ids_t = torch.tensor(finger_ids, dtype=torch.long, device=device)
        ee_ids, _ = robot.find_bodies([EE_BODY_NAME])
        ee_body_idx = ee_ids[0]
        ee_jac_idx = robot.body_names.index(EE_BODY_NAME) - 1
        finger_joint_id, _ = robot.find_joints(["finger_joint"])
        finger_limits = (
            robot.data.soft_joint_pos_limits[0, finger_joint_id[0]]
            .cpu()
            .tolist()
        )
        log(
            "Robotiq finger_joint "
            f"id={finger_joint_id[0]} soft_limits={finger_limits}"
        )
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=1,
            device=device,
        )
        home_q_t = torch.tensor([HOME_Q], device=device, dtype=torch.float32)

        state = "resetting"
        paused_from_state = "running"
        seed = _extra_args.seed
        _reset_remote_targets(scene, device)
        robot.write_joint_state_to_sim(
            home_q_t,
            torch.zeros_like(home_q_t),
            joint_ids=arm_ids_t,
        )
        settle_steps = 120
        trial = None
        player = None
        run_t = 0.0
        target_initial_z = 0.0
        best_lift = 0.0
        max_gripper_command = GRIPPER_OPEN
        max_finger_joint = GRIPPER_OPEN
        frozen_pos = torch.tensor(HOME_POS, device=device)
        frozen_quat = torch.tensor(EE_ORIENT_DOWN, device=device)
        raw_grip = GRIPPER_OPEN
        step = 0
        bridge.set_state("resetting", seed=seed)

        while app.is_running():
            command = bridge.poll_command()
            if command is not None:
                bridge.command_applied(command)
                if command["type"] == "task" and state == "waiting":
                    trial = command
                    target = scene[command["object"]]
                    resting = target.data.root_pos_w[0].cpu().numpy()
                    rotation = target.data.root_state_w[0, 3:7].cpu().numpy()
                    target_initial_z = float(resting[2])
                    best_lift = 0.0
                    max_gripper_command = float(GRIPPER_OPEN)
                    max_finger_joint = float(GRIPPER_OPEN)
                    run_t = 0.0
                    target_info = REMOTE_TARGETS[command["object"]]
                    trajectory = build_pick_place_trajectory(
                        target_info,
                        resting,
                        rotation,
                        PLACE_POSITIONS[command["position_index"]],
                    )
                    player = PoseTrajectoryPlayer(trajectory, device=device)

                    frozen_pos, frozen_quat, raw_grip, _ = player.sample(0.0)
                    state = "running"
                    paused_from_state = "running"
                    bridge.set_state(
                        "running",
                        trial_id=command["trial_id"],
                        object=command["object"],
                        task_index=command["task_index"],
                        position_index=command["position_index"],
                        progress={"t": 0.0, "total": player.total_time},
                        seed=seed,
                    )
                elif command["type"] == "control":
                    action = command["action"]
                    if action == "pause" and state in {"running", "settling"}:
                        paused_from_state = state
                        state = "paused"
                        bridge.set_state(
                            "paused",
                            progress={"t": run_t, "total": player.total_time},
                        )
                    elif action == "resume" and state == "paused":
                        state = paused_from_state
                        bridge.set_state(
                            "running",
                            progress={"t": run_t, "total": player.total_time},
                        )
                    elif action == "reset":
                        seed = seed if command["seed"] is None else command["seed"]
                        _reset_remote_targets(scene, device)
                        robot.write_joint_state_to_sim(
                            home_q_t,
                            torch.zeros_like(home_q_t),
                            joint_ids=arm_ids_t,
                        )
                        state = "resetting"
                        paused_from_state = "running"
                        trial = None
                        player = None
                        settle_steps = 120
                        frozen_pos = torch.tensor(HOME_POS, device=device)
                        frozen_quat = torch.tensor(EE_ORIENT_DOWN, device=device)
                        raw_grip = GRIPPER_OPEN
                        bridge.set_state("resetting", seed=seed)

            if state == "running" and player is not None:
                run_t += sim_dt
                frozen_pos, frozen_quat, desired_grip, finished = player.sample(run_t)
                grip_step = GRIPPER_SPEED_RAD_S * sim_dt
                raw_grip = float(
                    np.clip(
                        float(raw_grip)
                        + np.clip(
                            float(desired_grip) - float(raw_grip),
                            -grip_step,
                            grip_step,
                        ),
                        GRIPPER_OPEN,
                        REMOTE_GRIPPER_CLOSE,
                    )
                )
                target = scene[trial["object"]]
                lift_height = (
                    float(target.data.root_pos_w[0, 2].item()) - target_initial_z
                )
                best_lift = max(best_lift, lift_height)
                finger_q = (
                    float(robot.data.joint_pos[0, finger_joint_id[0]].item())
                    if finger_joint_id
                    else float(raw_grip)
                )
                max_gripper_command = max(max_gripper_command, float(raw_grip))
                max_finger_joint = max(max_finger_joint, finger_q)
                bridge.update_status(
                    progress={
                        "t": min(run_t, player.total_time),
                        "total": player.total_time,
                        "gripper_command": float(raw_grip),
                        "finger_joint": finger_q,
                    }
                )
                if run_t > max_trial_seconds:
                    finished = True
                if finished:
                    settle_steps = 40
                    state = "settling"
            elif state == "settling":
                settle_steps -= 1
                if settle_steps <= 0 and trial is not None:
                    target = scene[trial["object"]]
                    ee_pos = robot.data.body_state_w[0, ee_body_idx, :3].cpu().numpy()
                    gripper_q = float(
                        robot.data.joint_pos[0, finger_joint_id[0]].item()
                    )
                    threshold = 0.12 if trial["object"] == "bowl" else 0.10
                    place_pos = (
                        *PLACE_POSITIONS[trial["position_index"]],
                        0.0,
                    )
                    success, detail = detect_success(
                        target,
                        ee_pos,
                        target_initial_z,
                        gripper_q,
                        place_pos,
                        best_lift,
                        place_xy_threshold=threshold,
                    )
                    detail.update(
                        {
                            "gripper_close_target": float(REMOTE_GRIPPER_CLOSE),
                            "max_gripper_command": float(max_gripper_command),
                            "max_finger_joint": float(max_finger_joint),
                        }
                    )
                    result = {
                        "success": bool(success),
                        "detail": detail,
                        "reason": (
                            "completed" if success else "success_checks_failed"
                        ),
                    }
                    state = "done"
                    bridge.finish_trial(
                        trial_id=trial["trial_id"],
                        result=result,
                        progress={"t": player.total_time, "total": player.total_time},
                    )

            if state == "resetting":
                settle_steps -= 1
                if settle_steps <= 0:
                    poses = {
                        name: scene[name]
                        .data.root_pos_w[0]
                        .cpu()
                        .numpy()
                        .round(5)
                        .tolist()
                        for name in REMOTE_TARGET_KEYS
                    }
                    waiting_finger_q = float(
                        robot.data.joint_pos[0, finger_joint_id[0]].item()
                    )
                    yolo_visibility = object_visibility_report(scene["camera_yolo"])
                    goal_visibility = (
                        goal_visibility_report(scene["camera_yolo"])
                        if _extra_args.show_markers
                        else None
                    )
                    log(f"YOLO visibility: {yolo_visibility}")
                    if goal_visibility is not None:
                        log(f"YOLO goal visibility: {goal_visibility}")
                    state = "waiting"
                    bridge.set_state(
                        "waiting",
                        object_poses=poses,
                        seed=seed,
                        gripper={
                            "command": float(raw_grip),
                            "finger_joint": waiting_finger_q,
                            "soft_limits": finger_limits,
                        },
                        yolo_visibility=yolo_visibility,
                        goal_visibility=goal_visibility,
                    )
            _apply_ik(
                robot,
                ik,
                arm_ids_t,
                ee_body_idx,
                ee_jac_idx,
                frozen_pos,
                frozen_quat,
            )
            # The official physics asset propagates finger_joint through the
            # mimic/closed-loop constraints. Driving the other linkage joints
            # independently makes the fingers skew and the pads diverge.
            finger_cmd = torch.full(
                (1, 1), float(raw_grip), dtype=torch.float32, device=device
            )
            robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids_t)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            if step % stream_every == 0:
                rgb = scene["camera_yolo"].data.output.get("rgb")
                if rgb is not None:
                    bridge.publish_frame(rgb[0].cpu().numpy().astype(np.uint8))
            step += 1
    except KeyboardInterrupt:
        log("Ctrl+C received.")
    finally:
        bridge.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_app()
