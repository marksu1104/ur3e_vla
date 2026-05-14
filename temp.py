"""
sim_r.py — UR3e + Robotiq 2F-140 + YCB grasp + multi-view recording.

Usage:
    ./isaaclab.sh -p sim_r.py --target banana
    ./isaaclab.sh -p sim_r.py --target mug

    # 跑完軌跡後額外用 orbit camera 拍 24 張環繞快照:
    ./isaaclab.sh -p sim_r.py --target mug --orbit-snapshots 24

主要流程:
  Phase 1 — spawn + Robot Assembler  (在 envs.sim_setup 內)
  Phase 2 — SimulationContext + InteractiveScene
  Phase 3 — sim.reset() + sim.play()
  Phase 4 — robot init (joints, IK, gripper)
  Main loop — IK trajectory + dual-camera mp4/png recording + orbit capture

模組分布:
  envs/
    boot.py       — AppLauncher + IsaacLab 3.0 workarounds, args_cli, log
    config.py     — TARGETS, GRIPPER_MIMIC_MAP, 所有常數
    sim_setup.py  — Phase 1 spawn, SceneCfg, hide_proxy_meshes
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 1 — Boot SimulationApp (必須最先做)                      ║
# ╚══════════════════════════════════════════════════════════════════╝

from envs.boot import boot_app, args_cli, log
app = boot_app()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 2 — Imports (post-launch)                                ║
# ╚══════════════════════════════════════════════════════════════════╝

from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from envs.config import (
    PHYSICS_DT,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT,
    ORBIT_CAMERA_WIDTH, ORBIT_CAMERA_HEIGHT,
    ROBOT_BASE_POS, ROBOT_BASE_ROT,
    EE_BODY_NAME, EE_ORIENT_DOWN,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    HOME_POS, HOME_Q,
    ORBIT_CENTER, ORBIT_RADIUS, ORBIT_HEIGHT,
    TARGETS,
    GRIPPER_MIMIC_MAP,
    CAMERA_MAIN_POS, CAMERA_MAIN_ROT, CAMERA_MAIN_FOCAL,
)
from envs.sim_setup import (
    enable_extensions,
    spawn_raw_and_assemble,
    SceneCfg,
    hide_proxy_meshes,
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 3 — Trajectory player                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class PoseTrajectoryPlayer:
    """線性插值 7-segment 軌跡 (pos, quat, gripper)."""

    def __init__(self, trajectory, device):
        self.device = device
        self.segments = []
        t0 = 0.0
        prev = trajectory[0]
        for duration, pos, quat, grip in trajectory:
            self.segments.append((
                t0, t0 + duration,
                torch.tensor(prev[1], device=device),
                torch.tensor(pos,     device=device),
                torch.tensor(prev[2], device=device),
                torch.tensor(quat,    device=device),
                float(prev[3]),
                float(grip),
            ))
            t0 += duration
            prev = (duration, pos, quat, grip)
        self.total_time = t0

    def sample(self, t: float):
        if t >= self.total_time:
            _, _, _, p1, _, q1, _, g1 = self.segments[-1]
            return p1, q1, g1, True
        for start, end, p0, p1, q0, q1, g0, g1 in self.segments:
            if start <= t < end:
                a = (t - start) / (end - start)
                pos  = p0 + a * (p1 - p0)
                quat = q0 + a * (q1 - q0)
                quat = quat / torch.linalg.norm(quat)
                grip = g0 + a * (g1 - g0)
                return pos, quat, grip, False
        _, _, _, p1, _, q1, _, g1 = self.segments[-1]
        return p1, q1, g1, True


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 4 — Output paths / video writers                         ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class OutputPaths:
    output_dir: Path
    video_main_path: Path
    video_wrist_path: Path
    frames_main_dir: Path
    frames_wrist_dir: Path
    snapshots_dir: Path


def build_output_paths(target_name: str) -> OutputPaths:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs" / f"sim_{target_name}"
    frames_main_dir  = output_dir / "main_frames"
    frames_wrist_dir = output_dir / "wrist_frames"
    snapshots_dir    = output_dir / "snapshots"
    for d in (output_dir, frames_main_dir, frames_wrist_dir, snapshots_dir):
        d.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        output_dir=output_dir,
        video_main_path=output_dir / f"{target_name}_main.mp4",
        video_wrist_path=output_dir / f"{target_name}_wrist.mp4",
        frames_main_dir=frames_main_dir,
        frames_wrist_dir=frames_wrist_dir,
        snapshots_dir=snapshots_dir,
    )


def create_video_writers(paths: OutputPaths, camera_fps: int):
    h264 = cv2.VideoWriter_fourcc(*"avc1")
    out_main  = cv2.VideoWriter(str(paths.video_main_path),  h264, camera_fps, (CAMERA_WIDTH, CAMERA_HEIGHT))
    out_wrist = cv2.VideoWriter(str(paths.video_wrist_path), h264, camera_fps, (WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT))
    if not out_main.isOpened() or not out_wrist.isOpened():
        # Fallback to mp4v codec
        mp4v = cv2.VideoWriter_fourcc(*"mp4v")
        out_main  = cv2.VideoWriter(str(paths.video_main_path),  mp4v, camera_fps, (CAMERA_WIDTH, CAMERA_HEIGHT))
        out_wrist = cv2.VideoWriter(str(paths.video_wrist_path), mp4v, camera_fps, (WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT))
    if not out_main.isOpened() or not out_wrist.isOpened():
        raise RuntimeError("Failed to create video writers.")
    return out_main, out_wrist


def to_bgr(image_np):
    if image_np.shape[-1] == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 5 — Orbit capture (snapshots + video)                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def capture_orbit_snapshots(scene, sim, snapshots_dir, target_name, n_views,
                            center=ORBIT_CENTER, radius=ORBIT_RADIUS, height=ORBIT_HEIGHT):
    """軌跡完成後, 把 /World/CameraOrbit 環繞 center 移動 n_views 圈, 各拍一張 PNG."""
    from pxr import UsdGeom, Gf

    log(f"Capturing {n_views} orbit snapshots around {center}...")
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/CameraOrbit")
    if not cam_prim.IsValid():
        log("WARN: /World/CameraOrbit not found, skipping orbit capture.")
        return

    xform = UsdGeom.Xformable(cam_prim)
    cx, cy, cz = center
    sim_dt = sim.get_physics_dt()

    for i in range(n_views):
        theta = 2.0 * np.pi * i / n_views
        ex = cx + radius * np.cos(theta)
        ey = cy + radius * np.sin(theta)
        ez = height

        eye = Gf.Vec3d(float(ex), float(ey), float(ez))
        ctr = Gf.Vec3d(float(cx), float(cy), float(cz))
        up  = Gf.Vec3d(0.0, 0.0, 1.0)
        # SetLookAt 給 view matrix, 但相機要的是自身 transform => 取逆
        view_mat = Gf.Matrix4d().SetLookAt(eye, ctr, up)
        cam_mat  = view_mat.GetInverse()

        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(cam_mat)

        # 推較多 frame 等 RTX renderer 收斂 (太少會糊/有噪點)
        for _ in range(16):
            sim.step()
            scene.update(sim_dt)

        img = scene["camera_orbit"].data.output["rgb"][0].cpu().numpy()
        img_bgr = to_bgr(img)

        deg = int(round(np.degrees(theta)))
        out_path = snapshots_dir / f"{target_name}_orbit_{i:03d}_theta{deg:03d}.png"
        # PNG 無壓縮 (compression=0 = 最高畫質, 檔案最大)
        cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if (i + 1) % max(1, n_views // 8) == 0:
            log(f"  orbit snapshot {i+1}/{n_views}")

    log(f"Orbit snapshots saved to: {snapshots_dir}")


def capture_orbit_video(scene, sim, output_dir, target_name, n_frames, camera_fps,
                        center=ORBIT_CENTER, radius=ORBIT_RADIUS, height=ORBIT_HEIGHT,
                        warmup_steps=4):
    """環繞拍影片: 把 /World/CameraOrbit 連續移動 n_frames 個位置, 串成 mp4.

    warmup_steps 比 snapshots 版小很多 (4 vs 16) 因為要保持 fps 流暢.
    """
    from pxr import UsdGeom, Gf

    log(f"Capturing orbit video ({n_frames} frames) around {center}...")
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/CameraOrbit")
    if not cam_prim.IsValid():
        log("WARN: /World/CameraOrbit not found, skipping orbit video.")
        return

    xform = UsdGeom.Xformable(cam_prim)

    video_path = output_dir / f"{target_name}_orbit.mp4"
    h264 = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(video_path), h264, camera_fps,
                             (ORBIT_CAMERA_WIDTH, ORBIT_CAMERA_HEIGHT))
    if not writer.isOpened():
        mp4v = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), mp4v, camera_fps,
                                 (ORBIT_CAMERA_WIDTH, ORBIT_CAMERA_HEIGHT))
    if not writer.isOpened():
        log(f"WARN: Failed to open orbit video writer for {video_path}")
        return

    cx, cy, cz = center
    sim_dt = sim.get_physics_dt()

    for i in range(n_frames):
        theta = 2.0 * np.pi * i / n_frames
        ex = cx + radius * np.cos(theta)
        ey = cy + radius * np.sin(theta)
        ez = height

        eye = Gf.Vec3d(float(ex), float(ey), float(ez))
        ctr = Gf.Vec3d(float(cx), float(cy), float(cz))
        up  = Gf.Vec3d(0.0, 0.0, 1.0)
        view_mat = Gf.Matrix4d().SetLookAt(eye, ctr, up)
        cam_mat  = view_mat.GetInverse()

        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(cam_mat)

        for _ in range(warmup_steps):
            sim.step()
            scene.update(sim_dt)

        img = scene["camera_orbit"].data.output["rgb"][0].cpu().numpy()
        writer.write(to_bgr(img))

        if (i + 1) % max(1, n_frames // 10) == 0:
            log(f"  orbit video frame {i+1}/{n_frames}")

    writer.release()
    log(f"Orbit video saved to: {video_path}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 6 — Main                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    target_name = args_cli.target
    target_info = TARGETS[target_name]
    log(f"Target selected: {target_name.upper()}")
    log(f"  USD:        {target_info['usd_relative']}")
    log(f"  Spawn pos:  {target_info['spawn_pos']}")
    log(f"  Mass:       {target_info['mass']} kg")
    log(f"  grasp_z:    {target_info['grasp_z']}")
    log(f"  hover_z:    {target_info['hover_z']}")

    enable_extensions()

    camera_fps = max(1, int(args_cli.camera_fps))
    record_seconds = max(1, int(args_cli.record_seconds))
    max_video_frames = camera_fps * record_seconds
    capture_interval = 1.0 / camera_fps

    paths = build_output_paths(target_name)

    # ─────────────────────────────────────────────────────────────────
    # Phase 1 — spawn + Robot Assembler
    # ─────────────────────────────────────────────────────────────────
    log("Phase 1: spawn + Robot Assembler")
    spawn_raw_and_assemble()

    # ─────────────────────────────────────────────────────────────────
    # Phase 2 — SimulationContext + InteractiveScene
    # ─────────────────────────────────────────────────────────────────
    log("Phase 2: SimulationContext + InteractiveScene")

    log("Phase 2.1: Creating SimulationContext...")
    try:
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT))
        log("Phase 2.1: SimulationContext OK")
    except Exception as e:
        log(f"Phase 2.1 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc())
        raise

    log("Phase 2.2: Creating SceneCfg...")
    try:
        scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
        log("Phase 2.2: SceneCfg OK")
    except Exception as e:
        log(f"Phase 2.2 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc())
        raise

    log("Phase 2.3: Creating InteractiveScene...")
    try:
        scene = InteractiveScene(scene_cfg)
        log("Phase 2.3: InteractiveScene OK")
    except Exception as e:
        log(f"Phase 2.3 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc())
        raise

    # ─────────────────────────────────────────────────────────────────
    # Phase 2.5 — Attach YCB visual meshes to physics proxies
    # ─────────────────────────────────────────────────────────────────
    log("Attaching YCB visual meshes to invisible physics proxies...")
    from isaacsim.core.utils.stage import add_reference_to_stage
    import omni.client as _client
    for target_key, info in TARGETS.items():
        usd_abs = f"{ISAAC_NUCLEUS_DIR}/{info['usd_relative']}"
        visual_path = f"/World/{target_key.capitalize()}/Visuals"
        log(f"  Attaching {target_key}: {usd_abs} -> {visual_path}")
        result, _ = _client.stat(usd_abs)
        log(f"  stat result: {result}")
        try:
            add_reference_to_stage(usd_path=usd_abs, prim_path=visual_path)
            log(f"  attached OK")
        except Exception as e:
            log(f"  attach FAILED: {e}")
    log("All YCB visual meshes attached.")

    # ─────────────────────────────────────────────────────────────────
    # Phase 3 — sim.reset() + sim.play()
    # ─────────────────────────────────────────────────────────────────
    log("Phase 3.1: sim.reset()...")
    try:
        sim.reset()
        log("Phase 3.1: sim.reset() OK")
    except Exception as e:
        log(f"Phase 3.1 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc())
        raise

    log("Phase 3.2: sim.play()...")
    try:
        sim.play()
        log("Phase 3.2: sim.play() OK")
    except Exception as e:
        log(f"Phase 3.2 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc())
        raise
    sim_dt = sim.get_physics_dt()

    # 隱藏 cuboid proxy 的 mesh, 只留 YCB visual
    stage = omni.usd.get_context().get_stage()
    hide_proxy_meshes(stage, TARGETS.keys())

    # ─────────────────────────────────────────────────────────────────
    # Phase 4 — Init robot pose, joints, IK, gripper
    # ─────────────────────────────────────────────────────────────────
    robot = scene["robot"]
    device = str(sim.device)

    root_pose = torch.tensor(
        [[*ROBOT_BASE_POS, *ROBOT_BASE_ROT]],
        device=device, dtype=torch.float32,
    )
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    log(f"Camera main rgb shape:  {tuple(scene['camera_main'].data.output['rgb'].shape)}")
    log(f"Camera wrist rgb shape: {tuple(scene['camera_wrist'].data.output['rgb'].shape)}")
    log(f"Camera orbit rgb shape: {tuple(scene['camera_orbit'].data.output['rgb'].shape)}")

    out_main, out_wrist = create_video_writers(paths, camera_fps)
    log(f"Main video:  {paths.video_main_path}")
    log(f"Wrist video: {paths.video_wrist_path}")
    saved_frames = 0
    elapsed_since_capture = 0.0
    recording_done_logged = False

    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    arm_ids, _ = robot.find_joints(arm_joint_names)
    arm_ids_t = torch.tensor(arm_ids, dtype=torch.long, device=device)

    gripper_joint_ids = []
    gripper_joint_cfg = []
    for joint_name, (sign, lo, hi) in GRIPPER_MIMIC_MAP.items():
        ids, _ = robot.find_joints([joint_name])
        if ids:
            gripper_joint_ids.append(ids[0])
            gripper_joint_cfg.append((sign, lo, hi))

    if not gripper_joint_ids:
        raise RuntimeError("No gripper joints found in assembled robot articulation")

    finger_ids_t  = torch.tensor(gripper_joint_ids, dtype=torch.long, device=device)
    gripper_signs = torch.tensor([cfg[0] for cfg in gripper_joint_cfg], dtype=torch.float32, device=device)
    gripper_lows  = torch.tensor([cfg[1] for cfg in gripper_joint_cfg], dtype=torch.float32, device=device)
    gripper_highs = torch.tensor([cfg[2] for cfg in gripper_joint_cfg], dtype=torch.float32, device=device)

    # End-effector body / jacobian indices
    # Jacobian shape 是 [num_envs, num_bodies-1, 6, num_dofs] (扣掉 base)
    ee_ids, _ = robot.find_bodies([EE_BODY_NAME])
    ee_body_idx = ee_ids[0]
    body_names = robot.body_names
    ee_jac_idx = body_names.index(EE_BODY_NAME) - 1
    log(f"EE body '{EE_BODY_NAME}': body_idx={ee_body_idx}, jac_idx={ee_jac_idx}")

    home_q_t = torch.tensor([HOME_Q], device=device)
    zero_q_t = torch.zeros_like(home_q_t)
    robot.write_joint_state_to_sim(
        position=home_q_t, velocity=zero_q_t, joint_ids=arm_ids_t,
    )
    robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
    open_cmd = torch.clamp(
        (gripper_signs * GRIPPER_OPEN).unsqueeze(0),
        gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
    )
    robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)

    log("Settling 180 steps for objects to rest on table...")
    for _ in range(180):
        robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
        robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    banana_pos = scene["banana"].data.root_pos_w[0].cpu().numpy().round(3).tolist()
    mug_pos    = scene["mug"   ].data.root_pos_w[0].cpu().numpy().round(3).tolist()
    log(f"Banana resting at: {banana_pos}")
    log(f"Mug    resting at: {mug_pos}")

    target_obj = scene[target_name]
    target_resting = target_obj.data.root_pos_w[0].cpu().numpy().tolist()
    log(f"-> Target '{target_name}' resting at: "
        f"[{target_resting[0]:.3f}, {target_resting[1]:.3f}, {target_resting[2]:.3f}]")

    tx, ty, tz = target_resting
    tx_adj = tx
    ty_adj = ty + target_info["y_nudge"]
    hover_z = tz + target_info["hover_z"]
    grasp_z = tz + target_info["grasp_z"]

    HOVER_POS = (tx_adj, ty_adj, hover_z)
    GRASP_POS = (tx_adj, ty_adj, grasp_z)
    log(f"HOVER_POS = ({HOVER_POS[0]:.3f}, {HOVER_POS[1]:.3f}, {HOVER_POS[2]:.3f})")
    log(f"GRASP_POS = ({GRASP_POS[0]:.3f}, {GRASP_POS[1]:.3f}, {GRASP_POS[2]:.3f})")

    # 7-segment grasp trajectory
    TRAJECTORY = [
        (3.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_OPEN),
        (4.0, HOVER_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (3.0, GRASP_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (2.0, GRASP_POS, EE_ORIENT_DOWN, GRIPPER_CLOSE),
        (3.0, HOVER_POS, EE_ORIENT_DOWN, GRIPPER_CLOSE),
        (3.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_CLOSE),
        (2.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_OPEN),
    ]

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls",
    )
    ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)
    player = PoseTrajectoryPlayer(TRAJECTORY, device=device)

    def phase_name(t):
        if t <  3:    return "HOME-OPEN"
        if t <  7:    return "HOVER"
        if t < 10:    return "DESCEND"
        if t < 12:    return "CLOSE"
        if t < 15:    return "LIFT"
        if t < 18:    return "HOME-CLOSED"
        if t < 20:    return "RELEASE"
        return "DONE"

    # ─────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────
    t = 0.0
    step = 0
    finished_logged = False
    last_grip_target = GRIPPER_OPEN
    orbit_done = False
    log("Entering main loop. Ctrl+C to stop.")

    try:
        while app.is_running():
            tgt_pos_w, tgt_quat_w, grip_target, finished = player.sample(t)

            if finished and not finished_logged:
                log("=" * 70)
                log("TRAJECTORY FINISHED -- holding final pose.")
                target_final = target_obj.data.root_pos_w[0].cpu().numpy().tolist()
                log(f"{target_name.capitalize()} final:   "
                    f"[{target_final[0]:.3f}, {target_final[1]:.3f}, {target_final[2]:.3f}]")
                log(f"{target_name.capitalize()} initial: "
                    f"[{target_resting[0]:.3f}, {target_resting[1]:.3f}, {target_resting[2]:.3f}]")
                dx = target_final[0] - target_resting[0]
                dy = target_final[1] - target_resting[1]
                dz = target_final[2] - target_resting[2]
                log(f"Displacement: dx={dx:+.3f}  dy={dy:+.3f}  dz={dz:+.3f}")
                log("=" * 70)
                finished_logged = True

            # ── IK: pose target -> joint position target ──────────────
            root = robot.data.root_state_w[:, :7]
            root_pos = root[:, :3]

            tgt_pos_b  = tgt_pos_w.unsqueeze(0) - root_pos
            tgt_quat_b = tgt_quat_w.unsqueeze(0)
            ik_cmd = torch.cat([tgt_pos_b, tgt_quat_b], dim=-1)
            ik.set_command(ik_cmd)

            ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
            ee_pos_b  = ee_pose_w[:, :3] - root_pos
            ee_quat_b = ee_pose_w[:, 3:]

            q_current = robot.data.joint_pos[:, arm_ids_t]

            jac_full = robot.root_physx_view.get_jacobians()
            jac = jac_full[:, ee_jac_idx, :, :]
            jac = jac[:, :, arm_ids_t]

            q_target = ik.compute(ee_pos_b, ee_quat_b, jac, q_current)
            robot.set_joint_position_target(q_target, joint_ids=arm_ids_t)

            # ── Smooth scalar gripper command ─────────────────────────
            max_step = 0.008
            delta = grip_target - last_grip_target
            if delta > max_step:
                grip_target = last_grip_target + max_step
            elif delta < -max_step:
                grip_target = last_grip_target - max_step
            grip_target = float(np.clip(grip_target, GRIPPER_OPEN, GRIPPER_CLOSE))
            last_grip_target = grip_target

            finger_cmd = torch.clamp(
                (gripper_signs * grip_target).unsqueeze(0),
                gripper_lows.unsqueeze(0),
                gripper_highs.unsqueeze(0),
            )
            robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids_t)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # ── Recording: mp4 + per-frame PNG ────────────────────────
            if saved_frames < max_video_frames:
                elapsed_since_capture += sim_dt
                if elapsed_since_capture >= capture_interval:
                    elapsed_since_capture -= capture_interval

                    img_main = scene["camera_main"].data.output["rgb"][0].cpu().numpy()
                    img_main_bgr = to_bgr(img_main)
                    out_main.write(img_main_bgr)
                    cv2.imwrite(str(paths.frames_main_dir / f"frame_{saved_frames:05d}.png"), img_main_bgr)

                    img_wrist = scene["camera_wrist"].data.output["rgb"][0].cpu().numpy()
                    img_wrist_bgr = to_bgr(img_wrist)
                    out_wrist.write(img_wrist_bgr)
                    cv2.imwrite(str(paths.frames_wrist_dir / f"frame_{saved_frames:05d}.png"), img_wrist_bgr)

                    saved_frames += 1
                    if saved_frames % camera_fps == 0:
                        log(f"Frames: {saved_frames}/{max_video_frames}")

                    # ── Recording 完成: 收尾 + 拍 orbit ────────────────
                    if saved_frames >= max_video_frames:
                        out_main.release()
                        out_wrist.release()
                        if not recording_done_logged:
                            log("Recording completed.")
                            recording_done_logged = True

                            if not orbit_done:
                                if args_cli.orbit_snapshots > 0:
                                    capture_orbit_snapshots(
                                        scene, sim, paths.snapshots_dir, target_name,
                                        n_views=args_cli.orbit_snapshots,
                                    )
                                if args_cli.orbit_video_frames > 0:
                                    capture_orbit_video(
                                        scene, sim, paths.output_dir, target_name,
                                        n_frames=args_cli.orbit_video_frames,
                                        camera_fps=camera_fps,
                                    )
                                orbit_done = True
                            log("Capture work done. Continuing simulation (Ctrl+C to exit)...")

            t += sim_dt
            step += 1

            # ── Periodic status log (1 Hz) ────────────────────────────
            if step % 60 == 0:
                ee_now = ee_pose_w[0, :3].cpu().numpy().round(3).tolist()
                d_tgt  = torch.linalg.norm(ee_pose_w[0, :3] - tgt_pos_w).item()
                obj_np = target_obj.data.root_pos_w[0].cpu().numpy().round(3).tolist()
                main_finger_q = robot.data.joint_pos[0, finger_ids_t[0]].item()
                log(f"t={t:5.2f}s [{phase_name(t):<11}] "
                    f"ee={ee_now}  d_tgt={d_tgt:.3f}  {target_name}={obj_np}  "
                    f"finger={main_finger_q:.2f}(->{grip_target:.2f})")

    except KeyboardInterrupt:
        log("Ctrl+C received.")
    finally:
        sim.stop()
        if out_main is not None and out_main.isOpened():
            out_main.release()
        if out_wrist is not None and out_wrist.isOpened():
            out_wrist.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()