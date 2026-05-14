"""
sim_vla.py — OpenVLA 控制 sim 內 UR3e 手臂.

⚠️  跑在 env_isaaclab_ros2, 不是 env_openvla.
    vla_server.py 要先在另一個 terminal 跑起來.

Usage:
    # Terminal A: 起 VLA server (env_openvla)
    conda activate env_openvla
    python vla_server.py --model-path ~/models/openvla-7b

    # Terminal B: 跑 sim (env_isaaclab_ros2)
    conda activate env_isaaclab_ros2
    source /opt/ros/jazzy/setup.bash
    ./isaaclab.sh -p sim_vla.py \\
        --target mug \\
        --instruction "pick up the red mug" \\
        --vla-server http://localhost:8000

架構:
    - VLA 每 VLA_STEP_INTERVAL 個 sim step 推論一次 (預設 12 steps = 5Hz @ 60Hz sim)
    - VLA 輸出 7-DoF delta action: [Δx, Δy, Δz, Δr, Δp, Δyaw, gripper]
    - Delta 累積到 EE target pose, 用 IK 換算成 joint targets
    - 介於 VLA call 的中間 steps 用線性插值補上 (smooth motion)

VLA 使用注意:
    - out-of-box OpenVLA 不會抓你的 UR3e (訓練分布不對), 這是預期
    - 這個 script 的目標是「toolchain 通」, 不是「能成功抓」
    - 要能抓需要 fine-tune (蒐集 sim demo -> LoRA fine-tune -> 這個 script 載 fine-tuned)
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 1 — Boot                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

import sys
import argparse

# 先加 --instruction + --vla-server 到 sys.argv 再 boot
# (避免 AppLauncher 抱怨 unknown args)
_extra = argparse.ArgumentParser(add_help=False)
_extra.add_argument("--instruction", type=str, default="pick up the object")
_extra.add_argument("--vla-server",  type=str, default="http://localhost:8000")
_extra.add_argument("--vla-step-interval", type=int, default=12,
                    help="每幾個 sim steps 呼叫一次 VLA (12 = 5Hz @ 60Hz sim)")
_extra.add_argument("--action-scale", type=float, default=0.5,
                    help="VLA delta action 的縮放倍率 (太快就調小)")
_extra.add_argument("--max-steps", type=int, default=600,
                    help="最多跑幾步 (0 = 無限制)")
_extra_args, _ = _extra.parse_known_args()

from envs.boot import boot_app, args_cli, log
app = boot_app()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 2 — Imports                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

import base64
import io
import time
import threading

import cv2
import numpy as np
import torch
import requests
from PIL import Image

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from envs.config import (
    PHYSICS_DT,
    ROBOT_BASE_POS, ROBOT_BASE_ROT,
    EE_BODY_NAME,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    HOME_Q,
    TARGETS,
    GRIPPER_MIMIC_MAP,
    WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z,
)
from envs.sim_setup import (
    enable_extensions,
    spawn_raw_and_assemble,
    SceneCfg,
    hide_proxy_meshes,
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 3 — VLA Client                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

class VLAClient:
    """HTTP client for vla_server.py."""

    def __init__(self, server_url: str, timeout: float = 2.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._last_action = np.zeros(7, dtype=np.float32)
        self._request_count = 0
        self._error_count = 0

    def health_check(self) -> bool:
        """確認 server 活著. 啟動時用."""
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5.0)
            data = r.json()
            log(f"[VLA] Server status: {data['status']}")
            log(f"[VLA] Model: {data.get('model_path', '?')}")
            log(f"[VLA] Quantization: {data.get('quantization', '?')}")
            return data["status"] == "ok"
        except Exception as e:
            log(f"[VLA] Health check failed: {e}")
            return False

    def predict(self, rgb: np.ndarray, instruction: str,
                unnorm_key: str = "bridge_orig") -> np.ndarray:
        """送出 rgb image, 回傳 7-DoF action.

        Args:
            rgb: (H, W, 3) uint8 numpy array
            instruction: 自然語言指令
            unnorm_key: action unnormalization key

        Returns:
            action: (7,) float32 [Δx, Δy, Δz, Δr, Δp, Δyaw, gripper]
            如果推論失敗, 回傳 zeros (fail-safe)
        """
        self._request_count += 1
        t0 = time.monotonic()

        try:
            # 轉成 base64 JPEG
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            r = requests.post(
                f"{self.server_url}/predict",
                json={
                    "image_b64": img_b64,
                    "instruction": instruction,
                    "unnorm_key": unnorm_key,
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()

            action = np.array(data["action"], dtype=np.float32)
            self._last_action = action
            latency = data.get("latency_ms", (time.monotonic() - t0) * 1000)

            if self._request_count % 10 == 0:
                log(f"[VLA] #{self._request_count}: action={action.round(3).tolist()}"
                    f" latency={latency:.0f}ms err={self._error_count}")

            return action

        except Exception as e:
            self._error_count += 1
            if self._error_count <= 5 or self._error_count % 20 == 0:
                log(f"[VLA] predict failed (#{self._error_count}): {e}")
            # fail-safe: 回傳 last known action 或 zeros
            return self._last_action.copy()

    @property
    def stats(self):
        return {
            "requests": self._request_count,
            "errors": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
        }


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 4 — Action 應用 helper                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def apply_delta_action(
    ee_pos_current: torch.Tensor,     # (3,) 當下 EE 位置 (world frame)
    ee_quat_current: torch.Tensor,    # (4,) 當下 EE quaternion (w,x,y,z)
    action: np.ndarray,               # (7,) VLA 輸出
    action_scale: float,
    workspace_x: tuple,
    workspace_y: tuple,
    workspace_z: tuple,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """把 VLA 7-DoF delta action 加到當下 EE pose, 回傳新的 target pose.

    OpenVLA action format: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    - 前 6 個是 EE delta (base frame)
    - gripper: 0.0 = open, 1.0 = close

    Returns:
        target_pos:  (3,) 新的 EE 目標位置
        target_quat: (4,) 新的 EE 目標姿態 quaternion
        gripper_cmd: float, 0.0-1.0
    """
    device = ee_pos_current.device
    action_t = torch.tensor(action, dtype=torch.float32, device=device)

    # ── Delta translation ─────────────────────────────────────────
    delta_pos = action_t[:3] * action_scale
    target_pos = ee_pos_current + delta_pos

    # Clamp 到 workspace 邊界 (防止手臂飛出去)
    target_pos[0] = target_pos[0].clamp(workspace_x[0], workspace_x[1])
    target_pos[1] = target_pos[1].clamp(workspace_y[0], workspace_y[1])
    target_pos[2] = target_pos[2].clamp(workspace_z[0], workspace_z[1])

    # ── Delta rotation (euler ZYX -> quaternion delta) ────────────
    # 先用小角近似 (delta 很小時夠精確)
    delta_euler = action_t[3:6] * action_scale  # [Δr, Δp, Δy]

    # Convert euler delta to quaternion
    half = delta_euler * 0.5
    cos_h = torch.cos(half)
    sin_h = torch.sin(half)
    # ZYX euler to quat (w, x, y, z)
    dq_w = cos_h[0] * cos_h[1] * cos_h[2] + sin_h[0] * sin_h[1] * sin_h[2]
    dq_x = sin_h[0] * cos_h[1] * cos_h[2] - cos_h[0] * sin_h[1] * sin_h[2]
    dq_y = cos_h[0] * sin_h[1] * cos_h[2] + sin_h[0] * cos_h[1] * sin_h[2]
    dq_z = cos_h[0] * cos_h[1] * sin_h[2] - sin_h[0] * sin_h[1] * cos_h[2]
    delta_quat = torch.stack([dq_w, dq_x, dq_y, dq_z])

    # Quaternion multiplication: target = current * delta
    w0, x0, y0, z0 = ee_quat_current
    w1, x1, y1, z1 = delta_quat
    target_quat = torch.stack([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])
    target_quat = target_quat / torch.linalg.norm(target_quat)

    # ── Gripper ────────────────────────────────────────────────────
    gripper_cmd = float(action_t[6].clamp(0.0, 1.0))

    return target_pos, target_quat, gripper_cmd


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 5 — Main                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    target_name  = args_cli.target
    instruction  = _extra_args.instruction
    vla_server   = _extra_args.vla_server
    step_interval = _extra_args.vla_step_interval
    action_scale = _extra_args.action_scale
    max_steps    = _extra_args.max_steps

    log(f"Target: {target_name}")
    log(f"Instruction: '{instruction}'")
    log(f"VLA server: {vla_server}")
    log(f"VLA call every {step_interval} sim steps "
        f"(= {60/step_interval:.1f} Hz at 60Hz sim)")
    log(f"Action scale: {action_scale}")

    # ── (0) 確認 VLA server ────────────────────────────────────────
    vla = VLAClient(vla_server)
    log("[VLA] Checking server health...")
    if not vla.health_check():
        log("[VLA] WARNING: server not ready. 繼續跑但 VLA call 會 fallback 到 zeros.")
        log("[VLA] 確認 vla_server.py 有在跑: python vla_server.py --model-path ~/models/openvla-7b")
    else:
        log("[VLA] Server OK, model loaded.")

    enable_extensions()

    # ── Phase 1-3: 跟 sim_r.py 完全一樣 ─────────────────────────────
    log("Phase 1: spawn + Robot Assembler")
    spawn_raw_and_assemble()

    log("Phase 2: SimulationContext + InteractiveScene")
    log("Phase 2.1: Creating SimulationContext...")
    try:
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT))
        log("Phase 2.1: SimulationContext OK")
    except Exception as e:
        log(f"Phase 2.1 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc()); raise

    log("Phase 2.2: Creating SceneCfg...")
    try:
        scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
        log("Phase 2.2: SceneCfg OK")
    except Exception as e:
        log(f"Phase 2.2 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc()); raise

    log("Phase 2.3: Creating InteractiveScene...")
    try:
        scene = InteractiveScene(scene_cfg)
        log("Phase 2.3: InteractiveScene OK")
    except Exception as e:
        log(f"Phase 2.3 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc()); raise

    # Attach YCB visual mesh
    log("Attaching YCB visual meshes...")
    from isaacsim.core.utils.stage import add_reference_to_stage
    import omni.client as _client
    for target_key, info in TARGETS.items():
        usd_abs = f"{ISAAC_NUCLEUS_DIR}/{info['usd_relative']}"
        visual_path = f"/World/{target_key.capitalize()}/Visuals"
        result, _ = _client.stat(usd_abs)
        log(f"  {target_key}: stat={result}")
        try:
            add_reference_to_stage(usd_path=usd_abs, prim_path=visual_path)
        except Exception as e:
            log(f"  attach FAILED ({target_key}): {e}")

    log("Phase 3.1: sim.reset()...")
    try:
        sim.reset()
        log("Phase 3.1: sim.reset() OK")
    except Exception as e:
        log(f"Phase 3.1 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc()); raise

    log("Phase 3.2: sim.play()...")
    try:
        sim.play()
        log("Phase 3.2: sim.play() OK")
    except Exception as e:
        log(f"Phase 3.2 FAILED: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc()); raise

    sim_dt = sim.get_physics_dt()
    stage = omni.usd.get_context().get_stage()
    hide_proxy_meshes(stage, TARGETS.keys())

    # ── Phase 4: Robot init ─────────────────────────────────────────
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

    finger_ids_t  = torch.tensor(gripper_joint_ids, dtype=torch.long, device=device)
    gripper_signs = torch.tensor([c[0] for c in gripper_joint_cfg], dtype=torch.float32, device=device)
    gripper_lows  = torch.tensor([c[1] for c in gripper_joint_cfg], dtype=torch.float32, device=device)
    gripper_highs = torch.tensor([c[2] for c in gripper_joint_cfg], dtype=torch.float32, device=device)

    ee_ids, _ = robot.find_bodies([EE_BODY_NAME])
    ee_body_idx = ee_ids[0]
    body_names = robot.body_names
    ee_jac_idx = body_names.index(EE_BODY_NAME) - 1

    home_q_t = torch.tensor([HOME_Q], device=device, dtype=torch.float32)
    robot.write_joint_state_to_sim(
        position=home_q_t, velocity=torch.zeros_like(home_q_t), joint_ids=arm_ids_t,
    )

    open_cmd = torch.clamp(
        (gripper_signs * GRIPPER_OPEN).unsqueeze(0),
        gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
    )
    robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
    robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)

    log("Settling 180 steps...")
    for _ in range(180):
        robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
        robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # IK controller
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls",
    )
    ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    # 取出 workspace 邊界
    ws_x = getattr(__import__('envs.config', fromlist=['WORKSPACE_X']), 'WORKSPACE_X', (-0.4, 0.7))
    ws_y = getattr(__import__('envs.config', fromlist=['WORKSPACE_Y']), 'WORKSPACE_Y', (-0.5, 0.5))
    ws_z = getattr(__import__('envs.config', fromlist=['WORKSPACE_Z']), 'WORKSPACE_Z', (1.05, 1.65))

    # VLA target pose (初始值 = 當下 EE)
    ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
    ee_target_pos  = ee_pose_w[0, :3].clone()
    ee_target_quat = ee_pose_w[0, 3:7].clone()
    gripper_target = GRIPPER_OPEN

    # ── Main loop ─────────────────────────────────────────────────
    log("Entering main loop. Ctrl+C to stop.")
    step = 0
    last_log_t = time.monotonic()
    last_vla_action = np.zeros(7, dtype=np.float32)

    try:
        while app.is_running():
            # ── VLA call (每 step_interval 步一次) ─────────────────
            if step % step_interval == 0:
                # 從 wrist camera 拿圖
                cam_data = scene["camera_wrist"].data.output.get("rgb")
                if cam_data is not None:
                    rgb_np = cam_data[0].cpu().numpy().astype(np.uint8)

                    # 背景 thread 跑推論 (非同步, 不卡主迴圈)
                    def _async_predict():
                        nonlocal last_vla_action
                        action = vla.predict(rgb_np, instruction)
                        last_vla_action = action

                    t = threading.Thread(target=_async_predict, daemon=True)
                    t.start()

            # ── 把 VLA action 應用到 EE target ──────────────────
            # 用上一次 VLA 的 action (可能比當前 step 舊, 但不阻塞)
            if step % step_interval == 0 and not np.all(last_vla_action == 0):
                new_pos, new_quat, new_grip = apply_delta_action(
                    ee_target_pos, ee_target_quat,
                    last_vla_action, action_scale,
                    ws_x, ws_y, ws_z,
                )
                ee_target_pos  = new_pos
                ee_target_quat = new_quat
                gripper_target = new_grip

            # ── IK ─────────────────────────────────────────────
            root_pos = robot.data.root_state_w[:, :3]
            ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
            ee_pos_b  = ee_pose_w[:, :3] - root_pos
            ee_quat_b = ee_pose_w[:, 3:]

            tgt_pos_b  = ee_target_pos.unsqueeze(0) - root_pos
            tgt_quat_b = ee_target_quat.unsqueeze(0)
            ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))

            q_current = robot.data.joint_pos[:, arm_ids_t]
            jac_full = robot.root_physx_view.get_jacobians()
            jac = jac_full[:, ee_jac_idx, :, :][:, :, arm_ids_t]

            q_target = ik.compute(ee_pos_b, ee_quat_b, jac, q_current)
            robot.set_joint_position_target(q_target, joint_ids=arm_ids_t)

            # ── Gripper ────────────────────────────────────────
            GRIP_SMOOTH = 0.08
            raw_grip_binary = 1.0 if gripper_target > 0.7 else 0.0
            gripper_target = gripper_target + GRIP_SMOOTH * (raw_grip_binary - gripper_target)
            grip_scaled = gripper_target * (GRIPPER_CLOSE - GRIPPER_OPEN) + GRIPPER_OPEN
            finger_cmd = torch.clamp(
                (gripper_signs * grip_scaled).unsqueeze(0),
                gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
            )
            robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids_t)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # ── Status log (每 1 秒) ────────────────────────────
            now = time.monotonic()
            if now - last_log_t >= 1.0:
                ee_now = ee_pose_w[0, :3].cpu().numpy().round(3).tolist()
                tgt_now = ee_target_pos.cpu().numpy().round(3).tolist()
                obj = scene[target_name].data.root_pos_w[0].cpu().numpy().round(3).tolist()
                log(f"step={step} ee={ee_now} tgt={tgt_now} "
                    f"grip={gripper_target:.2f} {target_name}={obj} "
                    f"vla_req={vla.stats['requests']} err={vla.stats['errors']}")
                last_log_t = now

            step += 1
            if max_steps > 0 and step >= max_steps:
                log(f"Reached max_steps={max_steps}, stopping.")
                break

    except KeyboardInterrupt:
        log("Ctrl+C received.")
    finally:
        log(f"VLA stats: {vla.stats}")
        log("Simulation done. Press Ctrl+C to close window.")
        while app.is_running():
            sim.step() 


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()