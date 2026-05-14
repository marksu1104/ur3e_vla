"""
sim_collect.py — 蒐集 successful grasp demonstrations 給 OpenVLA fine-tune.

基於 sim_r.py 的 hardcoded trajectory, 加上:
  - Episode loop (跑完一條 reset 再跑下一條)
  - Randomization (物件位置、旋轉、光照)
  - Success detection (filter 失敗的 episode)
  - Demo logging (每 step 存 image + ee_pose + action_7d)

Usage:
    # 先試 5 條看 demo 品質
    ./isaaclab.sh -p sim_collect.py --target mug --episodes 5 --output-dir raw/test/

    # 確認 OK 後跑大量
    ./isaaclab.sh -p sim_collect.py --target mug --episodes 500 --output-dir raw/run0/

Output 結構:
    raw/run0/
    ├── ep_00000/
    │   ├── meta.json            episode 元資料
    │   ├── main_frames/         主相機 jpg per step
    │   │   ├── 00000.jpg
    │   │   └── ...
    │   ├── wrist_frames/        腕部相機 jpg per step
    │   │   └── ...
    │   ├── ee_poses.npz         (T, 7) [x,y,z, qw,qx,qy,qz]
    │   ├── joint_positions.npz  (T, 6) UR3e joint angles
    │   └── actions.npz          (T, 7) 7D delta actions (我們學的目標)
    ├── ep_00001/
    └── ...

⚠️  跟 sim_r.py 共用 envs/, 沒有重複定義場景常數.
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 1 — Boot                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

import argparse
import sys

# 加 --episodes / --output-dir 進 sys.argv 之前先 parse
_extra = argparse.ArgumentParser(add_help=False)
_extra.add_argument("--episodes", type=int, default=5,
                    help="蒐集多少條 successful episodes")
_extra.add_argument("--output-dir", type=str, default="raw/run0",
                    help="輸出目錄 (relative to ~/IsaacLab)")
_extra.add_argument("--max-episodes-tried", type=int, default=0,
                    help="最多嘗試幾條 (0 = episodes * 2)")
_extra.add_argument("--randomize-pos", action=argparse.BooleanOptionalAction, default=True,
                    help="隨機化物件位置")
_extra.add_argument("--randomize-rot", action=argparse.BooleanOptionalAction, default=True,
                    help="隨機化物件旋轉（保留物件原始 spawn_rot，再額外加 yaw）")
_extra.add_argument("--randomize-light", action=argparse.BooleanOptionalAction, default=True,
                    help="隨機化光照")
_extra.add_argument("--keep-sim-alive", action=argparse.BooleanOptionalAction, default=True,
                    help="蒐集完成後保持 simulation 繼續跑, 直到 Ctrl+C")
_extra.add_argument("--seed", type=int, default=42)
_extra_args, _ = _extra.parse_known_args()

# 強制設成 headless 蒐 demo (不需要 GUI, 速度更快)
# 如果你想看著蒐, 移除這行
if "--headless" not in sys.argv:
    sys.argv.append("--headless")

from envs.boot import boot_app, args_cli, log
app = boot_app()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 2 — Imports                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict

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
    ROBOT_BASE_POS, ROBOT_BASE_ROT,
    EE_BODY_NAME, EE_ORIENT_DOWN,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    HOME_POS, HOME_Q,
    TARGETS,
    GRIPPER_MIMIC_MAP,
)
from envs.sim_setup import (
    enable_extensions,
    spawn_raw_and_assemble,
    SceneCfg,
    hide_proxy_meshes,
)
from envs.control_api import compute_action_from_ee_poses


# 設 seed (numpy + python random, sim 內 PhysX 仍會有 stochastic 行為)
random.seed(_extra_args.seed)
np.random.seed(_extra_args.seed)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 3 — Randomization helpers                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def quat_mul(q1, q2):
    """Hamilton product for quaternions in IsaacLab wxyz order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def randomize_target_pose(target_info: dict, rng: np.random.Generator) -> tuple:
    """回傳隨機化的 (spawn_pos, spawn_rot).

    Position 在原始 spawn_pos 周圍 ±10cm 內 random.
    Rotation: 保留原始 spawn_rot，再繞 world Z 軸加 random yaw.
    """
    base_x, base_y, base_z = target_info["spawn_pos"]
    base_rot = target_info.get("spawn_rot", (1.0, 0.0, 0.0, 0.0))

    if _extra_args.randomize_pos:
        rand_x = base_x + rng.uniform(-0.08, 0.08)
        rand_y = base_y + rng.uniform(-0.10, 0.10)
    else:
        rand_x, rand_y = base_x, base_y

    if _extra_args.randomize_rot:
        yaw = rng.uniform(-np.pi, np.pi)
        # Yaw to quaternion (繞 Z 軸): [cos(y/2), 0, 0, sin(y/2)]
        yaw_rot = (np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2))
        spawn_rot = quat_mul(yaw_rot, base_rot)
    else:
        spawn_rot = base_rot

    return (rand_x, rand_y, base_z), spawn_rot


def randomize_lighting(stage, rng: np.random.Generator):
    """隨機化 dome light 強度跟色溫."""
    if not _extra_args.randomize_light:
        return
    try:
        from pxr import UsdLux
        light_prim = stage.GetPrimAtPath("/World/light")
        if not light_prim.IsValid():
            return
        light = UsdLux.DomeLight(light_prim)
        intensity = 3000.0 * rng.uniform(0.7, 1.3)
        light.GetIntensityAttr().Set(float(intensity))
        # 色溫: warm <-> cool
        if rng.random() < 0.5:
            color = (1.0, 0.95, 0.85)  # warm
        else:
            color = (0.85, 0.95, 1.0)  # cool
        light.GetColorAttr().Set(color)
    except Exception as e:
        log(f"randomize_lighting failed: {e}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 4 — Success detection                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def detect_success(
    target_obj, ee_pos: np.ndarray, target_initial_z: float,
    gripper_q: float,
) -> tuple[bool, dict]:
    """檢查 episode 是否成功.

    成功條件:
      1. 物件高度比起始位置高 > 5cm (確認被舉起)
      2. EE 還在合理工作空間
      3. Gripper 是 closed 狀態 (q > 0.3)
      4. 物件在 EE 附近 (< 20cm, 確認真的抓在手上)
    """
    obj_pos = target_obj.data.root_pos_w[0].cpu().numpy()

    obj_lifted     = (obj_pos[2] - target_initial_z) > 0.05
    gripper_closed = gripper_q > 0.3
    ee_safe        = (1.05 < ee_pos[2] < 1.7) and (-0.5 < ee_pos[0] < 0.7)
    obj_near_ee    = np.linalg.norm(obj_pos - ee_pos) < 0.20

    success = obj_lifted and gripper_closed and ee_safe and obj_near_ee
    return success, {
        "obj_lifted": obj_lifted,
        "gripper_closed": gripper_closed,
        "ee_safe": ee_safe,
        "obj_near_ee": obj_near_ee,
        "obj_pos_final": obj_pos.tolist(),
        "ee_pos_final": ee_pos.tolist(),
        "obj_lift_height": float(obj_pos[2] - target_initial_z),
    }


def detect_target_reached(
    ee_pos: np.ndarray,
    target_pos: tuple,
    target_initial_z: float,
    threshold: float = 0.08,
) -> tuple[bool, dict]:
    """只看 EE 是否到達指定目標點, 不依賴物件 root pose."""
    target_pos_np = np.asarray(target_pos, dtype=np.float32)
    ee_target_dist = float(np.linalg.norm(ee_pos - target_pos_np))
    target_reached = bool(ee_target_dist < threshold)
    ee_safe = bool((1.05 < ee_pos[2] < 1.7) and (-0.5 < ee_pos[0] < 0.7))

    return target_reached, {
        "target_reached": target_reached,
        "ee_target_dist": ee_target_dist,
        "target_pos": target_pos_np.tolist(),
        "ee_pos_final": ee_pos.tolist(),
        "obj_lifted": False,
        "gripper_closed": True,
        "ee_safe": ee_safe,
        "obj_near_ee": False,
        "obj_lift_height": 0.0,
        "target_initial_z": float(target_initial_z),
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 5 — Trajectory player (跟 sim_r.py 同一個)                 ║
# ╚══════════════════════════════════════════════════════════════════╝

class PoseTrajectoryPlayer:
    """線性插值 7-segment 軌跡, 跟 sim_r.py 內的完全相同."""

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
# ║  Section 6 — Demo recorder (per-episode 寫檔)                       ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class EpisodeBuffer:
    """累積一條 episode 的所有資料, 結束時一次性寫檔."""
    main_images: list   # list of (H, W, 3) uint8
    wrist_images: list
    ee_poses: list      # list of [x, y, z, qw, qx, qy, qz]
    joint_positions: list  # list of [q0..q5]
    gripper_states: list   # list of float (current finger_joint position)
    actions_7d: list       # list of [dx,dy,dz,droll,dpitch,dyaw,grip]
    timestamps: list       # list of float (sim time)


def to_jsonable(value):
    """Convert numpy/torch-ish scalars and arrays into plain JSON values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_episode(
    buffer: EpisodeBuffer,
    episode_base: Path,
    meta: dict,
):
    """Save one episode as training .npy + compact verification files."""
    episode_base.parent.mkdir(parents=True, exist_ok=True)

    episode_list = []
    num_steps = len(buffer.ee_poses)
    for i in range(num_steps):
        ee_pose = np.asarray(buffer.ee_poses[i], dtype=np.float32)
        joint_pos = np.asarray(buffer.joint_positions[i], dtype=np.float32)
        grip_binary = 1.0 if float(buffer.gripper_states[i]) > 0.2 else 0.0
        is_last = i == num_steps - 1

        state = np.concatenate([
            joint_pos[:6],
            ee_pose[:3],
            ee_pose[3:7],
            np.array([grip_binary, 0.0], dtype=np.float32),
        ]).astype(np.float32)
        action = np.concatenate([
            np.asarray(buffer.actions_7d[i], dtype=np.float32),
            np.array([1.0 if is_last else 0.0], dtype=np.float32),
        ]).astype(np.float32)

        episode_list.append({
            "image": np.asarray(buffer.main_images[i], dtype=np.uint8),
            "hand_image": np.asarray(buffer.wrist_images[i], dtype=np.uint8),
            "state": state,
            "action": action,
            "language_instruction": str(meta["instruction"]),
            "is_first": bool(i == 0),
            "is_last": bool(is_last),
            "is_terminal": bool(is_last),
        })

    np.save(str(episode_base.with_suffix(".npy")), episode_list, allow_pickle=True)

    verify_meta = {
        **meta,
        "schema": "list_of_step_dicts_npy",
        "files": {
            "episode": episode_base.with_suffix(".npy").name,
            "verification_json": episode_base.with_suffix(".json").name,
            "final_image": f"{episode_base.name}_final.jpg",
        },
        "sample_shapes": {
            "image": list(episode_list[0]["image"].shape) if episode_list else [],
            "hand_image": list(episode_list[0]["hand_image"].shape) if episode_list else [],
            "state": list(episode_list[0]["state"].shape) if episode_list else [],
            "action": list(episode_list[0]["action"].shape) if episode_list else [],
        },
    }
    with open(episode_base.with_suffix(".json"), "w") as f:
        json.dump(to_jsonable(verify_meta), f, indent=2)

    if buffer.main_images:
        final_img_bgr = cv2.cvtColor(buffer.main_images[-1], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(episode_base.parent / f"{episode_base.name}_final.jpg"),
                    final_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 7 — Single-episode runner                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_one_episode(
    sim, scene, robot, ik, sim_dt,
    arm_ids_t, finger_ids_t,
    gripper_signs, gripper_lows, gripper_highs,
    ee_body_idx, ee_jac_idx,
    target_name: str,
    target_info: dict,
    target_obj,
    rng: np.random.Generator,
    instruction: str,
    record_every_n_steps: int = 12,    # 12 sim steps @ 60Hz = 5Hz logging
) -> tuple[bool, EpisodeBuffer | None, dict]:
    """跑一條 episode, 回傳 (success, buffer or None, diagnostics).

    Returns:
        success: 是否成功
        buffer:  如果成功, 回傳資料; 失敗回傳 None (省記憶體)
        diag:    success detection 細節
    """
    device = str(sim.device)

    # ── Randomize target pose ──────────────────────────────────────
    spawn_pos, spawn_rot = randomize_target_pose(target_info, rng)

    # 把物件搬到新位置
    new_state = torch.tensor(
        [[*spawn_pos, *spawn_rot, 0, 0, 0, 0, 0, 0]],
        device=device, dtype=torch.float32,
    )  # (1, 13) = pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
    target_obj.write_root_pose_to_sim(new_state[:, :7])
    target_obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

    # 把手臂送回 home pose
    home_q_t = torch.tensor([HOME_Q], device=device, dtype=torch.float32)
    robot.write_joint_state_to_sim(
        position=home_q_t, velocity=torch.zeros_like(home_q_t), joint_ids=arm_ids_t,
    )
    open_cmd = torch.clamp(
        (gripper_signs * GRIPPER_OPEN).unsqueeze(0),
        gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
    )
    robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)

    # ── Randomize lighting ─────────────────────────────────────────
    stage = omni.usd.get_context().get_stage()
    randomize_lighting(stage, rng)

    # ── Settle: 等物件落地 ─────────────────────────────────────────
    for _ in range(60):
        robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
        robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # ── 讀取物件 settle 後的位置, 計算 grasp target ───────────────
    target_resting = target_obj.data.root_pos_w[0].cpu().numpy()
    target_initial_z = float(target_resting[2])
    tx, ty, tz = target_resting.tolist()
    hover_z = tz + target_info["hover_z"]
    grasp_z = tz + target_info["grasp_z"]

    HOVER_POS = (tx, ty + target_info["y_nudge"], hover_z)
    GRASP_POS = (tx, ty + target_info["y_nudge"], grasp_z)

    # ── Build trajectory ───────────────────────────────────────────
    trajectory = [
        (3.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_OPEN),
        (4.0, HOVER_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (3.0, GRASP_POS, EE_ORIENT_DOWN, GRIPPER_OPEN),
        (2.0, GRASP_POS, EE_ORIENT_DOWN, GRIPPER_CLOSE),
        (3.0, HOVER_POS, EE_ORIENT_DOWN, GRIPPER_CLOSE),
        (3.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_CLOSE),
        (2.0, HOME_POS,  EE_ORIENT_DOWN, GRIPPER_OPEN),
    ]
    player = PoseTrajectoryPlayer(trajectory, device=device)

    # ── Run trajectory + record every N steps ──────────────────────
    buffer = EpisodeBuffer(
        main_images=[], wrist_images=[],
        ee_poses=[], joint_positions=[],
        gripper_states=[], actions_7d=[], timestamps=[],
    )

    last_grip_target = GRIPPER_OPEN
    t_sim = 0.0
    step = 0
    prev_ee_pos  = None
    prev_ee_quat = None
    prev_grip    = GRIPPER_OPEN
    finished_at = None
    success_seen = False
    success_diag = None

    while True:
        tgt_pos_w, tgt_quat_w, grip_target, finished = player.sample(t_sim)

        # 跑完軌跡後再多跑 60 step 等物件穩定
        if finished and finished_at is None:
            finished_at = step
        if finished_at is not None and step - finished_at > 60:
            break

        # ── IK ─────────────────────────────────────────────────────
        root = robot.data.root_state_w[:, :7]
        root_pos = root[:, :3]
        tgt_pos_b  = tgt_pos_w.unsqueeze(0) - root_pos
        tgt_quat_b = tgt_quat_w.unsqueeze(0)
        ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))

        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, :7]
        ee_pos_b  = ee_pose_w[:, :3] - root_pos
        ee_quat_b = ee_pose_w[:, 3:]
        q_current = robot.data.joint_pos[:, arm_ids_t]

        jac_full = robot.root_physx_view.get_jacobians()
        jac = jac_full[:, ee_jac_idx, :, :][:, :, arm_ids_t]

        q_target = ik.compute(ee_pos_b, ee_quat_b, jac, q_current)
        robot.set_joint_position_target(q_target, joint_ids=arm_ids_t)

        # ── Gripper smoothing ──────────────────────────────────────
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
            gripper_lows.unsqueeze(0), gripper_highs.unsqueeze(0),
        )
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids_t)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # 判定只看「爪子是否在關閉指令後到達抓起後的目標點」。
        # 目前 YCB visual mesh 是 attach 在 proxy 下, 畫面上看起來有抓到時,
        # RigidObject root pose 仍可能沒有反映 visual mesh 的移動, 所以不要依賴 obj lift。
        if not success_seen:
            ee_pos_check = robot.data.body_state_w[0, ee_body_idx, :3].cpu().numpy()
            if grip_target > 0.2:
                success_now, diag_now = detect_target_reached(
                    ee_pos_check, HOVER_POS, target_initial_z,
                )
                if success_now:
                    success_seen = True
                    success_diag = diag_now
                    success_diag["success_step"] = int(step)
                    success_diag["success_time"] = float(t_sim)

        # ── Logging every record_every_n_steps ─────────────────────
        if step % record_every_n_steps == 0:
            # Read current state
            ee_pose_now = robot.data.body_state_w[0, ee_body_idx, :7].cpu().numpy()
            ee_pos_now  = ee_pose_now[:3]
            ee_quat_now = ee_pose_now[3:]  # [w, x, y, z]
            joint_now   = robot.data.joint_pos[0, arm_ids_t].cpu().numpy()
            grip_now    = float(robot.data.joint_pos[0, finger_ids_t[0]].cpu().item())

            # Compute action_7d from (prev -> current) EE pose
            if prev_ee_pos is None:
                # 第一個 step, 沒有 prev, 用 zero action
                action_7d = np.zeros(7, dtype=np.float32)
                # gripper 用當下的 binary target
                action_7d[6] = 1.0 if grip_target > 0.2 else 0.0
            else:
                action_7d = compute_action_from_ee_poses(
                    prev_ee_pos, prev_ee_quat,
                    ee_pos_now,  ee_quat_now,
                    gripper_target=1.0 if grip_target > 0.2 else 0.0,
                )

            # 取相機畫面
            try:
                main_rgb  = scene["camera_main" ].data.output["rgb"][0].cpu().numpy().astype(np.uint8)
                wrist_rgb = scene["camera_wrist"].data.output["rgb"][0].cpu().numpy().astype(np.uint8)
            except Exception as e:
                log(f"  camera read failed: {e}")
                main_rgb  = np.zeros((256, 256, 3), dtype=np.uint8)
                wrist_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

            buffer.main_images.append(main_rgb)
            buffer.wrist_images.append(wrist_rgb)
            buffer.ee_poses.append([
                float(ee_pos_now[0]), float(ee_pos_now[1]), float(ee_pos_now[2]),
                float(ee_quat_now[0]), float(ee_quat_now[1]),
                float(ee_quat_now[2]), float(ee_quat_now[3]),
            ])
            buffer.joint_positions.append(joint_now.tolist())
            buffer.gripper_states.append(grip_now)
            buffer.actions_7d.append(action_7d.tolist())
            buffer.timestamps.append(t_sim)

            prev_ee_pos  = ee_pos_now
            prev_ee_quat = ee_quat_now

        t_sim += sim_dt
        step += 1

    # ── Success check ──────────────────────────────────────────────
    if success_seen:
        success = True
        diag = success_diag
    else:
        ee_pos_final = robot.data.body_state_w[0, ee_body_idx, :3].cpu().numpy()
        success, diag = detect_target_reached(ee_pos_final, HOVER_POS, target_initial_z)

    diag["target_initial_pos"]  = list(map(float, target_resting.tolist()))
    diag["num_steps_recorded"]  = len(buffer.main_images)
    diag["spawn_pos"]           = list(map(float, spawn_pos))
    diag["spawn_rot"]           = list(map(float, spawn_rot))

    return success, (buffer if success else None), diag


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Section 8 — Main                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    target_name  = args_cli.target
    target_info  = TARGETS[target_name]
    instruction  = f"pick up the {target_name}"

    out_dir_rel  = _extra_args.output_dir
    out_dir      = (Path(__file__).resolve().parent / out_dir_rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_target_episodes = _extra_args.episodes
    max_tried = _extra_args.max_episodes_tried or max(n_target_episodes * 3, n_target_episodes + 10)

    log(f"Target object   : {target_name}")
    log(f"Instruction     : '{instruction}'")
    log(f"Target episodes : {n_target_episodes}")
    log(f"Max tried       : {max_tried}")
    log(f"argv            : {sys.argv}")
    log(f"Output dir      : {out_dir}")
    log(f"Random seed     : {_extra_args.seed}")
    log(f"Randomize pos / rot / light: {_extra_args.randomize_pos} / "
        f"{_extra_args.randomize_rot} / {_extra_args.randomize_light}")

    rng = np.random.default_rng(_extra_args.seed)

    # ── Phase 1-3: 跟 sim_r.py 完全一樣 ─────────────────────────────
    enable_extensions()
    log("Phase 1: spawn + Robot Assembler")
    spawn_raw_and_assemble()

    log("Phase 2: SimulationContext + InteractiveScene")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT))
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    log("Attaching YCB visual meshes...")
    from isaacsim.core.utils.stage import add_reference_to_stage
    for target_key, info in TARGETS.items():
        usd_abs = f"{ISAAC_NUCLEUS_DIR}/{info['usd_relative']}"
        visual_path = f"/World/{target_key.capitalize()}/Visuals"
        try:
            add_reference_to_stage(usd_path=usd_abs, prim_path=visual_path)
        except Exception as e:
            log(f"  attach failed ({target_key}): {e}")

    log("Phase 3: sim.reset() + sim.play()")
    sim.reset()
    sim.play()
    sim_dt = sim.get_physics_dt()

    stage = omni.usd.get_context().get_stage()
    hide_proxy_meshes(stage, TARGETS.keys())

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

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls",
    )
    ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    target_obj = scene[target_name]

    # ── Episode loop ──────────────────────────────────────────────
    log("=" * 60)
    log("Starting episode collection...")
    log("=" * 60)

    n_success = 0
    n_tried   = 0
    t_start   = time.monotonic()

    while n_success < n_target_episodes and n_tried < max_tried:
        n_tried += 1
        ep_id = n_success
        log(f"\n[Ep {n_tried:04d} | success={n_success}/{n_target_episodes}]")

        try:
            success, buffer, diag = run_one_episode(
                sim, scene, robot, ik, sim_dt,
                arm_ids_t, finger_ids_t,
                gripper_signs, gripper_lows, gripper_highs,
                ee_body_idx, ee_jac_idx,
                target_name, target_info, target_obj,
                rng, instruction,
            )
        except Exception as e:
            log(f"  EXCEPTION: {e}")
            import traceback; log(traceback.format_exc())
            continue

        if success:
            log(f"  ✅ SUCCESS  lift={diag['obj_lift_height']:.3f}m  "
                f"target_dist={diag.get('ee_target_dist', 0):.3f}m  "
                f"steps={diag['num_steps_recorded']}")
            ep_base = out_dir / f"episode_{ep_id:05d}"
            meta = {
                "episode_id":      ep_id,
                "target":          target_name,
                "instruction":     instruction,
                "success":         True,
                "diag":            diag,
                "num_steps":       diag["num_steps_recorded"],
                "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tried_index":     n_tried,
            }
            try:
                save_episode(buffer, ep_base, meta)
            except Exception as e:
                log(f"  SAVE FAILED: {type(e).__name__}: {e}")
                import traceback; log(traceback.format_exc())
                continue
            n_success += 1
            log(f"  Progress: success={n_success}/{n_target_episodes}, "
                f"tried={n_tried}/{max_tried}")
        else:
            log(f"  ❌ FAILED   lift={diag.get('obj_lift_height', 0):.3f}m  "
                f"lifted={diag.get('obj_lifted')}  grip={diag.get('gripper_closed')}  "
                f"near={diag.get('obj_near_ee')}  "
                f"target_reached={diag.get('target_reached')}  "
                f"target_dist={diag.get('ee_target_dist', 0):.3f}m")
            log(f"  Progress: success={n_success}/{n_target_episodes}, "
                f"tried={n_tried}/{max_tried}")

    elapsed = time.monotonic() - t_start
    log("=" * 60)
    if n_success >= n_target_episodes:
        log("Stop reason: reached target successful episodes")
    elif n_tried >= max_tried:
        log("Stop reason: reached max episode attempts")
    else:
        log("Stop reason: episode loop exited")
    log(f"Collection done. {n_success}/{n_target_episodes} episodes in {elapsed/60:.1f} min")
    log(f"Success rate: {n_success}/{n_tried} = {100*n_success/max(1,n_tried):.1f}%")
    log(f"Average per success episode: {elapsed/max(1,n_success):.1f} sec")
    log(f"Output: {out_dir}")

    if _extra_args.keep_sim_alive:
        log("Collection complete. Keeping simulation alive. Press Ctrl+C to close.")
        try:
            while app.is_running():
                sim.step()
                scene.update(sim_dt)
        except KeyboardInterrupt:
            log("Ctrl+C received. Closing simulation.")
    else:
        sim.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
