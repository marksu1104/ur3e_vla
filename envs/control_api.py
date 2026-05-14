"""
envs.control_api — Action contract.

定義 OpenVLA / scripted expert / replay 共用的 7-DoF delta action 規格.
這是「事實的記錄」, 不是「介面抽象」. 之後 sim_collect / sim_vla / real deploy
都應該用一致的 action format 處理 / 儲存資料.

Action contract:
    Dimension : 7
    Format    : [dx, dy, dz, droll, dpitch, dyaw, gripper]
    Frame     : robot base frame
    Rate      : 5 Hz (預設, 但可調)
    Units     : translation in meter, rotation in radian
    Gripper   : 0.0 = open, 1.0 = close

Usage:
    from envs.control_api import (
        ACTION_DIM, ACTION_RATE_HZ,
        clamp_action, action_to_dict,
    )

    # 在你的 main loop:
    action = vla.predict(...)        # 7-element numpy array
    action = clamp_action(action)    # 確保不超過安全範圍
    apply_to_robot(action)

    # 蒐 demo 時記錄:
    episode["actions"].append(action_to_dict(action))
"""

import numpy as np


# ── Contract constants ────────────────────────────────────────────────
ACTION_DIM     = 7
ACTION_RATE_HZ = 5     # VLA call frequency

# Safety limits (每個 control step 最多移動的量)
# 5 Hz 控制下, 3cm/step = 15cm/s 線速度, 是 UR3e 安全範圍
TRANSLATION_CLAMP = 0.03    # meter per step
ROTATION_CLAMP    = 0.10    # rad per step


def clamp_action(action_7d) -> np.ndarray:
    """Clamp 7D delta action 到安全範圍.

    Args:
        action_7d: 7-element array-like, [dx, dy, dz, droll, dpitch, dyaw, gripper]

    Returns:
        clamped numpy array (float32)
    """
    action = np.asarray(action_7d, dtype=np.float32).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D action, got shape {action.shape}")
    action = action.copy()
    action[:3]  = np.clip(action[:3],  -TRANSLATION_CLAMP, TRANSLATION_CLAMP)
    action[3:6] = np.clip(action[3:6], -ROTATION_CLAMP,    ROTATION_CLAMP)
    action[6]   = np.clip(action[6], 0.0, 1.0)
    return action


def action_to_dict(action_7d) -> dict:
    """轉成 dict 方便 logging / JSON 寫檔."""
    a = np.asarray(action_7d, dtype=np.float32).reshape(-1)
    return {
        "dx":      float(a[0]),
        "dy":      float(a[1]),
        "dz":      float(a[2]),
        "droll":   float(a[3]),
        "dpitch":  float(a[4]),
        "dyaw":    float(a[5]),
        "gripper": float(a[6]),
    }


def action_from_dict(d: dict) -> np.ndarray:
    """從 dict 還原成 numpy array (replay / dataset loading 用)."""
    return np.array([
        d["dx"], d["dy"], d["dz"],
        d["droll"], d["dpitch"], d["dyaw"],
        d["gripper"],
    ], dtype=np.float32)


def compute_action_from_ee_poses(
    ee_pos_curr, ee_quat_curr,
    ee_pos_next, ee_quat_next,
    gripper_target,
) -> np.ndarray:
    """從相鄰兩個 EE pose 反推 7D delta action.

    Used by sim_collect.py 蒐 demo 時:
        每個 step 看當下 EE 跟下一個 EE 的差異 → 計算 action
        這個 action 就是 VLA 學的 target.

    Args:
        ee_pos_curr, ee_pos_next: (3,) array, world frame position
        ee_quat_curr, ee_quat_next: (4,) array, [w, x, y, z]
        gripper_target: float, 0.0 = open, 1.0 = close

    Returns:
        (7,) numpy array
    """
    ee_pos_curr  = np.asarray(ee_pos_curr,  dtype=np.float32)
    ee_pos_next  = np.asarray(ee_pos_next,  dtype=np.float32)
    ee_quat_curr = np.asarray(ee_quat_curr, dtype=np.float32)
    ee_quat_next = np.asarray(ee_quat_next, dtype=np.float32)

    # Delta translation (world frame; 之後若需要 base frame, caller 自己轉)
    delta_pos = ee_pos_next - ee_pos_curr

    # Delta rotation: q_delta = q_next * q_curr^-1
    # q^-1 = [w, -x, -y, -z] for unit quaternion
    w0, x0, y0, z0 = ee_quat_curr
    w1, x1, y1, z1 = ee_quat_next
    # q_curr_inv
    w0i, x0i, y0i, z0i = w0, -x0, -y0, -z0
    # delta = q_next * q_curr_inv
    dw = w1*w0i - x1*x0i - y1*y0i - z1*z0i
    dx = w1*x0i + x1*w0i + y1*z0i - z1*y0i
    dy = w1*y0i - x1*z0i + y1*w0i + z1*x0i
    dz = w1*z0i + x1*y0i - y1*x0i + z1*w0i

    # Quaternion to small-angle Euler (近似, delta 小時 OK)
    # 2 * imaginary part ≈ axis * angle
    droll  = 2.0 * dx
    dpitch = 2.0 * dy
    dyaw   = 2.0 * dz

    return np.array([
        delta_pos[0], delta_pos[1], delta_pos[2],
        droll, dpitch, dyaw,
        float(gripper_target),
    ], dtype=np.float32)