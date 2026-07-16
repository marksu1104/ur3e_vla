"""Action and trajectory helpers shared by collection and VLA control.

Policy actions are 7D:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]

Dataset actions append one terminal flag and are therefore 8D.
"""

import numpy as np
import torch


ACTION_DIM = 7
DATASET_ACTION_DIM = 8
ACTION_RATE_HZ = 5

# Per-step safety limits at the policy control rate.
TRANSLATION_CLAMP = 0.03
ROTATION_CLAMP = 0.10


def clamp_action(action_7d) -> np.ndarray:
    """Clamp a 7D policy action to the configured safety limits."""
    action = np.asarray(action_7d, dtype=np.float32).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D action, got shape {action.shape}")
    action = action.copy()
    action[:3]  = np.clip(action[:3],  -TRANSLATION_CLAMP, TRANSLATION_CLAMP)
    action[3:6] = np.clip(action[3:6], -ROTATION_CLAMP,    ROTATION_CLAMP)
    action[6]   = np.clip(action[6], 0.0, 1.0)
    return action


def compute_action_from_ee_poses(
    ee_pos_curr, ee_quat_curr,
    ee_pos_next, ee_quat_next,
    gripper_target,
) -> np.ndarray:
    """Infer a 7D policy action from two adjacent end-effector poses."""
    ee_pos_curr = np.asarray(ee_pos_curr, dtype=np.float32)
    ee_pos_next = np.asarray(ee_pos_next, dtype=np.float32)
    ee_quat_curr = np.asarray(ee_quat_curr, dtype=np.float32)
    ee_quat_next = np.asarray(ee_quat_next, dtype=np.float32)

    delta_pos = ee_pos_next - ee_pos_curr

    w0, x0, y0, z0 = ee_quat_curr
    w1, x1, y1, z1 = ee_quat_next
    w0i, x0i, y0i, z0i = w0, -x0, -y0, -z0
    dx = w1 * x0i + x1 * w0i + y1 * z0i - z1 * y0i
    dy = w1 * y0i - x1 * z0i + y1 * w0i + z1 * x0i
    dz = w1 * z0i + x1 * y0i - y1 * x0i + z1 * w0i

    # Small-angle quaternion approximation: 2 * imaginary ~= axis * angle.
    droll = 2.0 * dx
    dpitch = 2.0 * dy
    dyaw = 2.0 * dz

    return np.array([
        delta_pos[0], delta_pos[1], delta_pos[2],
        droll, dpitch, dyaw,
        float(gripper_target),
    ], dtype=np.float32)


def apply_delta_action(
    ee_pos_current: torch.Tensor,
    ee_quat_current: torch.Tensor,
    action: np.ndarray,
    action_scale: float,
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
    workspace_z: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Apply a 7D delta action to the current end-effector pose."""
    device = ee_pos_current.device
    action_t = torch.tensor(action, dtype=torch.float32, device=device)

    delta_pos = action_t[:3] * action_scale
    target_pos = ee_pos_current + delta_pos
    target_pos[0] = target_pos[0].clamp(workspace_x[0], workspace_x[1])
    target_pos[1] = target_pos[1].clamp(workspace_y[0], workspace_y[1])
    target_pos[2] = target_pos[2].clamp(workspace_z[0], workspace_z[1])

    delta_euler = action_t[3:6] * action_scale
    half = delta_euler * 0.5
    cos_h = torch.cos(half)
    sin_h = torch.sin(half)

    delta_quat = torch.stack([
        cos_h[0] * cos_h[1] * cos_h[2] + sin_h[0] * sin_h[1] * sin_h[2],
        sin_h[0] * cos_h[1] * cos_h[2] - cos_h[0] * sin_h[1] * sin_h[2],
        cos_h[0] * sin_h[1] * cos_h[2] + sin_h[0] * cos_h[1] * sin_h[2],
        cos_h[0] * cos_h[1] * sin_h[2] - sin_h[0] * sin_h[1] * cos_h[2],
    ])

    w0, x0, y0, z0 = ee_quat_current
    w1, x1, y1, z1 = delta_quat
    target_quat = torch.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])
    target_quat = target_quat / torch.linalg.norm(target_quat)
    gripper_cmd = float(action_t[6].clamp(0.0, 1.0))

    return target_pos, target_quat, gripper_cmd


class PoseTrajectoryPlayer:
    """Linear pose trajectory sampler for scripted grasp demonstrations."""

    def __init__(self, trajectory, device):
        self.segments = []
        t0 = 0.0
        prev = trajectory[0]
        for duration, pos, quat, grip in trajectory:
            self.segments.append((
                t0,
                t0 + duration,
                torch.tensor(prev[1], device=device, dtype=torch.float32),
                torch.tensor(pos, device=device, dtype=torch.float32),
                torch.tensor(prev[2], device=device, dtype=torch.float32),
                torch.tensor(quat, device=device, dtype=torch.float32),
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
                alpha = (t - start) / (end - start)
                pos = p0 + alpha * (p1 - p0)
                q1_short = q1
                if torch.dot(q0, q1_short) < 0.0:
                    q1_short = -q1_short
                quat = q0 + alpha * (q1_short - q0)
                quat = quat / torch.linalg.norm(quat)
                grip = g0 + alpha * (g1 - g0)
                return pos, quat, grip, False
        _, _, _, p1, _, q1, _, g1 = self.segments[-1]
        return p1, q1, g1, True
