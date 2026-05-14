"""
envs.data_collector — Episode-level demo recorder.

整合自夥伴版本 (原始檔案有提供 image/hand_image/state/action/meta schema),
做了三個修正:

  1. 第一個 step 的 action = zeros (原版會用絕對 pose, distribution outlier)
  2. 移除 print 在 record_step 內 (蒐 demo 時太吵)
  3. 加 reset() 內可選傳 episode_idx, 方便 episode loop 整合

Usage (跟 sim_collect.py 整合):
    from envs.data_collector import DataCollector

    collector = DataCollector(
        save_dir="raw/run0",
        prompt="pick up the red mug",
    )

    for ep_id in range(N):
        collector.reset()
        # ... 跑 trajectory ...
        for step in main_loop:
            if step % record_every_n_steps == 0:
                collector.record_step(
                    scene=scene,
                    gripper_state=1 if grip > 0.2 else 0,
                    action_blocked=0,
                    is_last=False,
                    is_terminal=False,
                )
        # 最後一個 step 標記 terminal
        collector.record_step(
            scene=scene, gripper_state=..., is_last=True, is_terminal=True,
        )
        collector.save(episode_idx=ep_id)

Schema (per step):
    image                : (H, W, 3) uint8       — main camera RGB
    hand_image           : (H, W, 3) uint8       — wrist camera RGB
    state                : (15,) float32         — joints(6) + ee_pos(3) +
                                                    ee_quat(4) + grip(1) + blocked(1)
    action               : (8,) float32          — [dx, dy, dz, drx, dry, drz,
                                                    gripper, is_terminal]
    language_instruction : str
    is_first / is_last / is_terminal : bool

Action convention:
    - Delta in **TCP 相對 base_link 的 frame** (取自 ee_frame.target_pos_source)
    - Translation: meter
    - Rotation: euler XYZ (intrinsic), radian
    - Gripper: 0 = open, 1 = close
    - is_terminal: 0 = continue, 1 = end of episode
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R


class DataCollector:
    """Per-episode buffer + save to .npy.

    Schema: 跟 OpenVLA / RT-X 兼容的 list-of-dict format.
    """

    def __init__(self, save_dir: str, prompt: str = "pick up the object"):
        self.save_dir = save_dir
        self.prompt = prompt
        os.makedirs(self.save_dir, exist_ok=True)
        self.reset()

    def reset(self):
        """重置所有 buffer, 開始新 episode."""
        self.data_buffer = {
            "image":       [],
            "hand_image":  [],
            "robot_state": [],
            "action":      [],
            "task":        [],
            "is_first":    [],
            "is_last":     [],
            "is_terminal": [],
        }
        self.step_count = 0
        # 第一個 step 用 zeros, 之後計算 delta
        self.prev_ee_state = None

    def record_step(
        self, scene,
        gripper_state: int = 0,
        action_blocked: int = 0,
        is_last: bool = False,
        is_terminal: bool = False,
    ):
        """每一幀記錄一次. 從 scene 自動抓資料.

        Args:
            scene: IsaacLab InteractiveScene
            gripper_state: 0 = open, 1 = close (binary, 給 VLA 學)
            action_blocked: 1 = gripper 動作中, 其他動作禁用
            is_last:     此 step 是否為 episode 最後一步
            is_terminal: 此 step 是否為 terminal (用於 fine-tune)
        """
        self.step_count += 1

        # ── 1. Images ──────────────────────────────────────────────────
        img_main  = scene["camera_main" ].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        img_wrist = scene["camera_wrist"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)

        # ── 2. Robot state ─────────────────────────────────────────────
        current_joint_pos = scene["robot"].data.joint_pos[0].cpu().numpy()

        # 用 FrameTransformer 拿 TCP (相對 base_link)
        ee_data = scene["ee_frame"].data
        ee_pos  = ee_data.target_pos_source[0, 0].cpu().numpy()    # [x, y, z]
        ee_quat = ee_data.target_quat_source[0, 0].cpu().numpy()   # [w, x, y, z]

        # Quaternion -> Euler XYZ (radian)
        quat_scipy   = [ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]  # to [x,y,z,w]
        r            = R.from_quat(quat_scipy)
        euler_angles = r.as_euler('xyz', degrees=False)  # rx, ry, rz
        ee_state_euler = np.concatenate([ee_pos, euler_angles])  # (6,)

        # ── 3. Action: ee delta + gripper + terminate ─────────────────
        # ★ Bug fix: 第一個 step 用 zeros, 不要用絕對 pose
        if self.prev_ee_state is None:
            action_delta = np.zeros(6, dtype=np.float32)
        else:
            action_delta = (ee_state_euler - self.prev_ee_state).astype(np.float32)

        action = np.concatenate([
            action_delta,                                      # 6: dx,dy,dz,drx,dry,drz
            np.array([gripper_state, int(is_terminal)],
                     dtype=np.float32),                        # 2: grip, terminal
        ]).astype(np.float32)  # shape (8,)

        # ── 4. Combined robot_state (15,) ─────────────────────────────
        # joints(6) + ee_pos(3) + ee_quat(4) + grip(1) + blocked(1)
        combined_state = np.concatenate([
            current_joint_pos[:6],                  # arm 6 joints
            ee_pos,                                  # TCP xyz
            ee_quat,                                 # TCP quat wxyz
            np.array([gripper_state]),
            np.array([action_blocked]),
        ]).astype(np.float32)

        # ── 5. Append to buffer ───────────────────────────────────────
        self.data_buffer["image"].append(img_main)
        self.data_buffer["hand_image"].append(img_wrist)
        self.data_buffer["robot_state"].append(combined_state)
        self.data_buffer["action"].append(action)
        self.data_buffer["task"].append(self.prompt)
        self.data_buffer["is_first"].append(self.step_count == 1)
        self.data_buffer["is_last"].append(is_last)
        self.data_buffer["is_terminal"].append(is_terminal)

        # Update prev for next step's delta
        self.prev_ee_state = ee_state_euler

    def save(self, episode_idx: int = 0, logged: bool = False) -> str:
        """寫成 episode_{idx}.npy.

        Returns:
            完整檔案路徑 (給 caller 紀錄用)
        """
        file_path = os.path.join(self.save_dir, f"episode_{episode_idx:05d}.npy")
        num_steps = len(self.data_buffer["robot_state"])

        episode_list = []
        for i in range(num_steps):
            episode_list.append({
                'image':                np.array(self.data_buffer["image"][i],        dtype=np.uint8),
                'hand_image':           np.array(self.data_buffer["hand_image"][i],   dtype=np.uint8),
                'state':                np.array(self.data_buffer["robot_state"][i],  dtype=np.float32),
                'action':               np.array(self.data_buffer["action"][i],       dtype=np.float32),
                'language_instruction': str(self.data_buffer["task"][i]),
                'is_first':             bool(self.data_buffer["is_first"][i]),
                'is_last':              bool(self.data_buffer["is_last"][i]),
                'is_terminal':          bool(self.data_buffer["is_terminal"][i]),
            })

        # allow_pickle 因為存 list of dict
        np.save(file_path, episode_list, allow_pickle=True)

        if logged:
            print(f"[DataCollector] Saved {file_path} ({num_steps} steps)")
            if num_steps > 0:
                sample = episode_list[0]
                print(f"  image shape:     {sample['image'].shape}")
                print(f"  state shape:     {sample['state'].shape}")
                print(f"  action shape:    {sample['action'].shape}")
                print(f"  instruction:     '{sample['language_instruction']}'")

        return file_path

    @property
    def num_steps(self) -> int:
        return len(self.data_buffer["robot_state"])