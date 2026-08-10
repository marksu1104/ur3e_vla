"""Extra camera parameters for the multiview canonical scene.

Positions/rotations below place two new third-person cameras roughly
orthogonal to the existing ``camera_policy`` ("main") camera, looking
at the same in-front-of-robot point it's aimed at, in front of the
robot.

Derivation (see chat for the full derivation):
1. Reconstructed the world-space look direction of the existing main
   camera (``CAMERA_MAIN_POS`` / ``CAMERA_MAIN_ROT`` in
   ``vla_sim/config.py``) and intersected it with the approximate
   tabletop/object height (z ~= 1.15) to get an implied look-at point,
   ~(0.25, 0.30, 1.15), i.e. slightly in front of the robot base.
2. Rotated the main camera's position by +-90 degrees of azimuth
   around that look-at point (about the world Z axis, since Z is
   "up" in this scene), keeping the same radius and height offset.
3. Re-solved a standard look-at rotation (world-space Z-up,
   ``convention="opengl"`` camera axes: local -Z is the view
   direction) toward the same look-at point for each new position.

These are reasonable geometric starting points, not hand-tuned
composition -- nudge the ``*_POS`` z or the assumed look-at point and
recompute if the framing needs adjusting once you see it in the
simulator.
"""

# Quaternions are (w, x, y, z), matching the convention used throughout
# vla_sim/config.py (e.g. ROBOT_BASE_ROT, CAMERA_MAIN_ROT).
# 注視點(工作位置) (0.30, 0.30, 1.15)

CAMERA_TOP_POS = (0.3, 0.3, 1.9)          # 注視點正上方，離桌面約 1.15m
CAMERA_TOP_ROT = (1.0, 0.0, 0.0, 0.0)      # 單位四元數 = 垂直朝下看，無旋轉

# z=1.5
"""CAMERA_LEFT_POS = (-0.05, 0.65, 1.5)
CAMERA_LEFT_ROT = (-0.3108, -0.1542, 0.4170, 0.8401)
CAMERA_RIGHT_POS = (0.55, -0.05, 1.5)
CAMERA_RIGHT_ROT = (0.8401, 0.4170, 0.1542, 0.3108)"""

# z=1.33
"""CAMERA_LEFT_POS =  (-0.2, 0.7, 1.33)
CAMERA_LEFT_ROT =  (-0.3052, -0.2309, 0.5573, 0.7368)
CAMERA_RIGHT_POS =  (0.7, -0.2, 1.33)
CAMERA_RIGHT_ROT =  (0.7368, 0.5573, 0.2309, 0.3052)"""
CAMERA_LEFT_POS =  (-0.15, 0.65, 1.33)
CAMERA_LEFT_ROT =  (-0.3089, -0.2259, 0.5453, 0.7458)
CAMERA_RIGHT_POS =  (0.65, -0.15, 1.33)
CAMERA_RIGHT_ROT =  (0.7458, 0.5453, 0.2259, 0.3089)

# z=1.3
"""CAMERA_LEFT_POS = (-0.2, 0.8, 1.3)
CAMERA_LEFT_ROT = (-0.2974, -0.2409, 0.5816, 0.7179)
CAMERA_RIGHT_POS = (0.8, -0.2, 1.3)
CAMERA_RIGHT_ROT = (0.7179, 0.5816, 0.2409, 0.2974)"""


###### RUN THIS FILE DIRECTLY TO compute LEFT, RIGHT CAMERA angle
"""
cd ~/IsaacLab
conda activate env_isaaclab_ros2

python ur3e_vla/vla_sim/multiview/config_multiview.py
"""

def main():
    import numpy as np

    def _matrix_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
        m00, m01, m02 = R[0]
        m10, m11, m12 = R[1]
        m20, m21, m22 = R[2]
        tr = m00 + m11 + m22
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * S, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = np.sqrt(1.0 + m00 - m11 - m22) * 2
            w, x, y, z = (m21 - m12) / S, 0.25 * S, (m01 + m10) / S, (m02 + m20) / S
        elif m11 > m22:
            S = np.sqrt(1.0 + m11 - m00 - m22) * 2
            w, x, y, z = (m02 - m20) / S, (m01 + m10) / S, 0.25 * S, (m12 + m21) / S
        else:
            S = np.sqrt(1.0 + m22 - m00 - m11) * 2
            w, x, y, z = (m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, 0.25 * S
        q = np.array([w, x, y, z])
        return tuple((q / np.linalg.norm(q)).tolist())


    def _look_at_quat(
        pos: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])
    ) -> tuple[float, float, float, float]:
        """wxyz quaternion for an 'opengl'-convention CameraCfg offset.

        Camera local -Z points at `target`; local +Y is 'up' derived from world_up.
        """
        forward = target - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up_cam = np.cross(right, forward)
        R = np.stack([right, up_cam, -forward], axis=1)
        return _matrix_to_quat_wxyz(R)


    def compute_orthogonal_cameras(
        target: tuple[float, float, float],
        main_pos: tuple[float, float, float],
        height: float | None = None,
        world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> dict[str, dict[str, tuple[float, float, float]]]:
        """Derive LEFT/RIGHT camera pos+rot, +-90 deg azimuth around `target`.

        Keeps the same horizontal (xy-plane) distance to `target` as `main_pos`,
        places both new cameras at `height` (defaults to main_pos's own z), and
        points each one back at `target`.
        """
        target_v = np.asarray(target, dtype=float)
        main_v = np.asarray(main_pos, dtype=float)
        up_v = np.asarray(world_up, dtype=float)
        if height is None:
            height = float(main_v[2])

        rel = main_v - target_v
        radius_xy = float(np.hypot(rel[0], rel[1]))
        az0 = float(np.arctan2(rel[1], rel[0]))

        out: dict[str, dict[str, tuple]] = {}
        for name, dphi_deg in (("left", 90.0), ("right", -90.0)):
            az = az0 + np.radians(dphi_deg)
            pos = target_v + np.array([radius_xy * np.cos(az), radius_xy * np.sin(az), 0.0])
            pos[2] = height
            rot = _look_at_quat(pos, target_v, up_v)
            out[name] = {"pos": tuple(np.round(pos, 4).tolist()), "rot": tuple(np.round(rot, 4).tolist())}
        return out
    ### MAIN SETTING ###
    result = compute_orthogonal_cameras(
        target=(0.25, 0.25, 1.15),   # 注視點 x, y, z, 桌面1.05
        main_pos=(0.65, 0.65, 1.5),    # 現有 main camera 位置（拿來算半徑/方位角基準）
        height=1.33,                  # Left/Right 要放的高度；不填就跟 main 同高
    )

    CAMERA_LEFT_POS,  CAMERA_LEFT_ROT  = result["left"]["pos"],  result["left"]["rot"]
    CAMERA_RIGHT_POS, CAMERA_RIGHT_ROT = result["right"]["pos"], result["right"]["rot"]
    print("CAMERA_LEFT_POS = ", CAMERA_LEFT_POS, "\nCAMERA_LEFT_ROT = ", CAMERA_LEFT_ROT)
    print("CAMERA_RIGHT_POS = ", CAMERA_RIGHT_POS, "\nCAMERA_RIGHT_ROT = ", CAMERA_RIGHT_ROT)

if __name__ == "__main__":
    main()