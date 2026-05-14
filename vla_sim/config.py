"""Project constants that are safe to import before Isaac Lab starts."""

# Simulation timing.
PHYSICS_DT = 1.0 / 60.0

# Camera resolutions.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WRIST_CAMERA_WIDTH = 640
WRIST_CAMERA_HEIGHT = 480
ORBIT_CAMERA_WIDTH = 640
ORBIT_CAMERA_HEIGHT = 480

# Recording defaults.
DEFAULT_CAMERA_FPS = 30
DEFAULT_RECORD_SECONDS = 30
DEFAULT_ORBIT_SNAPSHOTS = 24
DEFAULT_ORBIT_VIDEO_FRAMES = 240

# USD asset paths relative to ISAAC_NUCLEUS_DIR.
UR3E_USD_RELATIVE = "Robots/UniversalRobots/ur3e/ur3e.usd"
GRIPPER_USD_RELATIVE = "Robots/Robotiq/2F-140/2f140_instanceable.usd"
YCB_NUCLEUS_PATH = "Props/YCB/Axis_Aligned"

# Stage prim paths.
ROBOT_PRIM_PATH = "/World/Robot"
GRIPPER_PRIM_PATH = "/World/Gripper"
UR3E_MOUNT_ABS = f"{ROBOT_PRIM_PATH}/wrist_3_link"
GRIPPER_MOUNT_REL_CANDIDATES = [
    "robotiq_arg2f_base_link",
    "base_link",
    "robotiq_base_link",
    "robotiq_2f140_base_link",
]

ASSEMBLY_NAMESPACE = "Gripper"
VARIANT_NAME = "ur3e_with_2f140"

# Robot pose and kinematics.
ROBOT_BASE_POS = (0.0, 0.0, 1.05)
ROBOT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)

EE_BODY_NAME = "wrist_3_link"
EE_ORIENT_DOWN = (0.0, 1.0, 0.0, 0.0)

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.38

HOME_POS = (0.30, 0.13, 1.35)
HOME_Q = [0.00, -1.57, 1.57, -1.57, -1.57, 0.00]


# End-effector workspace clamp (x, y, z).
WORKSPACE_X = (-0.4, 0.7)
WORKSPACE_Y = (-0.5, 0.5)
WORKSPACE_Z = (1.05, 1.65)


# Orbit camera trajectory used by optional debug captures.
ORBIT_CENTER = (0.30, 0.15, 1.15)
ORBIT_RADIUS = 1.0
ORBIT_HEIGHT = 1.55

CAMERA_MAIN_POS = (1, 0.0, 1.8)
CAMERA_MAIN_ROT = (0.71, 0.32, 0.27, 0.77)
CAMERA_MAIN_FOCAL = 12.0

# YCB target objects.
TARGETS = {
    "banana": {
        "usd_relative": f"{YCB_NUCLEUS_PATH}/011_banana.usd",
        "spawn_pos": (0.4, 0.2, 1.15),
        "spawn_rot": (1.0, 0.0, 0.0, 0.0),
        "mass": 0.15,
        "grasp_z": 0.180,
        "hover_z": 0.280,
        "y_nudge": 0.000,
        "size": (0.20, 0.06, 0.07),
    },
    "mug": {
        "usd_relative": f"{YCB_NUCLEUS_PATH}/025_mug.usd",
        "spawn_pos": (0.2, 0.30, 1.20),
        "spawn_rot": (0.7071, -0.7071, 0.0, 0.0),
        "mass": 0.20,
        "grasp_z": 0.200,
        "hover_z": 0.330,
        "y_nudge": 0.000,
        "size": (0.08, 0.08, 0.08),
    },
}

# Software mimic for 2f140_instanceable.usd.
# (sign, lower_bound, upper_bound) per joint
GRIPPER_MIMIC_MAP = {
    "finger_joint": (1.0, -0.02, 0.75),
    "left_inner_knuckle_joint": (1.0, -0.02, 0.02),
    "left_inner_finger_joint":   (-1.0, -0.75, 0.02),
    "right_outer_knuckle_joint": (-1.0, -0.75, 0.02),
    "right_inner_knuckle_joint": (-1.0, -0.02, 0.02),
    "right_inner_finger_joint": (1.0, -0.02, 0.75),
}
