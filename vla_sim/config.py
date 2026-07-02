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
ROBOT_BASE_ROT = (0.9239, 0.0, 0.0, -0.3827)

# Workspace furniture.
TABLE_USD_RELATIVE = "Props/Mounts/ThorlabsTable/table_instanceable.usd"
TABLE_SCALE = (1.8, 0.8, 1.0)
TABLE_A_POS = (1.17, 0.17, 1.05)
TABLE_B_POS = (1.17, 0.775, 1.05)
TABLE_ROT = (0.0, 0.0, 0.0, 1.0)

TABLE_MAT_SIZE = (1.2, 0.6, 0.01)
TABLE_MAT_A_POS = (0.46, 0.17, 1.055)
TABLE_MAT_B_POS = (0.46, 0.775, 1.055)
TABLE_MAT_COLOR = (0.08, 0.08, 0.08)

BACKDROP_HEIGHT = 1.90
BACKDROP_THICKNESS = 0.04
BACKDROP_Z = 0.95
BACKDROP_COLOR = (0.005, 0.005, 0.005)
BACKDROP_BACK_SIZE = (BACKDROP_THICKNESS, 1.90, BACKDROP_HEIGHT)
BACKDROP_BACK_POS = (-0.23, 0.61, BACKDROP_Z)
BACKDROP_SIDE_SIZE = (2.18, BACKDROP_THICKNESS, BACKDROP_HEIGHT)
BACKDROP_SIDE_POS = (0.76, -0.23, BACKDROP_Z)

EE_BODY_NAME = "wrist_3_link"
EE_ORIENT_DOWN = (0.0, 1.0, 0.0, 0.0)

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.26

HOME_POS = (0.30, 0.13, 1.35)
HOME_Q = [0.57, -1.57, 1.57, -1.57, -1.57, 0.00]


# End-effector workspace clamp (x, y, z).
WORKSPACE_X = (-0.4, 0.7)
WORKSPACE_Y = (-0.5, 0.5)
WORKSPACE_Z = (1.05, 1.48)


# Orbit camera trajectory used by optional debug captures.
ORBIT_CENTER = (0.30, 0.15, 1.15)
ORBIT_RADIUS = 1.0
ORBIT_HEIGHT = 1.55

CAMERA_MAIN_POS = (0.6, 0.6, 1.5)
CAMERA_MAIN_ROT = (0.28, 0.13, 0.43, 0.85)
CAMERA_MAIN_FOCAL = 11.0

# Small discrete view set for dataset collection. The first entry matches the
# standard camera used by run_vla; collect_demos can sample these per episode.
CAMERA_MAIN_VIEW_PRESETS = (
    {"name": "standard", "pos": CAMERA_MAIN_POS, "rot": CAMERA_MAIN_ROT},
    {"name": "left", "pos": (0.54, 0.66, 1.50), "rot": (0.29, 0.12, 0.39, 0.87)},
    {"name": "right", "pos": (0.66, 0.54, 1.50), "rot": (0.27, 0.15, 0.47, 0.83)},
    {"name": "high", "pos": (0.58, 0.58, 1.60), "rot": (0.34, 0.15, 0.42, 0.83)},
)

# YCB target objects.
TARGETS = {
    "banana": {
        "usd_relative": f"{YCB_NUCLEUS_PATH}/011_banana.usd",
        "spawn_pos": (0.5, 0.0, 1.00),
        "spawn_rot": (1.0, 0.0, 0.0, 0.0),
        "mass": 0.15,
        "grasp_z": 0.180,
        "hover_z": 0.280,
        "y_nudge": 0.000,
        "size": (0.20, 0.06, 0.07),
    },
    "red_mug": {
        "usd_relative": f"{YCB_NUCLEUS_PATH}/025_mug.usd",
        "collision_usd": "025_mug_collision.usda",
        "spawn_pos": (0.10, 0.30, 1.20),
        "spawn_rot": (0.7071, -0.7071, 0.0, 0.0),
        "pos_randomization": {
            "x": (-0.025, 0.025),
            "y": (-0.030, 0.030),
        },
        "mass": 0.20,
        "grasp_z": 0.195,
        "hover_z": 0.295,
        "lift_z": 0.295,
        "align_gripper_to_yaw": True,
        "grasp_yaw_offsets": (0.0, 3.1416),
        "grasp_yaw_jitter": (-0.15, 0.15),
        "gripper_yaw_offset": 0.0,
        "carry_extra_z": 0.045,
        "min_carry_z": 1.36,
        "x_nudge": 0.000,
        "y_nudge": 0.000,
        "grasp_randomization": {
            "name": "body",
            "grasp_z": (0.180, 0.210),
            "hover_z": (0.270, 0.320),
            "lift_z": (0.270, 0.320),
            "x_nudge": (-0.005, 0.005),
            "y_nudge": (-0.005, 0.005),
        },
        "size": (0.08, 0.08, 0.08),
        "color": (1, 0.2, 0.2),
    },
    "blue_mug": {
        "usd_relative": f"{YCB_NUCLEUS_PATH}/025_mug.usd",
        "collision_usd": "025_mug_collision.usda",
        "spawn_pos": (0.30, 0.30, 1.20),
        "spawn_rot": (0.7071, -0.7071, 0.0, 0.0),
        "pos_randomization": {
            "x": (-0.020, 0.020),
            "y": (-0.025, 0.025),
        },
        "mass": 0.20,
        "grasp_z": 0.195,
        "hover_z": 0.295,
        "lift_z": 0.295,
        "align_gripper_to_yaw": True,
        "grasp_yaw_offsets": (0.0, 3.1416),
        "grasp_yaw_jitter": (-0.15, 0.15),
        "gripper_yaw_offset": 0.0,
        "carry_extra_z": 0.035,
        "min_carry_z": 1.34,
        "x_nudge": 0.000,
        "y_nudge": 0.000,
        "grasp_randomization": {
            "name": "body",
            "grasp_z": (0.180, 0.210),
            "hover_z": (0.270, 0.320),
            "lift_z": (0.270, 0.320),
            "x_nudge": (-0.005, 0.005),
            "y_nudge": (-0.005, 0.005),
        },
        "size": (0.08, 0.08, 0.08),
        "color": (0.02, 0.18, 0.85),
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
