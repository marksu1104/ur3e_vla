"""Project constants that are safe to import before Isaac Lab starts."""

# Simulation timing.
PHYSICS_DT = 1.0 / 60.0

# Camera resolutions.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WRIST_CAMERA_WIDTH = 640
WRIST_CAMERA_HEIGHT = 480

# USD asset paths relative to ISAAC_NUCLEUS_DIR.
UR3E_USD_RELATIVE = "Robots/UniversalRobots/ur3e/ur3e.usd"
GRIPPER_USD_RELATIVE = (
    "Robots/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd"
)
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
GRIPPER_CLOSE = 0.785398

HOME_POS = (0.30, 0.13, 1.35)
HOME_Q = [0.57, -1.57, 1.57, -1.57, -1.57, 0.00]


# End-effector workspace clamp (x, y, z).
WORKSPACE_X = (-0.4, 0.7)
WORKSPACE_Y = (-0.5, 0.5)
WORKSPACE_Z = (1.05, 1.48)


CAMERA_MAIN_POS = (0.6, 0.6, 1.5)
CAMERA_MAIN_ROT = (0.28, 0.13, 0.43, 0.85)
CAMERA_MAIN_FOCAL = 11.0

# Persistent bridge and YOLO stream.
BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8100
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_EVERY_N_STEPS = 2
STREAM_JPEG_QUALITY = 85
MAX_TRIAL_SECONDS = 45.0

YOLO_CAMERA_POS = (0.13, 0.84, 1.31)
YOLO_CAMERA_ROT = (0.0, 0.0, 0.5671412673963498, 0.8236205332652058)
YOLO_CAMERA_FOCAL = 21.0
YOLO_VISIBILITY_MARGIN_PX = 12
YOLO_MIN_VISIBLE_PIXELS = 100
YOLO_TARGET_ROIS = {
    "spoon": (0.17, 0.45, 0.33, 0.73),
    "red_mug": (0.35, 0.38, 0.57, 0.72),
    "bowl": (0.57, 0.44, 0.84, 0.74),
}
YOLO_GOAL_ROIS = (
    (0.09, 0.40, 0.23, 0.51),
    (0.16, 0.32, 0.28, 0.42),
    (0.27, 0.36, 0.39, 0.46),
)

# Canonical left-to-right objects in camera_yolo.
TARGETS = {
    "red_mug": {
        "usd_relative": f"{YCB_NUCLEUS_PATH}/025_mug.usd",
        "collision_usd": "025_mug_collision.usda",
        "spawn_pos": (0.150, 0.390, 1.200),
        "spawn_rot": (0.7071, -0.7071, 0.0, 0.0),
        "mass": 0.200,
        "color": (1.00, 0.20, 0.20),
        "grasp_z": 0.195,
        "hover_z": 0.295,
        "x_nudge": 0.0,
        "y_nudge": 0.0,
        "align_gripper_to_yaw": True,
        "grasp_yaw_offsets": (0.0, 3.1416),
        "grasp_yaw_jitter": (-0.15, 0.15),
        "gripper_yaw_offset": 0.0,
        "min_carry_z": 1.360,
        "carry_extra_z": 0.045,
    },
    "spoon": {
        "collision_usd": "spoon_convexdecomposition_collision.usda",
        "scale": 0.670,
        "spawn_pos": (0.290, 0.390, 1.080),
        "spawn_rot": (0.0, 0.0, 0.0, 1.0),
        "mass": 0.060,
        "color": (0.08, 0.30, 0.90),
        "grasp_z": 0.215,
        "hover_z": 0.295,
        "x_nudge": 0.0,
        "y_nudge": -0.060,
        "align_gripper_to_yaw": True,
        "grasp_yaw_offsets": (0.0, 3.1416),
        "grasp_yaw_jitter": (0.0, 0.0),
        "gripper_yaw_offset": 1.5708,
        "min_carry_z": 1.340,
        "carry_extra_z": 0.030,
    },
    "bowl": {
        "usd_relative": f"{YCB_NUCLEUS_PATH}/024_bowl.usd",
        "collision_usd": "024_bowl_baked_collision.usda",
        "scale": 0.800,
        "spawn_pos": (-0.010, 0.400, 1.150),
        "spawn_rot": (0.7071, -0.7071, 0.0, 0.0),
        "mass": 0.250,
        "color": (0.10, 0.12, 0.14),
        "grasp_z": 0.215,
        "hover_z": 0.350,
        "x_nudge": 0.0,
        "y_nudge": -0.055,
        "align_gripper_to_yaw": False,
        "min_carry_z": 1.320,
        "carry_extra_z": 0.030,
    },
}
TARGET_KEYS = ("spoon", "red_mug", "bowl")
TASK_INDEX_MAP = dict(enumerate(TARGET_KEYS))

PLACE_POSITIONS = ((0.37, 0.13), (0.34, -0.025), (0.21, 0.045))
PLACE_MARKER_COLORS = (
    (0.05, 0.95, 0.10),
    (0.95, 0.05, 0.85),
    (1.00, 0.30, 0.02),
)
PLACE_MARKER_RADIUS = 0.045
PLACE_MARKER_THICKNESS = 0.002
