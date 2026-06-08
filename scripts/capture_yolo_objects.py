"""Capture static object images using the collect_demos scene setup."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument(
    "--output-dir",
    type=str,
    default="outputs/test/yolo_object_views",
    help="Directory for PNG captures and metadata.",
)
_extra.add_argument(
    "--num-images",
    type=int,
    default=1,
    help="Number of static RGB images to save.",
)
_extra.add_argument(
    "--render-steps",
    type=int,
    default=60,
    help="Simulation steps to hold the collect_demos home scene before capture.",
)
_extra.add_argument(
    "--show-gui",
    action="store_true",
    help="Show the Isaac Sim GUI while capturing.",
)
_extra.add_argument("--seed", type=int, default=42)
_extra_args, _ = _extra.parse_known_args()

if not _extra_args.show_gui and "--headless" not in sys.argv:
    sys.argv.append("--headless")

from vla_sim.isaac_app import boot_app, log

app = boot_app()
_app_closed = False


def close_app_once():
    """Close Isaac Sim once; some Kit versions dislike duplicate closes."""
    global _app_closed
    if _app_closed:
        return
    _app_closed = True
    try:
        app.close(wait_for_replicator=False)
    except TypeError:
        app.close()


import numpy as np
import torch
from PIL import Image

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from vla_sim.config import (
    PHYSICS_DT,
    ROBOT_BASE_POS,
    ROBOT_BASE_ROT,
    HOME_Q,
    GRIPPER_OPEN,
    TARGETS,
    GRIPPER_MIMIC_MAP,
    YCB_NUCLEUS_PATH,
)
from vla_sim.scene import (
    enable_extensions,
    spawn_raw_and_assemble,
    SceneCfg,
    apply_scene_colors,
    hide_proxy_meshes,
)


np.random.seed(_extra_args.seed)

# Capture-only object tweaks. Edit these values directly; config.py is untouched.
BANANA_POS = (0.475, 0.30, 1.08)
BANANA_ROT = TARGETS["banana"].get("spawn_rot", (1.0, 0.0, 0.0, 0.0))

REPLACE_RED_MUG_WITH_BOWL = True
BOWL_PRIM_PATH = "/World/CaptureBowl"
BOWL_USD_RELATIVE = f"{YCB_NUCLEUS_PATH}/024_bowl.usd"
BOWL_POS = (0.125, 0.30, 1.105)
BOWL_ROT_X_DEG = -90.0
BOWL_SCALE = (1.0, 1.0, 1.0)

# Capture-only front camera. It looks from the +Y side toward the object row.
CAPTURE_CAMERA_POS = (0.30, 0.66, 1.28)
CAPTURE_CAMERA_TARGET = (0.30, 0.30, 1.13)
CAPTURE_CAMERA_FOCAL = 14.0
CAPTURE_CAMERA_WIDTH = 2560
CAPTURE_CAMERA_HEIGHT = 1440


def quat_wxyz_from_matrix(rot):
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray((w, x, y, z), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def look_at_opengl_quat(camera_pos, target_pos):
    pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)
    forward = target - pos
    forward /= np.linalg.norm(forward)
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rot = np.column_stack((right, up, -forward))
    return quat_wxyz_from_matrix(rot)


CAPTURE_CAMERA_ROT = look_at_opengl_quat(CAPTURE_CAMERA_POS, CAPTURE_CAMERA_TARGET)


@configclass
class CaptureSceneCfg(SceneCfg):
    camera_main = CameraCfg(
        prim_path="/World/CameraMain",
        update_period=0.0,
        height=CAPTURE_CAMERA_HEIGHT,
        width=CAPTURE_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAPTURE_CAMERA_FOCAL),
        offset=CameraCfg.OffsetCfg(
            pos=CAPTURE_CAMERA_POS,
            rot=CAPTURE_CAMERA_ROT,
            convention="opengl",
        ),
    )


def attach_ycb_visual_meshes():
    """Same visual attach loop used by collect_demos.py."""
    from isaacsim.core.utils.stage import add_reference_to_stage

    log("Attaching YCB visual meshes...")
    for target_key, info in TARGETS.items():
        if info.get("collision_usd"):
            log(f"  {target_key}: using local collision USD, skip visual attach")
            continue
        usd_abs = f"{ISAAC_NUCLEUS_DIR}/{info['usd_relative']}"
        visual_path = f"/World/{target_key.capitalize()}/Visuals"
        try:
            add_reference_to_stage(usd_path=usd_abs, prim_path=visual_path)
            log(f"  {target_key}: visual attached at {visual_path}")
        except Exception as exc:
            log(f"  attach failed ({target_key}): {exc}")


def set_prim_visibility(stage, prim_path: str, visible: bool) -> None:
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    token = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    for prim in Usd.PrimRange(root):
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.GetVisibilityAttr().Set(token)


def add_capture_bowl() -> None:
    if not REPLACE_RED_MUG_WITH_BOWL:
        return

    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom
    import omni.client as _client

    usd_abs = f"{ISAAC_NUCLEUS_DIR}/{BOWL_USD_RELATIVE}"
    result, _ = _client.stat(usd_abs)
    log(f"  bowl: stat={result}, usd={usd_abs}")
    try:
        add_reference_to_stage(usd_path=usd_abs, prim_path=BOWL_PRIM_PATH)
    except Exception as exc:
        log(f"  bowl attach failed: {exc}")
        return

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(BOWL_PRIM_PATH)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*BOWL_POS))
    xform.AddOrientOp().Set(
        Gf.Quatf(
            math.cos(math.radians(BOWL_ROT_X_DEG) / 2.0),
            math.sin(math.radians(BOWL_ROT_X_DEG) / 2.0),
            0.0,
            0.0,
        )
    )
    xform.AddScaleOp().Set(Gf.Vec3f(*BOWL_SCALE))
    set_prim_visibility(stage, "/World/Red_mug", visible=False)
    log(f"Replaced red_mug with bowl at {BOWL_POS}")


def command_collect_demo_home(robot, device: str):
    """Return a hold-home command matching collect_demos.py."""
    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    arm_ids, _ = robot.find_joints(arm_joint_names)
    arm_ids_t = torch.tensor(arm_ids, dtype=torch.long, device=device)
    home_q_t = torch.tensor([HOME_Q], device=device, dtype=torch.float32)

    gripper_joint_ids = []
    gripper_joint_cfg = []
    for joint_name, (sign, lo, hi) in GRIPPER_MIMIC_MAP.items():
        ids, _ = robot.find_joints([joint_name])
        if ids:
            gripper_joint_ids.append(ids[0])
            gripper_joint_cfg.append((sign, lo, hi))
    finger_ids_t = torch.tensor(gripper_joint_ids, dtype=torch.long, device=device)
    gripper_signs = torch.tensor(
        [cfg[0] for cfg in gripper_joint_cfg], dtype=torch.float32, device=device
    )
    gripper_lows = torch.tensor(
        [cfg[1] for cfg in gripper_joint_cfg], dtype=torch.float32, device=device
    )
    gripper_highs = torch.tensor(
        [cfg[2] for cfg in gripper_joint_cfg], dtype=torch.float32, device=device
    )
    open_cmd = torch.clamp(
        (gripper_signs * GRIPPER_OPEN).unsqueeze(0),
        gripper_lows.unsqueeze(0),
        gripper_highs.unsqueeze(0),
    )

    def hold_home():
        robot.write_joint_state_to_sim(
            position=home_q_t,
            velocity=torch.zeros_like(home_q_t),
            joint_ids=arm_ids_t,
        )
        robot.set_joint_position_target(home_q_t, joint_ids=arm_ids_t)
        robot.set_joint_position_target(open_cmd, joint_ids=finger_ids_t)

    return hold_home


def resolve_banana_pose():
    pos = np.asarray(BANANA_POS, dtype=np.float32)
    return tuple(float(v) for v in pos), BANANA_ROT


def write_object_pose(scene, name: str, pos, rot, device: str):
    obj = scene[name]
    state = torch.tensor(
        [[*pos, *rot, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    obj.write_root_pose_to_sim(state[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))


def save_camera_image(scene, image_path: Path):
    rgb = scene["camera_main"].data.output["rgb"][0].cpu().numpy().astype(np.uint8)
    Image.fromarray(rgb[..., :3]).save(image_path)


def main():
    out_dir = Path(_extra_args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Output dir : {out_dir}")
    log(f"Num images : {_extra_args.num_images}")
    banana_pos, banana_rot = resolve_banana_pose()

    log(f"Seed       : {_extra_args.seed}")
    log("Setup      : collect_demos scene setup, no trajectory loop")
    log(f"Camera pos : {CAPTURE_CAMERA_POS}")
    log(f"Camera tgt : {CAPTURE_CAMERA_TARGET}")
    log(f"Banana pos : {banana_pos}")

    enable_extensions()
    log("Phase 1: spawn + Robot Assembler")
    spawn_raw_and_assemble()

    log("Phase 2: SimulationContext + InteractiveScene")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT))
    scene_cfg = CaptureSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    attach_ycb_visual_meshes()
    add_capture_bowl()

    log("Phase 3: sim.reset() + sim.play()")
    sim.reset()
    sim.play()
    sim_dt = sim.get_physics_dt()

    stage = omni.usd.get_context().get_stage()
    apply_scene_colors(stage)
    hide_proxy_meshes(stage, TARGETS.keys())

    robot = scene["robot"]
    device = str(sim.device)

    root_pose = torch.tensor(
        [[*ROBOT_BASE_POS, *ROBOT_BASE_ROT]],
        device=device,
        dtype=torch.float32,
    )
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    hold_home = command_collect_demo_home(robot, device)
    write_object_pose(scene, "banana", banana_pos, banana_rot, device)

    captures = []
    for image_idx in range(_extra_args.num_images):
        for _ in range(max(1, _extra_args.render_steps)):
            hold_home()
            write_object_pose(scene, "banana", banana_pos, banana_rot, device)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        image_path = out_dir / f"objects_{image_idx:04d}.png"
        save_camera_image(scene, image_path)
        captures.append({"image": image_path.name, "camera": "camera_main"})
        log(f"Saved {image_path}")

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "num_images": len(captures),
                "seed": _extra_args.seed,
                "setup": "collect_demos_without_trajectory",
                "banana_pos": list(banana_pos),
                "banana_rot_wxyz": list(banana_rot),
                "replace_red_mug_with_bowl": bool(REPLACE_RED_MUG_WITH_BOWL),
                "bowl_pos": list(BOWL_POS),
                "bowl_rot_x_deg": float(BOWL_ROT_X_DEG),
                "camera_pos": list(CAPTURE_CAMERA_POS),
                "camera_target": list(CAPTURE_CAMERA_TARGET),
                "camera_focal": float(CAPTURE_CAMERA_FOCAL),
                "camera_width": int(CAPTURE_CAMERA_WIDTH),
                "camera_height": int(CAPTURE_CAMERA_HEIGHT),
                "captures": captures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"Wrote {meta_path}")
    log("Capture complete. Exiting process.")
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        close_app_once()
