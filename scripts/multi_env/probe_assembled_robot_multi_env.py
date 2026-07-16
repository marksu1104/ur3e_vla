"""Probe whether the exported UR3e+Robotiq assembled USD can be used in multi-env.

This script does not collect data.  It references the exported assembled stage
under each IsaacLab env and checks tensor shapes, joint names, and whether the
Robotiq gripper appears inside the robot articulation or only as a separate USD
subtree.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_parser.add_argument("--num-envs", type=int, default=4, help="Number of IsaacLab envs to instantiate.")
_parser.add_argument("--env-spacing", type=float, default=2.0, help="Spacing between env origins.")
_parser.add_argument("--settle-steps", type=int, default=60, help="Simulation steps after reset.")
_parser.add_argument(
    "--asset",
    type=Path,
    default=PROJECT_ROOT / "assets" / "ur3e_robotiq_2f140_assembled_stage.usda",
    help="Assembled stage USD exported by scripts/multi_env/export_assembled_robot_usd.py.",
)
_parser.add_argument(
    "--probe-gripper-articulation",
    action="store_true",
    help="Also try registering /Assembled/Gripper as a separate ArticulationCfg.",
)
_parser.add_argument("--show-gui", action="store_true", help="Run with Isaac Sim GUI.")
_extra_args, _ = _parser.parse_known_args()

if not _extra_args.show_gui and "--headless" not in sys.argv:
    sys.argv.append("--headless")

from vla_sim.isaac_app import boot_app, close_app, log

app = boot_app()


import traceback

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from vla_sim.config import GRIPPER_MIMIC_MAP, PHYSICS_DT, ROBOT_BASE_POS, ROBOT_BASE_ROT

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ASSEMBLED_PRIM = "{ENV_REGEX_NS}/Assembled"
ROBOT_PRIM = f"{ASSEMBLED_PRIM}/Robot"
GRIPPER_PRIM = f"{ASSEMBLED_PRIM}/Gripper"


def _make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=ROBOT_BASE_POS, rot=ROBOT_BASE_ROT),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                stiffness=10000.0,
                damping=500.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
            ),
            "gripper_if_present": ImplicitActuatorCfg(
                joint_names_expr=list(GRIPPER_MIMIC_MAP.keys()),
                stiffness=60.0,
                damping=8.0,
                effort_limit_sim=8.0,
            ),
        },
    )


def _make_gripper_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=GRIPPER_PRIM,
        spawn=None,
        actuators={
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=list(GRIPPER_MIMIC_MAP.keys()),
                stiffness=60.0,
                damping=8.0,
                effort_limit_sim=8.0,
            ),
        },
    )


@configclass
class AssembledRobotProbeCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )
    assembled = AssetBaseCfg(
        prim_path=ASSEMBLED_PRIM,
        spawn=sim_utils.UsdFileCfg(usd_path=str(_extra_args.asset.expanduser().resolve())),
    )
    robot = _make_robot_cfg()


@configclass
class AssembledRobotAndGripperProbeCfg(AssembledRobotProbeCfg):
    gripper = _make_gripper_cfg()


def _shape_of(value):
    try:
        return tuple(value.shape)
    except Exception:
        return None


def _joint_names(asset) -> list[str]:
    names = getattr(asset.data, "joint_names", None)
    if names is None:
        names = getattr(asset, "joint_names", None)
    if names is None:
        return []
    return list(names)


def _log_asset_report(scene, name: str) -> bool:
    try:
        asset = scene[name]
    except Exception as exc:
        log(f"{name}: unavailable: {exc}")
        return False

    names = _joint_names(asset)
    log(f"{name} root_pos_w shape : {_shape_of(asset.data.root_pos_w)}")
    log(f"{name} joint_pos shape  : {_shape_of(asset.data.joint_pos)}")
    log(f"{name} joint names      : {names}")
    log(f"{name} arm joints found : {[n for n in ARM_JOINT_NAMES if n in names]}")
    log(f"{name} gripper found    : {[n for n in GRIPPER_MIMIC_MAP if n in names]}")
    return True


def _log_stage_prims(stage) -> None:
    env0 = stage.GetPrimAtPath("/World/envs/env_0/Assembled")
    log(f"/World/envs/env_0/Assembled valid: {env0.IsValid()}")
    for path in (
        "/World/envs/env_0/Assembled/Robot",
        "/World/envs/env_0/Assembled/Gripper",
        "/World/envs/env_0/Assembled/Gripper/robotiq_arg2f_base_link",
        "/World/envs/env_0/Assembled/Gripper/robotiq_arg2f_base_link/AssemblerFixedJoint",
    ):
        prim = stage.GetPrimAtPath(path)
        log(f"{path}: valid={prim.IsValid()} type={prim.GetTypeName() if prim.IsValid() else 'N/A'}")


def main() -> None:
    asset_path = _extra_args.asset.expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Assembled USD not found: {asset_path}")

    num_envs = max(1, int(_extra_args.num_envs))
    log("Assembled UR3e+Robotiq multi-env probe. No H5 data will be written.")
    log(f"Asset       : {asset_path}")
    log(f"num_envs    : {num_envs}")
    log(f"probe gripper articulation: {_extra_args.probe_gripper_articulation}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=PHYSICS_DT))
    cfg_cls = AssembledRobotAndGripperProbeCfg if _extra_args.probe_gripper_articulation else AssembledRobotProbeCfg
    scene = InteractiveScene(cfg_cls(num_envs=num_envs, env_spacing=float(_extra_args.env_spacing)))

    log("Phase: sim.reset() + sim.play()")
    sim.reset()
    sim.play()
    sim_dt = sim.get_physics_dt()

    for _ in range(max(1, int(_extra_args.settle_steps))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    stage = __import__("omni.usd").usd.get_context().get_stage()
    log("=" * 60)
    log("Stage report")
    _log_stage_prims(stage)

    log("=" * 60)
    log("Tensor report")
    log(f"scene.num_envs       : {getattr(scene, 'num_envs', 'unknown')}")
    log(f"env_origins shape    : {_shape_of(scene.env_origins)}")
    try:
        log(f"env_origins          : {scene.env_origins.detach().cpu().numpy().round(3).tolist()}")
    except Exception as exc:
        log(f"env_origins read failed: {exc}")

    robot_ok = _log_asset_report(scene, "robot")
    gripper_ok = False
    if _extra_args.probe_gripper_articulation:
        gripper_ok = _log_asset_report(scene, "gripper")

    robot = scene["robot"] if robot_ok else None
    vectorized = bool(robot_ok and int(robot.data.root_pos_w.shape[0]) == num_envs)
    robot_joints = _joint_names(robot) if robot_ok else []
    gripper_inside_robot = any(name in robot_joints for name in GRIPPER_MIMIC_MAP)

    log("=" * 60)
    log("Result")
    log(f"robot vectorized          : {vectorized}")
    log(f"gripper joints in robot   : {gripper_inside_robot}")
    log(f"separate gripper works    : {gripper_ok}")
    if vectorized and gripper_inside_robot:
        log("PASS: assembled asset behaves like one vectorized robot articulation.")
    elif vectorized:
        log("PARTIAL: arm is vectorized, but gripper is not inside robot articulation.")
        log("Next step may require treating gripper as a separate articulation or building a cleaner single-root USD.")
    else:
        log("FAIL: assembled asset is not ready for multi-env collection yet.")

    log("Exiting process.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"EXCEPTION: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise
    finally:
        close_app()
