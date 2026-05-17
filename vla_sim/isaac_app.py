"""Isaac Lab application startup helpers.

This module must be imported before other Isaac Lab or Omni modules in scripts
that need a SimulationApp. The workarounds here are isolated so the rest of the
project can stay focused on scene setup and control logic.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from vla_sim.config import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_RECORD_SECONDS,
    DEFAULT_ORBIT_SNAPSHOTS,
    DEFAULT_ORBIT_VIDEO_FRAMES,
    TARGETS,
)


def log(msg: str):
    print(f"[DBG] {msg}", flush=True)


# Isaac Lab checks sys.argv for this flag before AppLauncher applies settings.
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")


def parse_cli_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--record-seconds", type=int, default=DEFAULT_RECORD_SECONDS)
    parser.add_argument("--camera-fps",     type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument(
        "--target", type=str, default="banana", choices=list(TARGETS.keys()),
        help="Which YCB object to grasp",
    )
    parser.add_argument(
        "--orbit-snapshots", type=int, default=DEFAULT_ORBIT_SNAPSHOTS,
        help="Number of orbit snapshots to capture after trajectory (0 = skip)",
    )
    parser.add_argument(
        "--orbit-video-frames", type=int, default=DEFAULT_ORBIT_VIDEO_FRAMES,
        help="Number of frames in orbit video (0 = skip video)",
    )
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    return args


args_cli = parse_cli_args()

# IsaacLab 3.0 beta kit files can miss extensions in Isaac Sim 6.0.
args_cli.experience = "isaacsim.exp.full.kit"


_launcher = None
_app = None


def boot_app():
    """Start SimulationApp and apply IsaacLab compatibility settings."""
    global _launcher, _app

    if _app is not None:
        return _app

    _launcher = AppLauncher(args_cli)
    _app = _launcher.app

    import carb
    settings = carb.settings.get_settings()

    # The full kit may not set Isaac asset roots. Set them before importing
    # modules that resolve ISAAC_NUCLEUS_DIR.
    asset_url = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/5.1"
    )
    for key in ("/persistent/isaac/asset_root/cloud",
                "/persistent/isaac/asset_root/default",
                "/persistent/isaac/asset_root/nvidia"):
        settings.set(key, asset_url)
    print(f"[DBG] Asset root set to: {asset_url}")

    settings.set_bool("/isaaclab/cameras_enabled", True)
    print(f"[DBG] /isaaclab/cameras_enabled = "
          f"{settings.get('/isaaclab/cameras_enabled')}")

    return _app
