"""Isaac Lab application startup helpers.

This module must be imported before other Isaac Lab or Omni modules in scripts
that need a SimulationApp. The workarounds here are isolated so the rest of the
project can stay focused on scene setup and control logic.
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from vla_sim.config import TARGETS


def log(msg: str):
    print(f"[DBG] {msg}", flush=True)


# Isaac Lab checks sys.argv for this flag before AppLauncher applies settings.
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")


def parse_cli_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        "--target", type=str, default="red_mug", choices=list(TARGETS.keys()),
        help="Which YCB object to grasp",
    )
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    return args


args_cli = parse_cli_args()

# Preserve the original Isaac Sim application and its stage/render defaults.
args_cli.experience = "isaacsim.exp.full.kit"


def _use_writable_shader_cache() -> None:
    """Keep runtime RTX caches in the active user's shared-runtime cache.

    Isaac Sim's pip bundle defaults these two writable caches to its package
    directory in portable mode.  That directory is read-only in the shared
    workstation install, causing every camera process to repeat a roughly
    one-minute RTX initialization.  The prebuilt shader caches remain in the
    package; only newly generated runtime entries are redirected here.
    """
    cache_root = os.environ.get("ISAAC_ROS2_KIT_CACHE_DIR")
    if not cache_root:
        return

    shader_cache = Path(cache_root) / "shadercache"
    driver_cache = Path(cache_root) / "nv_shadercache"
    shader_cache.mkdir(parents=True, exist_ok=True)
    driver_cache.mkdir(parents=True, exist_ok=True)

    kit_args = getattr(args_cli, "kit_args", "") or ""
    settings = (
        ("/rtx/shaderDb/shaderCachePath", shader_cache),
        ("/rtx/shaderDb/driverShaderCachePath", driver_cache),
    )
    for setting, path in settings:
        if f"--{setting}=" not in kit_args:
            kit_args = f"{kit_args} --{setting}={path}".strip()
    args_cli.kit_args = kit_args


_use_writable_shader_cache()


_launcher = None
_app = None
_app_closed = False


def boot_app():
    """Start SimulationApp and apply IsaacLab compatibility settings."""
    global _launcher, _app

    if _app is not None:
        return _app

    _launcher = AppLauncher(args_cli)
    _app = _launcher.app

    import carb
    settings = carb.settings.get_settings()

    # Set asset roots before importing modules that resolve ISAAC_NUCLEUS_DIR.
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


def close_app():
    """Close the shared SimulationApp once."""
    global _app_closed
    if _app is None or _app_closed:
        return
    _app_closed = True
    try:
        _app.close(wait_for_replicator=False)
    except TypeError:
        _app.close()
