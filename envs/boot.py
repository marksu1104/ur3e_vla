"""
envs.boot — AppLauncher 啟動 + IsaacLab 3.0 beta workarounds.

呼叫順序:
    from envs.boot import boot_app, args_cli, log
    app = boot_app()                              # 啟動 SimulationApp
    # 之後才能 import isaaclab.* / omni.* / carb

提供的 args 包含 IsaacLab 內建的 (--enable_cameras 等) 加上本 script 用的:
    --target              {banana, mug}
    --record-seconds      錄影秒數 (default 30)
    --camera-fps          錄影 fps (default 30)
    --orbit-snapshots     軌跡跑完後拍 N 張環繞照 (default 24, 0=skip)
    --orbit-video-frames  環繞影片 frame 數 (default 240, 0=skip)

⚠️  IsaacLab 3.0 BETA 還在 RC 階段, 需要 4 個 workaround 才能跑起來.
    每個 workaround 都有 # WORKAROUND #N 標記, 等正式版發布後可逐一檢查移除.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from envs.config import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_RECORD_SECONDS,
    DEFAULT_ORBIT_SNAPSHOTS,
    DEFAULT_ORBIT_VIDEO_FRAMES,
    TARGETS,
)


def log(msg: str):
    print(f"[DBG] {msg}", flush=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  IsaacLab 3.0 BETA WORKAROUND #1                                  ║
# ║                                                                   ║
# ║  AppLauncher 在 3.0 beta 必須從 sys.argv 看到 --enable_cameras   ║
# ║  才會設對應的 carb flag。光是 args_cli.enable_cameras = True 不夠.║
# ║  詳見 boot_app() 內的 WORKAROUND #4 (cameras_enabled).            ║
# ╚══════════════════════════════════════════════════════════════════╝
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")


def parse_cli_args():
    parser = argparse.ArgumentParser()
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
    args.headless = False
    args.enable_cameras = True
    return args


# parse 一次, 之後 sim_r.py 等其他 script 直接 import args_cli 用
args_cli = parse_cli_args()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  IsaacLab 3.0 BETA WORKAROUND #2                                  ║
# ║                                                                   ║
# ║  IsaacLab 3.0 beta 自己的 apps/*.kit 還停在 2.3.2 版,            ║
# ║  依賴 omni.kit.property.physx 但 Isaac Sim 6.0 沒這個 ext.       ║
# ║  改用 isaacsim 內建的 isaacsim.exp.full.kit.                    ║
# ╚══════════════════════════════════════════════════════════════════╝
args_cli.experience = "isaacsim.exp.full.kit"

# DEBUG: 確認 args 真的有設好 (出問題時很重要)
print(f"[DBG] enable_cameras = {getattr(args_cli, 'enable_cameras', 'NOT SET')}")
print(f"[DBG] headless       = {getattr(args_cli, 'headless', 'NOT SET')}")
print(f"[DBG] sys.argv       = {sys.argv}")


# 全域 launcher / app, 由 boot_app() 設定
_launcher = None
_app = None


def boot_app():
    """啟動 SimulationApp + 套用所有 carb settings workarounds.

    回傳 SimulationApp 物件 (sim_r.py main 結束時要 app.close()).
    """
    global _launcher, _app

    if _app is not None:
        return _app  # 已經啟動過

    _launcher = AppLauncher(args_cli)
    _app = _launcher.app

    # ── 必須在 isaaclab.utils.assets / isaaclab.sim 之前 set 這些 carb settings,
    #    因為那些 module 在 import 時就會讀 carb settings.
    import carb
    settings = carb.settings.get_settings()

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  IsaacLab 3.0 BETA WORKAROUND #3 — Asset root 為 None        ║
    # ║                                                               ║
    # ║  isaacsim.exp.full.kit 沒設 /persistent/isaac/asset_root/*,  ║
    # ║  導致 ISAAC_NUCLEUS_DIR 變成字面值 "None/Isaac".             ║
    # ║  手動指向 5.1 S3 (6.0 還沒上完整 asset).                     ║
    # ╚══════════════════════════════════════════════════════════════╝
    asset_url = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/5.1"
    )
    for key in ("/persistent/isaac/asset_root/cloud",
                "/persistent/isaac/asset_root/default",
                "/persistent/isaac/asset_root/nvidia"):
        settings.set(key, asset_url)
    print(f"[DBG] Asset root set to: {asset_url}")

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  IsaacLab 3.0 BETA WORKAROUND #4 — cameras_enabled flag      ║
    # ║                                                               ║
    # ║  使用 isaacsim.exp.full.kit (非 IsaacLab 自家 kit) 不會自動  ║
    # ║  設這個 flag, 但 isaaclab/sensors/camera/camera.py:           ║
    # ║  _initialize_impl() 會檢查它, 否則 sim.reset() 階段會丟      ║
    # ║  "A camera was spawned without --enable_cameras".            ║
    # ╚══════════════════════════════════════════════════════════════╝
    settings.set_bool("/isaaclab/cameras_enabled", True)
    print(f"[DBG] /isaaclab/cameras_enabled = "
          f"{settings.get('/isaaclab/cameras_enabled')}")

    return _app