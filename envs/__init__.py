"""
grasp_lib — UR3e + Robotiq 2F-140 + YCB grasp 場景的共用模組.

Modules:
  boot      — AppLauncher 啟動 + IsaacLab 3.0 beta workarounds.
  config    — 純資料常數 (TARGETS, GRIPPER_MIMIC_MAP, 尺寸/路徑).
  sim_setup — Phase 1 spawn + RobotAssembler + SceneCfg + helpers.

Import 順序 (重要!):
  1. from grasp_lib.boot import boot_app, args_cli
     app = boot_app()                          # 必須先做這個
  2. 之後才能 import grasp_lib.sim_setup (因為它依賴 isaaclab.* / omni.*)

config 是純資料, 任何時候都能 import.
"""