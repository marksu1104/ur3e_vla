# UR3e VLA Simulation

This project uses one canonical Isaac Lab scene for scripted remote tasks, H5
collection, VLA rollout, and read-only real-to-sim arm mirroring.

## Canonical Scene

The scene contains `spoon`, `red_mug`, and `bowl`, the official Robotiq 2F-140
physics asset, three hidden placement markers, a YOLO camera, a policy camera,
and a wrist data camera. `camera_yolo` is reserved for viewing, streaming, and
YOLO. `camera_policy` is the image supplied to OpenVLA.

All canonical gripper requests use logical values: `0.0` = open and `1.0` =
closed. `RobotController` alone maps this to the Robotiq `finger_joint`.

Canonical H5 data is marked `scene_profile=canonical_remote_v1`. Do not mix it
with the earlier mug H5 datasets: the scene, objects, and physical gripper
representation are different.

## Entry Points

```text
scripts/run_remote_pick_place.py  Canonical scripted task + HTTP/WebSocket bridge.
scripts/run_scene.py              Canonical scene without bridge, VLA, or ROS.
scripts/run_vla.py                OpenVLA rollout using camera_policy.
scripts/collect_demos.py          Canonical 5 Hz H5 collection.
scripts/run_real_mirror.py        Read-only /joint_states → virtual UR3e mirror.
scripts/test_bridge_client.py     Manual bridge validation client, not a runtime dependency.
```

`scripts/multi_env/` and `scripts/capture_yolo_objects.py` are retained as
experimental legacy tools. They still use the earlier scene/mimic-gripper
implementation and are not part of the canonical workflow.

## Common Environment

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_ros2
cd ~/IsaacLab
```

## Canonical Commands

Start the persistent remote scene and local bridge on port 8100:

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/run_remote_pick_place.py \
  --headless --enable_cameras --seed 42
```

Collect canonical red-mug H5 demonstrations at 5 Hz. Use a new directory;
`--overwrite` is required when reusing it.

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 500 --max-episodes-tried 700 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/h5/canonical_remote_red_mug \
  --overwrite
```

Run VLA against the canonical scene. `camera_policy` is the only policy camera.

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/run_vla.py \
  --headless --enable_cameras \
  --target red_mug \
  --instruction "pick up the red mug" \
  --vla-server http://localhost:8000 \
  --camera camera_policy \
  --vla-step-interval 12 --action-scale 0.5 --max-steps 6000
```

Run a read-only physical-to-virtual mirror. It subscribes to `/joint_states`,
never publishes robot motion, holds its last pose after a stale timeout, and
keeps the YOLO bridge stream available on port 8100.

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/run_real_mirror.py \
  --headless --enable_cameras \
  --joint-states-topic /joint_states \
  --joint-state-timeout 0.5
```

For bridge endpoints and Unity integration, see
[docs/remote_bridge.md](docs/remote_bridge.md). More command variants are in
[docs/command_reference.md](docs/command_reference.md).

## Development Notes

- Keep generated data and models under `outputs/`; it is ignored by Git.
- Put smoke-test artifacts under `outputs/test/`.
- Do not commit model exports or generated H5 data.
- `remote_bridge.py` is the persistent transport contract. Do not replace it
  with the manual Python client in production.
