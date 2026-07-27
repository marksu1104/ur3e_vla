# UR3e VLA Simulation

This project uses one canonical Isaac Lab scene for scripted remote tasks, H5
collection, VLA rollout, and read-only real-to-sim arm synchronization.

## Canonical Scene

The scene contains `spoon`, `red_mug`, and `bowl`, the official Robotiq 2F-140
physics asset, three hidden placement markers, a YOLO camera, a policy camera,
and a wrist data camera. `camera_yolo` is reserved for viewing, streaming, and
YOLO. `camera_policy` is the image supplied to OpenVLA.

All canonical gripper requests use logical values: `0.0` = open and `1.0` =
closed. `RobotController` alone maps this to the Robotiq `finger_joint`.

Canonical H5 data is marked `scene_profile=canonical_scene_v1`. Do not mix it
with the earlier mug H5 datasets: the scene, objects, and physical gripper
representation are different.

The banana collision asset and blue-cup files (`cup.usda`, `cup.usdc`) are
retained as reserves but are not loaded by the canonical scene.

## Entry Points

```text
scripts/run_remote_pick_place.py  Canonical scripted task + HTTP/WebSocket bridge.
scripts/run_scene.py              Canonical scene without bridge, VLA, or ROS.
scripts/run_vla.py                OpenVLA rollout using camera_policy.
scripts/collect_demos.py          Canonical 5 Hz H5 collection.
scripts/collect_demos_multi_env.py  Vectorized canonical H5 collection.
scripts/sync_sim_real.py          Persistent real/Isaac joint synchronization.
scripts/tools/test_bridge_client.py  Manual bridge validation client.
```

Maintenance and validation utilities, including the assembled-USD exporter, are
under `scripts/tools/`. Both collectors use the canonical objects, policy camera,
planning, and Robotiq joint settings.

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

Start the persistent scene and bridge on port 8100:

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
  --output-dir ~/IsaacLab/ur3e_vla/outputs/h5/canonical_scene_red_mug \
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

Run read-only physical-to-virtual joint synchronization. It subscribes to `/joint_states`,
never publishes robot motion, holds its last pose after a stale timeout, and
keeps the YOLO bridge stream available on port 8100.

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/sync_sim_real.py \
  --headless --enable_cameras \
  --joint-states-topic /joint_states \
  --joint-state-timeout 0.5
```

For bridge endpoints and Unity integration, see
[docs/remote_bridge.md](docs/remote_bridge.md). More command variants are in
[docs/command_reference.md](docs/command_reference.md). A complete ordered
verification checklist is in
[docs/full_test_commands.md](docs/full_test_commands.md).
The currently supported sim-to-real task and reset procedure are documented in
[docs/sim_real_sync.md](docs/sim_real_sync.md).

## Development Notes

- Keep generated data and models under `outputs/`; it is ignored by Git.
- Put smoke-test artifacts under `outputs/test/`.
- Do not commit model exports or generated H5 data.
- `bridge.py` is the persistent transport contract. Do not replace it
  with the manual Python client in production.
