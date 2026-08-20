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

## Scene Customization

The project keeps one physical scene and composes small, named profiles for each
workflow. This prevents a remote-only camera or light adjustment from silently
changing data collection or VLA rollout.

| Change | Authoritative location |
| --- | --- |
| Positions, materials, lighting values, and profile membership | `vla_sim/config.py` |
| Robot, object, light, and camera construction | `vla_sim/scene.py` |
| Render materials, smoothing, and visual-only presentation | `vla_sim/visuals.py` |
| Remote destination/reference props | `vla_sim/fixtures.py` |
| Simulation stepping, controllers, and state backends | `vla_sim/runtime.py` |
| Workflow-only behavior and CLI | the corresponding file under `scripts/` |
| MultiView experiment | `vla_sim/multiview/` and `scripts/collect_demos_multi_view.py` |

Add a reusable camera builder in `scene.py`, then list it only in the profiles
that need it in `config.py`. Give a workflow its own `LightingSettings` when its
lighting must differ. Put shared physical objects in the canonical scene; put
presentation-only destination props in `fixtures.py`. New entry points should
select a profile and reuse the runtime instead of copying a `SceneCfg` or robot
configuration. MultiView remains isolated so its ongoing development is not
changed by routine canonical-scene work.

This follows Isaac Lab's compositional `configclass` and `InteractiveSceneCfg`
model while retaining the Standalone workflow needed for explicit bridge and
robot-control loops. See the official [configuration design](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/api_env_design.html)
and [sensor composition tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/04_sensors/add_sensors_on_robot.html).

## Common Environment

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_ros2
cd ~/IsaacLab
```

## Workstation Baseline

The current project is pinned to the exact workstation stack below. Do not
upgrade or downgrade one component independently without rerunning the complete
smoke-test workflow.

```text
Ubuntu             24.04.4 LTS (kernel 7.0.0-28-generic)
GPU                NVIDIA GeForce RTX 5090, 32607 MiB
NVIDIA driver      595.84
Python             3.12.13
Isaac Sim          6.0.0.0 (pip)
Isaac Lab          0.54.3, editable source at ~/IsaacLab
Isaac Lab commit   d94504bcf91cb7ab7ff956a2d48ecd1bca82797a
PyTorch            2.7.0+cu128
ROS                Jazzy (ros-base 0.11.0)
MoveIt             2.12.4
UR robot driver    3.7.0
```

This is a project-validated compatibility setup rather than the default pairing
documented by that Isaac Lab checkout, whose normal installation guide still
targets Isaac Sim 5.1 and Python 3.11. `vla_sim/isaac_app.py` intentionally uses
the Isaac Sim 6.0 full-kit experience and the Isaac 5.1 cloud asset root. Keep
those compatibility settings until the full scene, camera, collision, data, and
robot checks have passed on a replacement stack.

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

## 3D-Printable Scene Parts

Export the exact visible spoon, mug, bowl, cutlery tray, and coaster geometry as
millimetre-scale STL files:

```bash
cd ~/IsaacLab/ur3e_vla
python3 scripts/tools/export_printable_assets.py --overwrite
```

Files are written to `outputs/printable/` with a `manifest.json` containing
quantity, source, dimensions, triangle counts, and watertight checks. Open the
STLs in a slicer to choose orientation, supports, wall settings, and print
tolerances. The export intentionally uses rendered geometry rather than the
simplified physics collision meshes. Review the source terms recorded in the
manifest before redistributing derived YCB/NVIDIA geometry.

## Development Notes

- Keep generated data and models under `outputs/`; it is ignored by Git.
- Put smoke-test artifacts under `outputs/test/`.
- Do not commit model exports or generated H5 data.
- Treat `SCENE_PROFILES` as workflow composition, not as a second set of scene
  constants. Change a shared physical property once at its authoritative source.
- `bridge.py` is the persistent transport contract. Do not replace it
  with the manual Python client in production.
