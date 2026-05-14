# UR3e VLA Simulation

This project runs UR3e grasping experiments in Isaac Lab, collects scripted
demonstrations, and controls the simulated robot through an OpenVLA inference
server.

## Layout

```text
scripts/
  collect_demos.py   Collect scripted demonstration episodes.
  run_vla.py         Run the Isaac Lab simulation with VLA actions.
  vla_server.py      Serve OpenVLA inference over HTTP.

vla_sim/
  config.py          Shared constants for targets, robot, cameras, and limits.
  isaac_app.py       Isaac Lab AppLauncher setup and compatibility settings.
  scene.py           Scene, asset, camera, and target configuration.
  actions.py         Action contract, pose deltas, and scripted trajectories.
  data_collector.py  Episode buffer and dataset file export.
  vla_client.py      Lightweight HTTP client for the VLA server.
```

Generated data and debug outputs stay local under `data/`, `raw/`, and
`outputs/`; they are ignored by Git.

## Start the VLA Server

Run this in the OpenVLA environment:

```bash
conda activate openvla
cd ~/IsaacLab/ur3e_vla
python scripts/vla_server.py --model-path openvla/openvla-7b --port 8000
```

Wait until the log prints `[VLA Server] Ready.`.

If the model has already been downloaded to a local directory, replace
`openvla/openvla-7b` with that directory path.

## Run VLA Control

Run this in the Isaac Lab environment:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab
./isaaclab.sh -p ./ur3e_vla/scripts/run_vla.py \
    --target mug \
    --instruction "pick up the red mug" \
    --vla-server http://localhost:8000 \
    --action-scale 0.1 \
    --max-steps 600
```

## Collect Demonstrations

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab
./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
    --target mug \
    --episodes 5 \
    --output-dir outputs/data \
    --seed 42 \
    --show-gui
```

The default output path is target-based. Successful episodes are appended to a
single HDF5 file, for example:

```text
outputs/data/mug/demos.h5
outputs/data/banana/demos.h5
```

Each episode group stores:

```text
/data/demo_N/image
/data/demo_N/other/hand_image
/data/demo_N/robot_state
/data/demo_N/action
/data/demo_N/task
```

Only successful demonstrations are written to the file. Each `demo_N` group has
a `success=True` attribute for bookkeeping.

Add `--show-gui` when you want to inspect the simulation visually while
collecting a small debug run.
