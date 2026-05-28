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
    --target red_mug \
    --instruction "pick up the red mug" \
    --vla-server http://localhost:8000 \
    --camera camera_main \
    --unnorm-key ur3e_vla_dataset \
    --action-scale 0.5 \
    --lock-orientation \
    --max-steps 6000
```

## Collect Demonstrations

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab
./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
    --target red_mug \
    --episodes 5 \
    --output-dir outputs/data \
    --seed 42 \
    --overwrite \
    --show-gui
```

The default output path is target-based. Successful episodes are appended to a
single HDF5 file, for example:

```text
outputs/data/red_mug/demos.h5
outputs/data/blue_mug/demos.h5
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

## Build RLDS From HDF5

The local TFDS builder in `rlds_builder/ur3e_vla_dataset` reads the HDF5 file directly. It does not require an intermediate `episode_*.npy` export.

```bash
conda activate rlds_env
cd ~/IsaacLab/ur3e_vla/rlds_builder/ur3e_vla_dataset

export UR3E_VLA_H5_PATH=~/IsaacLab/ur3e_vla/outputs/data/red_mug/demos.h5
export UR3E_VLA_VAL_RATIO=0.1
export UR3E_VLA_SPLIT_SEED=42

tfds build --overwrite
```

The builder emits RLDS steps with:

```text
observation/image
observation/hand_image
observation/state
action  # 8D in TFDS: 7D policy action + terminal flag; OpenVLA transform drops the flag.
language_instruction
language_embedding
is_first
is_last
is_terminal
reward
discount
```

The generated TFDS dataset is written to:

```text
~/tensorflow_datasets/ur3e_vla_dataset/1.0.0
```

## Prepare OpenVLA Fine-Tuning Data

Copy the generated TFDS dataset into the OpenVLA workspace:

```bash
cp -r ~/tensorflow_datasets/ur3e_vla_dataset ~/vla_ws/openvla/openvla/datasets/
```

OpenVLA must also register `ur3e_vla_dataset` in its RLDS dataset config,
standardization transform registry, and mixture registry:

```text
~/vla_ws/openvla/openvla/prismatic/vla/datasets/rlds/oxe/configs.py
~/vla_ws/openvla/openvla/prismatic/vla/datasets/rlds/oxe/transforms.py
~/vla_ws/openvla/openvla/prismatic/vla/datasets/rlds/oxe/mixtures.py
```

After registration, the dataset can be used with:

```text
--data_root_dir datasets
--dataset_name ur3e_vla_dataset
```

## Fine-Tune OpenVLA

Run this in the OpenVLA environment:

```bash
conda activate openvla
cd ~/vla_ws/openvla/openvla

WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir datasets \
    --dataset_name ur3e_vla_dataset \
    --run_root_dir ./runs \
    --adapter_tmp_dir ./checkpoints \
    --final_model_dir ./exports/ur3e_vla_red_mug_latest \
    --batch_size 1 \
    --max_steps 1000 \
    --save_steps 250 \
    --learning_rate 1e-4 \
    --shuffle_buffer_size 1000 \
    --image_aug False
```

By default, OpenVLA saves the fused fine-tuned model under `runs/` with its
full experiment name. Passing `--final_model_dir` also writes the fused model to
a short, stable inference path:

```text
~/vla_ws/openvla/openvla/exports/ur3e_vla_red_mug_latest
```

Then point `scripts/vla_server.py` to that path:

```bash
conda activate openvla
cd ~/IsaacLab/ur3e_vla
python scripts/vla_server.py \
    --model-path ~/vla_ws/openvla/openvla/exports/ur3e_vla_red_mug_latest \
    --unnorm-key ur3e_vla_dataset \
    --port 8000
```

If `--final_model_dir` is omitted, the original OpenVLA output path under
`runs/` is used.
