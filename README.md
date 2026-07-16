# UR3e VLA Simulation

This project runs UR3e mug-grasping experiments in Isaac Lab, collects scripted
demonstrations, builds RLDS/TFDS datasets, fine-tunes OpenVLA, runs simulated
rollouts, and tests a sim-trained policy on the real UR3e through MoveIt Servo.

## Current Scope

The current stable training scope is mug-only:

```text
red_mug
blue_mug
```

`banana` is kept as a future experiment but should not be included in the current
OpenVLA fine-tuning dataset. Collection and rollout use the same policy rate:

```text
12 sim steps per action at 60 Hz = 5 Hz
```

Official collection uses the fixed standard `camera_main` view. Camera-view
randomization is experimental and only enabled when explicitly requested with
`--randomize-camera-view`.

## Project Layout

```text
scripts/
  collect_demos.py        Collect scripted H5 demonstrations.
  run_vla.py              Run Isaac Lab simulation with VLA actions.
  real_vla_servo.py       Send real-camera VLA twist commands through MoveIt Servo.
  check_vla_prediction.py Compare server predictions with recorded H5 actions.
  subset_h5_demos.py      Create smaller H5 subsets.
  export_h5_frames.py     Export recorded H5 images or videos for inspection.
  capture_yolo_objects.py Capture static object-reference images.
  make_usd_collision_asset.py  Rebuild local collision USD assets.
  vla_server.py           Serve OpenVLA inference over HTTP.
  multi_env/              Experimental multi-env collection and assembled-USD tools.

docs/
  command_reference.md          Complete customizable command examples.
  project_layout.md             Artifact/storage policy.
  real_robot_vla_runbook.md     Real UR3e test procedure.

vla_sim/
  isaac_app.py            Shared Isaac Lab startup and shutdown handling.
  config.py               Shared target, robot, camera, and limit config.
  scene.py                Isaac scene and assets.
  actions.py              Scripted trajectories and action conversion.
  geometry.py             Quaternion and angle helpers.
  demo_planning.py        Scene randomization, grasp planning, and success checks.
  data_collector.py       Episode buffers and H5 export.
  vla_client.py           Lightweight VLA HTTP client.
  video.py                MP4 recording helper for trajectory previews.
```

Generated data, TFDS builds, media, and model exports stay under `outputs/`,
which is ignored by Git. Temporary checks belong in `outputs/test/`; curated
training data belongs in `outputs/h5/`. See `docs/project_layout.md`.
Raw command templates for normal runs and custom experiments are in `docs/command_reference.md`.

## Official Workflow

The commands below are the recommended baseline flow. They are written out
explicitly so paths and parameters stay visible. For custom experiments and
extra variants, see [docs/command_reference.md](docs/command_reference.md).

Do not add `--randomize-camera-view` to the baseline commands below. This keeps
collection aligned with the fixed camera used by simulated rollout.

### 1. Collect 500-Demo Mug Datasets

Collect red mug:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab

./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
  --target red_mug \
  --episodes 500 \
  --max-episodes-tried 700 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/h5/mugs_500 \
  --overwrite \
  --enable_cameras
```

Collect blue mug:

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
  --target blue_mug \
  --episodes 500 \
  --max-episodes-tried 700 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/h5/mugs_500 \
  --overwrite \
  --enable_cameras
```

### Preview Trajectory Video Without Saving H5

Use this when you only want to inspect the scripted trajectory. `--no-save-h5`
prevents writing `demos.h5`; `--record-video` temporarily raises `camera_main`
to 2K for the MP4 only. Normal dataset collection still uses the configured H5
camera resolution.

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab

./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
  --target red_mug \
  --episodes 1 \
  --max-episodes-tried 3 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/trajectory_preview \
  --no-save-h5 \
  --record-video \
  --video-path ~/IsaacLab/ur3e_vla/outputs/test/videos/red_mug_trajectory.mp4 \
  --video-camera camera_main \
  --video-width 2560 \
  --video-height 1440 \
  --video-fps 60 \
  --video-every-n-steps 1 \
  --enable_cameras
```

For blue mug, change `--target blue_mug` and use a blue-mug video filename.

### 2. Create 100-Demo Subsets

```bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab/ur3e_vla

python3 scripts/subset_h5_demos.py \
  --input outputs/h5/mugs_500/red_mug/demos.h5 \
  --output outputs/h5/mugs_100/red_mug/demos.h5 \
  --episodes 100 \
  --seed 42 \
  --overwrite

python3 scripts/subset_h5_demos.py \
  --input outputs/h5/mugs_500/blue_mug/demos.h5 \
  --output outputs/h5/mugs_100/blue_mug/demos.h5 \
  --episodes 100 \
  --seed 42 \
  --overwrite
```

### 3. Build TFDS

For the 500-demo dataset:

```bash
conda activate rlds_env
cd ~/IsaacLab/ur3e_vla/rlds_builder/ur3e_vla_dataset

export UR3E_VLA_H5_PATHS=~/IsaacLab/ur3e_vla/outputs/h5/mugs_500/red_mug/demos.h5,~/IsaacLab/ur3e_vla/outputs/h5/mugs_500/blue_mug/demos.h5
export UR3E_VLA_VAL_RATIO=0.1
export UR3E_VLA_SPLIT_SEED=42

tfds build --overwrite --data_dir ~/IsaacLab/ur3e_vla/outputs/tfds/mugs_500
```

For the 100-demo dataset, use `outputs/h5/mugs_100/...` and
`--data_dir ~/IsaacLab/ur3e_vla/outputs/tfds/mugs_100`.

### 4. Fine-Tune OpenVLA

For the 500-demo model:

```bash
conda activate openvla
cd ~/vla_ws/openvla/openvla

WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ~/IsaacLab/ur3e_vla/outputs/tfds/mugs_500 \
  --dataset_name ur3e_vla_dataset \
  --run_root_dir ~/IsaacLab/ur3e_vla/outputs/models/runs \
  --adapter_tmp_dir ~/IsaacLab/ur3e_vla/outputs/models/checkpoints \
  --final_model_dir ~/IsaacLab/ur3e_vla/outputs/models/ur3e_vla_mugs_500 \
  --batch_size 1 \
  --max_steps 2000 \
  --save_steps 250 \
  --learning_rate 1e-4 \
  --shuffle_buffer_size 1000 \
  --image_aug False
```

For the 100-demo model, change `mugs_500` to `mugs_100` and output to
`outputs/models/ur3e_vla_mugs_100`.

### 5. Serve Model

```bash
conda activate openvla
cd ~/IsaacLab/ur3e_vla

python3 scripts/vla_server.py \
  --model-path ~/IsaacLab/ur3e_vla/outputs/models/ur3e_vla_mugs_500 \
  --unnorm-key ur3e_vla_dataset \
  --host 0.0.0.0 \
  --port 8000
```

### 6. Run Sim VLA

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
  --action-scale 1.0 \
  --vla-step-interval 12 \
  --no-lock-orientation \
  --max-steps 6000
```

Use `--target blue_mug` and `--instruction "pick up the blue mug"` for blue mug.

Temporary or smoke-test outputs should go under `outputs/test/`, not the curated
`outputs/h5/`, `outputs/tfds/`, or `outputs/models/` folders.

## Real Robot Test

Detailed setup is in `docs/real_robot_vla_runbook.md`. The current real robot
test is reaching/alignment only because the real arm has no gripper installed.
`real_vla_servo.py` logs the VLA gripper output but ignores it.

Start with dry-run commands from `docs/command_reference.md`. Add
`--enable-motion` to `scripts/real_vla_servo.py` only after returning the robot
to the fixed teach-pendant initial pose and confirming Servo diagnostics are
healthy.

## Development Notes

- Use H5 as the source of truth; TFDS can be rebuilt from H5.
- Do not commit `outputs/` or model exports.
- Keep Python functionality in `scripts/`.
- Keep terminal command templates in `docs/command_reference.md`.
- Keep real robot setup notes in `docs/real_robot_vla_runbook.md`.
