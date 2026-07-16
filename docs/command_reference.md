# Command Reference

This file keeps the explicit terminal commands for normal runs and custom experiments. Copy and edit these commands directly when testing changes.

Run most project commands from:

```bash
cd ~/IsaacLab/ur3e_vla
```

## Environment Blocks

ROS + IsaacLab environment:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab
```

OpenVLA training/server environment:

```bash
conda activate openvla
cd ~/vla_ws/openvla/openvla
```

RLDS builder environment:

```bash
conda activate rlds_env
cd ~/IsaacLab/ur3e_vla/rlds_builder/ur3e_vla_dataset
```

## Collect Demos

Raw command for one target:

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

For quick tests, write into `outputs/test/` instead of curated H5 folders:

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
  --target red_mug \
  --episodes 1 \
  --max-episodes-tried 3 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/h5 \
  --overwrite \
  --enable_cameras
```

Useful collect variations:

```bash
# Experimental only: sample one of the configured camera presets per episode.
--randomize-camera-view

# Fixed object pose/light for debugging.
--no-randomize-pos --no-randomize-rot --no-randomize-light

# Keep Isaac open after collection.
--keep-sim-alive

# Show GUI.
--show-gui

# Change random seed.
--seed 123
```

## Capture YOLO Object Views

Capture static RGB images for AR/YOLO checks using the `collect_demos.py` scene
setup without running the grasp trajectory. The script keeps the robot at the
collect-demo home pose, replaces the red mug with a capture-only bowl, pins the
banana pose, and saves 2K `camera_main` PNGs. Edit the capture-only constants in
`scripts/capture_yolo_objects.py` to tune object and camera placement.

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab

./isaaclab.sh -p ./ur3e_vla/scripts/capture_yolo_objects.py \
  --num-images 12 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/yolo_object_views \
  --enable_cameras
```

Useful capture variations:

```bash
--show-gui
--render-steps 90
--seed 123
```

## Create 100-Demo Subset

Raw commands:

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

## Build RLDS / TFDS

Raw command for `mugs_500`:

```bash
conda activate rlds_env
cd ~/IsaacLab/ur3e_vla/rlds_builder/ur3e_vla_dataset

export UR3E_VLA_H5_PATHS=~/IsaacLab/ur3e_vla/outputs/h5/mugs_500/red_mug/demos.h5,~/IsaacLab/ur3e_vla/outputs/h5/mugs_500/blue_mug/demos.h5
export UR3E_VLA_VAL_RATIO=0.1
export UR3E_VLA_SPLIT_SEED=42

tfds build --overwrite --data_dir ~/IsaacLab/ur3e_vla/outputs/tfds/mugs_500
```

For `mugs_100`, change both H5 paths and `--data_dir` to `mugs_100`.

## Fine-Tune OpenVLA

Raw command for `mugs_500`:

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

Common training knobs:

```bash
--max_steps 1000
--learning_rate 5e-5
--shuffle_buffer_size 2000
--image_aug True
```

## Serve Model

Raw command:

```bash
conda activate openvla
cd ~/IsaacLab/ur3e_vla

python3 scripts/vla_server.py \
  --model-path ~/IsaacLab/ur3e_vla/outputs/models/ur3e_vla_mugs_500 \
  --unnorm-key ur3e_vla_dataset \
  --host 0.0.0.0 \
  --port 8000
```

## Sim Rollout

Raw command:

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

Useful sim rollout variations:

```bash
--target blue_mug
--camera camera_wrist
--action-scale 0.5
--vla-step-interval 18
--lock-orientation
--max-steps 12000
```

## Real Robot VLA Servo

Raw dry-run command:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab/ur3e_vla

python3 scripts/real_vla_servo.py \
  --image-topic /image_raw \
  --instruction "pick up the red mug" \
  --vla-server http://localhost:8000 \
  --unnorm-key ur3e_vla_dataset \
  --duration 10 \
  --query-rate-hz 1 \
  --publish-rate-hz 20 \
  --linear-gain 1 \
  --max-linear-speed 0.05 \
  --max-angular-speed 0 \
  --angular-gain 0 \
  --diagnostics
```

Add motion only after real robot setup is healthy:

```bash
--enable-motion
```

Useful real-run variations:

```bash
--instruction "pick up the blue mug"
--duration 30
--linear-gain 2
--max-linear-speed 0.10
--max-angular-speed 0.02
--angular-gain 0.5
--image-topic /image_raw
```

## Camera Checks

```bash
ros2 topic list | grep -E "image|camera"
ros2 topic hz /image_raw
ros2 topic echo /image_raw --once
ros2 run rqt_image_view rqt_image_view
```

## Servo Checks

```bash
ros2 topic echo /servo_node/status --once
ros2 control list_controllers
ros2 param get /servo_node moveit_servo.command_out_topic
ros2 param get /servo_node moveit_servo.command_out_type
ros2 topic echo /forward_velocity_controller/commands --once
ros2 run tf2_ros tf2_echo base_link tool0
```


## Preview Scripted Trajectory Video

Run one scripted episode, record a 2K `camera_main` MP4, and skip H5 writing.
This is for visual inspection only.

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
  --video-path ~/IsaacLab/ur3e_vla/outputs/test/videos/red_mug_trajectory_2k.mp4 \
  --video-camera camera_main \
  --video-width 2560 \
  --video-height 1440 \
  --video-fps 30 \
  --video-every-n-steps 2 \
  --enable_cameras
```
