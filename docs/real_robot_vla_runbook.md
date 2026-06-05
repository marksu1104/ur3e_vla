# UR3e Real Robot VLA Runbook

This runbook describes the current validated flow for testing the sim-trained mug VLA model on the real UR3e. The current goal is **reaching / alignment only**. The robot does not have a real gripper installed, so the VLA gripper output is logged but ignored.

## Current Scope

- Targets: `red_mug`, `blue_mug`
- Input image: one RGB frame from `/image_raw`
- VLA output: 7D action `[dx, dy, dz, droll, dpitch, dyaw, gripper]`
- Real robot control: MoveIt Servo twist command
- Servo topic: `/servo_node/delta_twist_cmds`
- Gripper command: ignored
- Rotation: disabled for the current safest test by setting `--max-angular-speed 0`

## Safety Notes

The VLA model can still produce unstable actions on real camera images. It may switch directions, confuse targets if the scene is poorly arranged, or output close/open phase changes that are not useful without a gripper.

For the current test:

- Always return the robot to the fixed teach-pendant initial pose before running VLA.
- Keep motion speed capped.
- Keep rotation disabled at first.
- Keep your hand near the teach pendant / emergency stop.
- Do not run long tests until short tests move in the expected direction.
- If the robot moves toward the wrong object, stop and adjust cup placement/camera view.
- If diagnostics show NaN controller commands, stop VLA and reset to a better initial pose.

## Terminal A: UR Driver

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash

ros2 launch ur_robot_driver ur3e.launch.py \
  robot_ip:=192.168.10.175 \
  use_mock_hardware:=false \
  initial_joint_controller:=forward_velocity_controller \
  launch_rviz:=false
```

Mock hardware option:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash

ros2 launch ur_robot_driver ur3e.launch.py \
  robot_ip:=yyy.yyy.yyy.yyy \
  use_mock_hardware:=true \
  initial_joint_controller:=forward_velocity_controller \
  launch_rviz:=false
```

## Terminal B: MoveIt Servo + Bridge

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash

ros2 launch vla_ros_bridge vla_bridge.launch.py
```

## Terminal C: Camera Node

The current USB camera publishes 640x480 RGB images on `/image_raw`.

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash

ros2 run v4l2_camera v4l2_camera_node \
  --ros-args \
  -p video_device:=/dev/video0 \
  -p image_size:="[640,480]"
```

Expected image topics:

```text
/image_raw
/camera_info
```

Optional image viewer:

```bash
ros2 run rqt_image_view rqt_image_view
```

## Terminal D: Enforce Controller Path

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash

ros2 control switch_controllers \
  --activate forward_velocity_controller \
  --deactivate forward_position_controller scaled_joint_trajectory_controller

ros2 service call /servo_node/switch_command_type \
  moveit_msgs/srv/ServoCommandType "{command_type: 1}"
```

Verify controller state:

```bash
ros2 control list_controllers
ros2 param get /servo_node moveit_servo.command_out_topic
ros2 param get /servo_node moveit_servo.command_out_type
ros2 topic echo /servo_node/status
```

Expected:

```text
forward_velocity_controller: active
moveit_servo.command_out_topic: /forward_velocity_controller/commands
moveit_servo.command_out_type: std_msgs/Float64MultiArray
```

## Terminal E: VLA Server

Use the currently best mug model. For example, the 500-demo model:

```bash
conda activate openvla
cd ~/IsaacLab/ur3e_vla

python scripts/vla_server.py \
  --model-path ~/IsaacLab/ur3e_vla/outputs/models/ur3e_vla_mugs_500 \
  --unnorm-key ur3e_vla_dataset \
  --host 0.0.0.0 \
  --port 8000
```

Wait until the server prints:

```text
[VLA Server] Ready.
```

## Terminal G: Real Robot VLA Servo Test

Start with a short, no-rotation reaching test. The gripper dimension is ignored.

Red mug:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab/ur3e_vla

python scripts/real_vla_servo.py \
  --image-topic /image_raw \
  --instruction "pick up the red mug" \
  --vla-server http://localhost:8000 \
  --duration 10 \
  --query-rate-hz 1 \
  --publish-rate-hz 20 \
  --linear-gain 1 \
  --max-linear-speed 0.05 \
  --max-angular-speed 0 \
  --angular-gain 0 \
  --diagnostics \
  --enable-motion
```

Blue mug:

```bash
python scripts/real_vla_servo.py \
  --image-topic /image_raw \
  --instruction "pick up the blue mug" \
  --vla-server http://localhost:8000 \
  --duration 10 \
  --query-rate-hz 1 \
  --publish-rate-hz 20 \
  --linear-gain 1 \
  --max-linear-speed 0.05 \
  --max-angular-speed 0 \
  --angular-gain 0 \
  --diagnostics \
  --enable-motion
```

Expected behavior:

- Red instruction should move toward the red mug.
- Blue instruction should move toward the blue mug.
- The robot may not complete a grasp because there is no real gripper.
- If the robot drifts toward the wrong mug, stop and adjust cup placement/camera view.

## Emergency Stop / Zero Command

Publish a zero twist command if needed:

```bash
ros2 topic pub /servo_node/delta_twist_cmds geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: base_link}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}" \
  --once
```

Also keep the physical emergency stop / teach pendant ready.

## Current Observations

- The real camera pipeline works: `/image_raw -> VLA server -> action`.
- Latency has been around 120-150 ms.
- The model reacts differently to red and blue instructions after adjusting cup positions.
- If both mugs are poorly placed, the model may confuse targets.
- On real camera images, gripper output may jump between `0.0` and `0.996`; it is ignored for now.
- Rotation/yaw can be unstable, so the current real test disables angular motion.
- MoveIt Servo may output NaN joint velocities from a poor initial wrist/arm pose. The fixed teach-pendant initial pose is the current workaround.
- Use `--diagnostics` to check `controller=finite=True` before trusting a run.

## Next Improvements

Before longer or more autonomous tests, add:

- workspace boundary checks,
- minimum table-height / z limit,
- action median or temporal smoothing,
- logging to CSV,
- fixed camera placement matching Isaac `camera_main`,
- optional small real-image fine-tuning data.

