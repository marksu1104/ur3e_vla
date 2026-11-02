# UR3e Real Robot VLA Runbook

Updated 2026-10-30. The user confirmed on 2026-10-14 that a real gripper is
installed and VLA has been tried on hardware: alignment is possible, but lifting
the object remains difficult. Reliable grasping has not been established. The
commands below describe the existing test interface; this update ran no robot
or ROS session.

## Current Scope

- Historical reaching tests used `red_mug` and `blue_mug`; use instructions and objects supported by the particular trained model. The canonical simulation uses spoon, red mug and bowl.
- Input image: one RGB frame from `/image_raw`
- VLA output: 7D action `[dx, dy, dz, droll, dpitch, dyaw, gripper]`
- Real robot control: MoveIt Servo twist command
- Servo topic: `/servo_node/delta_twist_cmds`
- Gripper command: `action[6] > --gripper-close-threshold` sends a close request through `SetIO`; otherwise it sends open. Default DO pin is 0, configurable with `--gripper-io-pin`. Commands require `--enable-motion` and an available IO service.
- Rotation: disabled for the current safest test by setting `--max-angular-speed 0`

## Safety Notes

The VLA model can produce unstable actions on real camera images. It may
switch directions, confuse targets, or issue close/open requests at the wrong
time. Enabling motion also enables these gripper requests; reaching examples
do not disable the gripper.

For the current test:

- Always return the robot to the fixed teach-pendant initial pose before running VLA.
- Keep motion speed capped.
- Keep rotation disabled at first.
- Keep your hand near the teach pendant / emergency stop.
- Do not run long tests until short tests move in the expected direction.
- If the robot moves toward the wrong object, stop and adjust cup placement/camera view.
- If diagnostics show NaN controller commands, stop VLA and reset to a better initial pose.

Set `ROBOT_IP` for your robot cell before starting the driver. This workstation's
connection values are retained in the Git-ignored `docs/local_commands.md`.

## Terminal A: UR Driver

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

ros2 launch ur_robot_driver ur3e.launch.py \
  robot_ip:="${ROBOT_IP:?Set ROBOT_IP for your robot cell}" \
  use_mock_hardware:=false \
  initial_joint_controller:=forward_velocity_controller \
  launch_rviz:=false
```

Mock hardware option:

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

ros2 launch ur_robot_driver ur3e.launch.py \
  robot_ip:=127.0.0.1 \
  use_mock_hardware:=true \
  initial_joint_controller:=forward_velocity_controller \
  launch_rviz:=false
```

## Terminal B: MoveIt Servo + Bridge

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

ros2 launch vla_ros_bridge vla_bridge.launch.py
```

## Terminal C: Camera Node

The current USB camera publishes 640x480 RGB images on `/image_raw`.

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

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
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

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

Use a model that was trained and separately validated for this real-robot
workflow. The path below is only an example; do not run this test until the
model and its `dataset_statistics.json` exist and pass the simulation checks.

```bash
conda activate openvla
cd ~/IsaacLab/ur3e_vla

MODEL_PATH=/srv/isaac_ros2/models/pick_place_v1
ls "$MODEL_PATH/dataset_statistics.json"

python scripts/vla_server.py \
  --model-path "$MODEL_PATH" \
  --unnorm-key ur3e_vla_dataset \
  --host 0.0.0.0 \
  --port 8000
```

Wait until the server prints:

```text
[VLA Server] Ready.
```

## Terminal G: Real Robot VLA Servo Test

Start with a short, no-rotation test. These commands explicitly enable both
arm motion and gripper IO. Omitting `--enable-motion` keeps the script in dry
run mode. Blue-mug examples are historical and require a model trained for
that object; they are not evidence of canonical-scene model support.

Red mug:

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

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
- A complete grasp and lift are not yet a validated outcome; record failures rather than treating alignment as success.
- If the robot drifts toward the wrong mug, stop and adjust cup placement/camera view.

## Emergency Stop / Zero Command

Publish a zero twist command if needed:

```bash
ros2 topic pub /servo_node/delta_twist_cmds geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: base_link}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}" \
  --once
```

Also keep the physical emergency stop / teach pendant ready.

## Reported and Historical Observations

These are previous observations, not new hardware measurements from this update.
The latest user report is installed gripper, partial alignment and difficult grasping.

- The real camera pipeline works: `/image_raw -> VLA server -> action`.
- Latency has been around 120-150 ms.
- The model reacts differently to red and blue instructions after adjusting cup positions.
- If both mugs are poorly placed, the model may confuse targets.
- On real camera images, gripper output was observed jumping between `0.0` and `0.996`; the current script interprets it as open/close, so verify timing and physical feedback.
- Rotation/yaw can be unstable, so the current real test disables angular motion.
- MoveIt Servo may output NaN joint velocities from a poor initial wrist/arm pose. The fixed teach-pendant initial pose is the current workaround.
- Use `--diagnostics` to check `controller=finite=True` before trusting a run.

## Next Improvements

Before longer or more autonomous tests, add:

- workspace boundary checks,
- minimum table-height / z limit,
- action median or temporal smoothing,
- logging to CSV,
- fixed camera placement and preprocessing matching the training image source (`camera_policy` in the canonical workflow),
- optional small real-image fine-tuning data.

## Action Mapping and Offline Checks

The current project records 5 Hz pose deltas, while this script multiplies
translation/rotation outputs by gains and sends velocity commands in
`--frame-id` (default `base_link`). Script defaults are 1 Hz inference and
10 Hz publishing, with angular gain 0. The examples above override publishing
to 20 Hz. Check the model's dataset statistics, action units, coordinate frame,
TCP, actual inference latency and real camera placement before attributing poor
grasping to the model. No gains, rate defaults or action conversion were changed
in this update; this difference has not been established as the cause of failure.

CPU-only regressions now check that failed inference clears residual smoothed
velocity and sends zero through the existing motion gate without issuing a new
gripper request. The next valid prediction can recover. `SetIO` rejection or a
failed future is logged as failure and permits the matching gripper request to
be retried; a delayed failure cannot clear a newer opposite command's cache.
An accepted IO request does not itself prove a physical grasp.

Still pending: camera/action freshness stopping, handling interrupted inference
and shutdown, workspace/table limits, and integration of physical grasp feedback
with VLA task completion. These CPU checks do not validate physical stopping or
grasping. The scripted synchronization path has its own DI feedback and timeout
logic and remains limited to `obj=1, dest=2`.
