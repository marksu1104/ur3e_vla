# Sim–Real Synchronization

`scripts/sync_sim_real.py` keeps the canonical Isaac scene, bridge stream, and
real UR3e joint state in one persistent session.

## Supported Scope

The current sim-to-real implementation is intentionally limited to:

```text
object:      1 (red_mug)
destination: 2
control:     reset
```

Every other object/destination pair is rejected before a trial is reserved.
`reset` preserves the currently validated behavior: it opens the real gripper,
resets the virtual objects, and moves the real arm back to `HOME_POS`. It is a
physical motion command when `--enable-motion` is active.

The stream and `/status` remain available throughout the session on port 8100.

## Environment

Use the ROS domain values configured for the current robot cell. The values
below are specific to this workstation; do not add them globally to shell
startup files.

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_ros2
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

Open a separate terminal for each long-running process below. Set the same ROS
domain variables in every ROS and Isaac terminal.

## Terminal A: UR Driver

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

ros2 launch ur_robot_driver ur3e.launch.py \
  robot_ip:=192.168.10.175 \
  use_mock_hardware:=false \
  initial_joint_controller:=forward_velocity_controller \
  launch_rviz:=false
```

On the teach pendant, load the External Control program, press Play, and leave
the speed slider at the previously validated setting. Keep this terminal and
the External Control program running.

For read-only real-to-sim mirroring, Terminal A and a healthy `/joint_states`
topic are sufficient. Sim-to-real motion also requires Terminals B and C.

## Terminal B: MoveIt Servo

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

ros2 launch vla_ros_bridge vla_bridge.launch.py
```

Keep this terminal running while sim-to-real is active.

## Terminal C: Controller and ROS Checks

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

ros2 control switch_controllers \
  --activate forward_velocity_controller \
  --deactivate forward_position_controller scaled_joint_trajectory_controller

ros2 service call /servo_node/switch_command_type \
  moveit_msgs/srv/ServoCommandType "{command_type: 1}"

ros2 control list_controllers
ros2 topic info /joint_states -v
ros2 topic echo /joint_states --once
ros2 topic echo /servo_node/status --once
ros2 topic echo /io_and_status_controller/io_states --once
```

Do not continue to sim-to-real motion unless all of these are true:

- `forward_velocity_controller` is `active`.
- `/joint_states` has exactly one expected publisher and reports the six UR3e
  arm joints with plausible, finite values.
- Servo status has no singularity, collision, or halt warning.
- Gripper IO is available; this cell uses DI 0 for released and DI 2 for a
  confirmed grasp.
- The real arm is at the fixed initial pose and the workspace is clear.

## Terminal D: Read-Only Real-to-Sim

This mode subscribes to the real six-axis joint state and never publishes
robot motion:

```bash
cd ~/IsaacLab

./isaaclab.sh -p ./ur3e_vla/scripts/sync_sim_real.py \
  --headless --enable_cameras \
  --direction real-to-sim \
  --joint-states-topic /joint_states \
  --joint-state-timeout 0.5
```

## Terminal D: Sim-to-Real Dry Run

Start without `--enable-motion` first:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_ros2
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
cd ~/IsaacLab

./isaaclab.sh -p ./ur3e_vla/scripts/sync_sim_real.py \
  --headless --enable_cameras \
  --direction sim-to-real \
  --joint-states-topic /joint_states \
  --joint-state-timeout 0.5 \
  --max-linear-speed 0.05 \
  --grasp-z-offset 0.03 \
  --place-z-offset 0.05
```

The bridge should report:

```json
{
  "supported_tasks": [
    {"obj": 1, "dest": 2}
  ]
}
```

Send the supported task:

```bash
cd ~/IsaacLab/ur3e_vla
conda activate env_isaaclab_ros2

python3 scripts/tools/test_bridge_client.py \
  --server http://127.0.0.1:8100 \
  --obj 1 \
  --dest 2
```

## Terminal D: Validated Sim-to-Real Motion

Only after the dry run and physical safety checks pass, restart with the same
parameters plus `--enable-motion`:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_ros2
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
cd ~/IsaacLab

./isaaclab.sh -p ./ur3e_vla/scripts/sync_sim_real.py \
  --enable_cameras \
  --direction sim-to-real \
  --joint-states-topic /joint_states \
  --joint-state-timeout 0.5 \
  --enable-motion \
  --max-linear-speed 0.05 \
  --grasp-z-offset 0.03 \
  --place-z-offset 0.05
```

## Terminal E: Task and Reset Commands

Send only:

```bash
cd ~/IsaacLab/ur3e_vla
conda activate env_isaaclab_ros2

python3 scripts/tools/test_bridge_client.py \
  --server http://127.0.0.1:8100 \
  --obj 1 \
  --dest 2
```

After the trial, `reset` opens the gripper and physically homes the arm:

```bash
python3 scripts/tools/test_bridge_client.py \
  --server http://127.0.0.1:8100 \
  --control reset
```

Stop the synchronization process with `Ctrl+C`. Do not leave Isaac, bridge, or
test clients running after the session.

## Not Yet Supported

- Any object other than `obj=1`.
- Destination 0 or 1.
- Pause/resume during sim-to-real motion.
- General-purpose calibrated TCP control.
- VLA-driven simultaneous real/sim motion.
