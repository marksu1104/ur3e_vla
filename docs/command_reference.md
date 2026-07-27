# Command Reference

Run Isaac commands after loading the ROS and Isaac Lab environment:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_ros2
cd ~/IsaacLab
```

## Remote Bridge

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/run_remote_pick_place.py \
  --headless --enable_cameras --seed 42
```

The bridge listens on `127.0.0.1:8100`. For a local machine, tunnel it with any
OpenSSH client:

```console
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:18100:127.0.0.1:8100 -p 6002 acolab@140.112.42.35
```

`test_bridge_client.py` is only a manual protocol check:

```console
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:18100 --watch
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:18100 --obj 0 --dest 0
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:18100 --control reset
```

## Canonical H5 Collection

Canonical collection writes 640×480 main/wrist images at 5 Hz and a logical
binary gripper action. It must use a new H5 location, because it cannot mix
with legacy mug H5 data.

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 500 --max-episodes-tried 700 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/h5/canonical_scene_red_mug \
  --overwrite
```

One no-save trajectory smoke test:

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/collect_demos.py \
  --headless --enable_cameras \
  --target red_mug --episodes 1 --max-episodes-tried 1 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/collection_smoke \
  --no-save-h5
```

## VLA Rollout

Start the server in the OpenVLA environment, then run the policy with the
canonical policy camera:

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/run_vla.py \
  --headless --enable_cameras \
  --target red_mug \
  --instruction "pick up the red mug" \
  --vla-server http://localhost:8000 \
  --camera camera_policy \
  --action-scale 0.5 --vla-step-interval 12 --max-steps 6000
```

## Read-Only Real-to-Sim Sync

This command consumes `/joint_states` by joint name and never publishes Servo,
trajectory, or other real-robot motion commands:

```bash
./isaaclab.sh -p ./ur3e_vla/scripts/sync_sim_real.py \
  --headless --enable_cameras \
  --joint-states-topic /joint_states \
  --joint-state-timeout 0.5
```

The sync runner also exposes the normal YOLO bridge stream on port 8100. Use
`/control` with `pause`, `resume`, or `reset` for the virtual scene only.

The physical sim-to-real mode currently supports only `obj=1, dest=2` and
reset. Use the exact commands in
[`docs/sim_real_sync.md`](sim_real_sync.md); other pairs are rejected.

## Multi-Env Collection

`scripts/collect_demos_multi_env.py` uses the canonical three-object config
and official Robotiq `finger_joint`. It loads the assembled USD required for
vectorized environments. Rebuild it with `scripts/tools/export_robot_asset.py` after
robot or gripper changes.
