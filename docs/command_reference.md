# Command Reference

Run Isaac commands from a fresh terminal after loading the shared ROS and Isaac
Lab environment. Deactivate Conda first if it is already active:

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash
```

## Remote Bridge

```bash
isaaclab scripts/run_remote_pick_place.py \
  --headless --enable_cameras --seed 42
```

The bridge listens on `127.0.0.1:8100`. Set your own SSH account, host and
port, then tunnel it from the local machine with any OpenSSH client:

```bash
REMOTE_USER=your_user
REMOTE_HOST=your_host
SSH_PORT=your_ssh_port
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:18100:127.0.0.1:8100 -p "$SSH_PORT" "$REMOTE_USER@$REMOTE_HOST"
```

`test_bridge_client.py` is only a manual protocol check:

```console
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:18100 --watch
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:18100 --obj 0 --dest 0
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:18100 --control reset
```

## H5 Collection

Canonical collection writes 640×480 main/wrist images at 5 Hz and a logical
binary gripper action. It must use a new H5 location, because it cannot mix
with legacy mug H5 data.

```bash
isaaclab scripts/collect_demos.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 500 --max-episodes-tried 700 \
  --output-dir outputs/h5/pick_place_v1
```

One no-save trajectory smoke test:

```bash
isaaclab scripts/collect_demos.py \
  --headless --enable_cameras \
  --target red_mug --episodes 1 --max-episodes-tried 1 \
  --output-dir outputs/test/collection_smoke \
  --no-save-h5
```

## Optional shared datasets and models

Each developer can keep a separate Git checkout while sharing generated
artifacts through `/srv/isaac_ros2`. This is optional: the default project
outputs remain inside `outputs/`. The examples below assume the account is in
the `isaac_ros2` group; use a new login session after group membership changes.
Do not put datasets or models in `/opt/isaac_ros2`, which holds the installed
runtime.

Collect a new red-mug dataset directly into shared storage (run from your own
checkout, after `source /opt/isaac_ros2/setup.bash`):

```bash
cd /path/to/your/ur3e_vla
umask 002
SHARED=/srv/isaac_ros2

isaaclab scripts/collect_demos.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 500 --max-episodes-tried 700 \
  --output-dir "$SHARED/datasets/h5/pick_place_v1"
```

This writes `.../h5/pick_place_v1/red_mug/demos.h5`; collect spoon and bowl by changing
`--target`. Choose a **new dataset name** for changed scenes or action formats.
The collector refuses to replace existing H5 unless `--overwrite` is supplied.
Do not combine these new-scene demos with legacy `mugs_*` data.

Build TFDS from the shared H5 without moving it (in the `rlds_env` terminal):

```bash
cd /path/to/your/ur3e_vla
SHARED=/srv/isaac_ros2
umask 002
conda activate rlds_env
export UR3E_VLA_H5_PATH="$SHARED/datasets/h5/pick_place_v1/red_mug/demos.h5"
export UR3E_VLA_VAL_RATIO=0.1
export UR3E_VLA_SPLIT_SEED=42
cd rlds_builder/ur3e_vla_dataset
tfds build --data_dir "$SHARED/datasets/tfds/pick_place_v1"
```

For multiple objects, set `UR3E_VLA_H5_PATHS` to a comma-separated list of
their H5 files and unset `UR3E_VLA_H5_PATH`. TFDS adds its own
`ur3e_vla_dataset/1.0.0` subdirectory beneath `--data_dir`.

### Prepare the external OpenVLA checkout

This repository does not contain OpenVLA or its weights. OpenVLA needs to know
the project's RLDS dataset name, image/state fields and action format. The
small patch below adds those registrations to one pinned OpenVLA revision; it
also adds the optional `--final_model_dir` export argument. It changes 15D
state to 8D and removes the terminal flag from the 8D action, leaving a 7D
policy action. Keep the OpenVLA environment separate from Isaac and TFDS.

Use a **clean** OpenVLA checkout; do not apply this over another developer's
uncommitted work. Set `PROJECT_ROOT` to your own clone of this repository:

```bash
PROJECT_ROOT=/path/to/ur3e_vla
OPENVLA_SRC=/path/to/openvla
cd "$OPENVLA_SRC"
test -z "$(git status --porcelain)" || { echo 'OpenVLA checkout is not clean'; false; }
git switch --detach c8f03f48af692657d3060c19588038c7220e9af9
git apply --check "$PROJECT_ROOT/scripts/tools/openvla_ur3e.patch"
git apply "$PROJECT_ROOT/scripts/tools/openvla_ur3e.patch"
git diff --check
```

The patch was checked with a one-step fine-tune on this revision. Install the
separate OpenVLA Python environment using the upstream instructions for this
revision before training. A different base model must support the same
OpenVLA action-prediction interface and UR3e statistics; changing the model
path alone does not establish compatibility.

### Fine-tune

Fine-tune and export a model to shared storage (in the separate `openvla`
environment) after applying the patch and building TFDS. For a personal run,
replace the `/srv/isaac_ros2` paths with this checkout's `outputs/` paths.

```bash
SHARED=/srv/isaac_ros2
OPENVLA_SRC=/path/to/your/openvla
BASE_MODEL=openvla/openvla-7b
umask 002
conda activate openvla
cd "$OPENVLA_SRC"

WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 vla-scripts/finetune.py \
  --vla_path "$BASE_MODEL" \
  --data_root_dir "$SHARED/datasets/tfds/pick_place_v1" \
  --dataset_name ur3e_vla_dataset \
  --run_root_dir "$SHARED/models/runs" \
  --adapter_tmp_dir "$SHARED/models/checkpoints" \
  --final_model_dir "$SHARED/models/pick_place_v1" \
  --batch_size 1 --max_steps 2000 --save_steps 250 \
  --learning_rate 1e-4 --shuffle_buffer_size 1000 --image_aug False
```

The server can load that shared export with
`--model-path /srv/isaac_ros2/models/pick_place_v1` and
`--unnorm-key ur3e_vla_dataset`. Keep the model's
`dataset_statistics.json` next to its weights. The shared H5, TFDS, and model
directories are generated artifacts, not Git-tracked project files.

## VLA Rollout

Start the server in the OpenVLA environment, then run the policy with the
canonical policy camera:

```bash
isaaclab scripts/run_vla.py \
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
isaaclab scripts/sync_sim_real.py \
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
