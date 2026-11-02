# Command Reference

可分享的完整操作指令：Remote、收集、TFDS、訓練、server、rollout 與驗收工具。
這台工作站的直接操作版本在 `docs/local_commands.md`（Git 忽略）；專用現場
步驟仍以 sim_real_sync.md 與 real_robot_vla_runbook.md 為準。

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

All collectors return a nonzero exit code when the requested episode count is
not reached or a camera/export fails. Successfully written episodes remain in
the partial H5 for inspection. An existing H5 requires `--overwrite`; choose a
new output directory to preserve it. The H5 writer also rejects duplicate
episode IDs rather than replacing existing demonstrations. Every camera stream must
contain same-shaped three-channel uint8 frames; the writer rejects nonnumeric,
non-finite, floating-point or out-of-range images instead of silently casting them.

Validate stored canonical or MultiView H5 without launching Isaac or modifying
the file (in the shared environment):

```bash
python scripts/tools/validate_h5.py outputs/h5/pick_place_v1/red_mug/demos.h5 \
  --expected-profile canonical_scene_v1 --expected-episodes 500
```

For MultiView use `--expected-profile multiview_scene_v1`. The validator checks
episode counts, camera sizes/lengths, all-black frames, finite state/action and
pose values, quaternion norms, logical gripper values, terminal flags and
metadata. It prints one summary and returns nonzero on failure. The 5 Hz check
uses metadata; visual quality and measured timing still need separate review.

## MultiView and Multi-Env Collection

Use separate dataset names; do not mix their H5 files with single-env data
until scene and action compatibility is verified. Choose a fresh destination
when rerunning. All three collectors accept `spoon`, `red_mug`, and `bowl`.

Five-view H5 plus preview video, one successful episode:

```bash
isaaclab scripts/collect_demos_multi_view.py \
  --headless --enable_cameras --seed 42 \
  --target red_mug --episodes 1 --max-episodes-tried 2 \
  --output-dir outputs/h5/multiview_trial001 --save-video

python scripts/tools/validate_h5.py outputs/h5/multiview_trial001/red_mug/demos.h5 \
  --expected-profile multiview_scene_v1 --expected-episodes 1
```

Multi-env collection, two environments and two successful episodes:

```bash
isaaclab scripts/collect_demos_multi_env.py \
  --headless --enable_cameras --seed 42 \
  --target red_mug --num-envs 2 --episodes 2 --max-episodes-tried 4 \
  --output-dir outputs/h5/multi_env_trial001
```

Multi-env uses the canonical three-object config, official Robotiq
`finger_joint`, and an assembled USD. Rebuild the asset with
`export_robot_asset.py` only after robot or gripper changes; ordinary runs do
not require re-exporting it. Full asset and camera checks are in
[full_test_commands.md](full_test_commands.md).

## Local H5 to TFDS

In a fresh terminal, activate TFDS separately from Isaac:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds_env
cd ~/IsaacLab/ur3e_vla
PROJECT_ROOT="$PWD"
unset UR3E_VLA_H5_PATHS
export UR3E_VLA_H5_PATH="$PROJECT_ROOT/outputs/h5/pick_place_v1/red_mug/demos.h5"
export UR3E_VLA_VAL_RATIO=0.1
export UR3E_VLA_SPLIT_SEED=42
cd rlds_builder/ur3e_vla_dataset
tfds build --data_dir "$PROJECT_ROOT/outputs/tfds/pick_place_v1"
```

For three collected objects from the same compatible scene profile, replace
the single input selection before `tfds build`:

```bash
unset UR3E_VLA_H5_PATH
export UR3E_VLA_H5_PATHS="$PROJECT_ROOT/outputs/h5/pick_place_v1/spoon/demos.h5,$PROJECT_ROOT/outputs/h5/pick_place_v1/red_mug/demos.h5,$PROJECT_ROOT/outputs/h5/pick_place_v1/bowl/demos.h5"
```

The builder writes `ur3e_vla_dataset/1.1.1` beneath `--data_dir`.
See the [builder README](../rlds_builder/ur3e_vla_dataset/README.md) for source
keys, split rules, filtering and migration. Moving H5 input paths changes
episode keys; preserve each training run's original input list.

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
`ur3e_vla_dataset/1.1.1` subdirectory beneath `--data_dir`. This builder rejects
mixed scene profiles and gripper encodings, filters failed episodes before
splitting, and distinguishes source runs in episode keys. Keep historical
`1.0.0` exports separate; see the builder README for split and migration details.

### Prepare the external OpenVLA checkout

This repository does not contain OpenVLA or its weights. OpenVLA needs to know
the project's RLDS dataset name, image/state fields and action format. The
patch below adds those registrations to one pinned OpenVLA revision; it
also adds the optional `--final_model_dir` export argument. It changes 15D
state to 8D and removes the terminal flag from the 8D action, leaving a 7D
policy action. Keep the OpenVLA environment separate from Isaac and TFDS.

`--max_steps` counts completed optimizer updates, including gradient
accumulation. The final update is always exported even when it falls before
the next `--save_steps` interval or retained checkpoints are enabled. All three
step parameters must be positive.

Use a **clean** OpenVLA checkout; do not apply this over another developer's
uncommitted work. Set `PROJECT_ROOT` to your own clone of this repository:

```bash
(
set -euo pipefail
PROJECT_ROOT=/path/to/ur3e_vla
OPENVLA_SRC=/path/to/openvla
cd "$OPENVLA_SRC"
test -z "$(git status --porcelain)" || { echo 'OpenVLA checkout is not clean'; exit 1; }
git switch --detach c8f03f48af692657d3060c19588038c7220e9af9
git apply --check "$PROJECT_ROOT/scripts/tools/openvla_ur3e.patch"
git apply "$PROJECT_ROOT/scripts/tools/openvla_ur3e.patch"
git diff --check
)
```

The 2026-10-04 integration ran a real one-update BF16 LoRA fine-tune,
model export, HTTP inference and a 240-step Isaac rollout; see
[the archived integration evidence](archive/validation_history.md#2026-10-04-integration).
That run predates the latest atomic-export hardening. The current patch passes
CPU regression tests; its real short training/export still needs acceptance. Install the
separate OpenVLA Python environment using the upstream instructions for this
revision before training. A different base model must support the same
OpenVLA action-prediction interface and UR3e statistics; changing the model
path alone does not establish compatibility.

### Fine-tune

Fine-tune and export a model to shared storage (in the separate `openvla`
environment) after applying the patch and building TFDS. For a personal run,
replace the `/srv/isaac_ros2` paths with this checkout's `outputs/` paths.

```bash
(
set -euo pipefail
SHARED=/srv/isaac_ros2
OPENVLA_SRC=/path/to/your/openvla
BASE_MODEL=openvla/openvla-7b
FINAL_MODEL="$SHARED/models/pick_place_v1_run001"
umask 002
conda activate openvla
cd "$OPENVLA_SRC"
test ! -e "$FINAL_MODEL" || { echo "Choose a new FINAL_MODEL path"; exit 1; }

WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 vla-scripts/finetune.py \
  --vla_path "$BASE_MODEL" \
  --data_root_dir "$SHARED/datasets/tfds/pick_place_v1" \
  --dataset_name ur3e_vla_dataset \
  --run_root_dir "$SHARED/models/runs" \
  --adapter_tmp_dir "$SHARED/models/checkpoints" \
  --final_model_dir "$FINAL_MODEL" \
  --batch_size 1 --max_steps 2000 --save_steps 250 \
  --learning_rate 1e-4 --shuffle_buffer_size 1000 --image_aug False
)
```

`--final_model_dir` must not already exist; choose a new run-specific path
instead of reusing an old model directory. The patched trainer writes weights,
processor files and `dataset_statistics.json` into a hidden sibling staging
directory, validates the complete export, then publishes it with one directory
rename. A failed training or export leaves the requested final path absent and
removes its staging directory, so it cannot mix old weights with new statistics.

The server can load that shared export with
`--model-path /srv/isaac_ros2/models/pick_place_v1_run001` and
`--unnorm-key ur3e_vla_dataset`. The shared H5, TFDS, and model directories are
generated artifacts, not Git-tracked project files.

## VLA Server

In a fresh terminal, use the external OpenVLA environment. Set the source and
model paths to your own checkout and complete trained export:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla
cd ~/IsaacLab/ur3e_vla
OPENVLA_SRC=/path/to/openvla
MODEL_PATH=/srv/isaac_ros2/models/pick_place_v1_run001
export PYTHONPATH="$OPENVLA_SRC${PYTHONPATH:+:$PYTHONPATH}"

python scripts/vla_server.py \
  --model-path "$MODEL_PATH" --unnorm-key ur3e_vla_dataset \
  --host 127.0.0.1 --port 8000
```

Keep this terminal open. In another terminal check readiness:

```bash
curl -fsS http://127.0.0.1:8000/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/config | python3 -m json.tool
```

The statistics must include `ur3e_vla_dataset`. A one-update smoke-test export
can check this path but does not establish grasping quality.

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

## Data Inspection and Acceptance Preflight

Use the shared Isaac/ROS Python in a fresh terminal:

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

python scripts/tools/export_h5_frames.py \
  --h5 outputs/h5/pick_place_v1/red_mug/demos.h5 \
  --out-dir outputs/test/data/h5_preview_trial001 --video

python scripts/tools/subset_h5_demos.py \
  --input outputs/h5/pick_place_v1/red_mug/demos.h5 \
  --output outputs/test/data/subset_trial001/demos.h5 --episodes 1

python scripts/tools/check_vla_prediction.py \
  --h5 outputs/h5/pick_place_v1/red_mug/demos.h5 \
  --server http://127.0.0.1:8000 --steps 0 5 10
```

Run the preflight independently when you need a versioned automatic report:

```bash
python scripts/tools/acceptance_preflight.py \
  --h5 outputs/h5/pick_place_v1/red_mug/demos.h5
```

It creates a fresh directory under `outputs/test/preflight/`, runs the CPU
suite and compile/Git checks, and validates the selected H5 without modifying
it. A failed check or skipped test gives a nonzero exit code. H5 validation,
unit tests, and preview videos do not replace manual acceptance.

Tests and tools are summarized in [Project Layout](project_layout.md#tests-and-tools).
