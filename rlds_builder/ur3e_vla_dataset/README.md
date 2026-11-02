# UR3e VLA Dataset

This TFDS builder converts UR3e VLA HDF5 demonstrations into RLDS episodes.

Input must be selected explicitly. From the repository root, after collecting
H5 demos and activating `rlds_env`:

```bash
PROJECT_ROOT="$PWD"
export UR3E_VLA_H5_PATH="$PROJECT_ROOT/outputs/h5/pick_place_v1/red_mug/demos.h5"
cd "$PROJECT_ROOT/rlds_builder/ur3e_vla_dataset"
tfds build --data_dir "$PROJECT_ROOT/outputs/tfds/pick_place_v1"
```

Use multiple HDF5 files for multitask training with a comma-separated list:

```bash
export UR3E_VLA_H5_PATHS="$PROJECT_ROOT/outputs/h5/pick_place_v1/spoon/demos.h5,$PROJECT_ROOT/outputs/h5/pick_place_v1/red_mug/demos.h5,$PROJECT_ROOT/outputs/h5/pick_place_v1/bowl/demos.h5"
unset UR3E_VLA_H5_PATH
```

`UR3E_VLA_H5_PATHS` takes precedence over `UR3E_VLA_H5_PATH`.
The same commands work with shared H5 and TFDS paths under
`/srv/isaac_ros2/datasets/`; see `docs/command_reference.md` in the project
root. Do not mix H5 files from incompatible scenes or action formats.

Optional split settings:

```bash
export UR3E_VLA_VAL_RATIO=0.1
export UR3E_VLA_SPLIT_SEED=42
```

The current builder is **1.1.0**. Its output is
`<data_dir>/ur3e_vla_dataset/1.1.0/`; existing `1.0.0` exports remain historical
artifacts and do not gain these checks. Rebuild from the selected H5 inputs and
point training at the new export. Keep old exports separate when comparing runs.

Each build accepts one scene profile: `canonical_scene_v1` or
`multiview_scene_v1`. Files from those two profiles cannot be combined in one
build. Missing or unknown file profiles are rejected. Successful episodes must
use `gripper_action_encoding=logical_binary_0_open_1_closed`; missing or other
encodings are rejected. A successful episode's scene profile, when present,
must match its file. Episodes explicitly marked `success=False` are removed
before splitting; an input selection with no successful episodes is rejected.

Input paths are resolved, de-duplicated and sorted before the seeded split.
Reordering the comma-separated list or repeating a path/symlink cannot change
the split or repeat an episode. Episode keys include a SHA-256 hash of the
resolved source path, so separate collection runs with the same target and
`demo_0` can coexist. Moving source files changes these keys and may change split
membership; preserve the resolved input-path list and split settings with each
training run.

With a nonzero validation ratio and at least two successful episodes, the
builder reserves at least one episode for each of `train` and `val`. A single
successful episode stays in `train` without a `val` split. Setting the ratio to
`0.0` places all successful episodes in the TFDS `train` split. The external
OpenVLA loader may choose its own validation fallback when no `val` split exists.
