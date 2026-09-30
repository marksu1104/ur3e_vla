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
