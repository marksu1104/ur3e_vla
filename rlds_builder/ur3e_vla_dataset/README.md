# UR3e VLA Dataset

This TFDS builder converts UR3e VLA HDF5 demonstrations into RLDS episodes.

Default input:

```text
../../outputs/h5/mugs_500/red_mug/demos.h5
```

Override the input file with:

```bash
export UR3E_VLA_H5_PATH=/path/to/demos.h5
```

Use multiple HDF5 files for multitask training with a comma-separated list:

```bash
export UR3E_VLA_H5_PATHS="../../outputs/h5/mugs_500/red_mug/demos.h5,../../outputs/h5/mugs_500/blue_mug/demos.h5"
```

`UR3E_VLA_H5_PATHS` takes precedence over `UR3E_VLA_H5_PATH`.

Optional split settings:

```bash
export UR3E_VLA_VAL_RATIO=0.1
export UR3E_VLA_SPLIT_SEED=42
```
