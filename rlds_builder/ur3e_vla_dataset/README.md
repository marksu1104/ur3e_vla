# UR3e VLA Dataset

This TFDS builder converts UR3e VLA HDF5 demonstrations into RLDS episodes.

Default input:

```text
../../outputs/data/mug/demos.h5
```

Override the input file with:

```bash
export UR3E_VLA_H5_PATH=/path/to/demos.h5
```

Optional split settings:

```bash
export UR3E_VLA_VAL_RATIO=0.1
export UR3E_VLA_SPLIT_SEED=42
```
