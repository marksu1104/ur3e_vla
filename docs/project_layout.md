# Project Layout and Artifacts

The repository keeps source code, shell entry points, and documentation in Git.
Generated data and model artifacts live under `outputs/`, which is ignored by
Git. This keeps the project self-contained on disk without committing large
files.

## Artifact Layout

```text
outputs/
  h5/
    mugs_500/
      red_mug/demos.h5
      blue_mug/demos.h5
    mugs_100/
      red_mug/demos.h5
      blue_mug/demos.h5
    mugs_250/
      red_mug/demos.h5
      blue_mug/demos.h5
    legacy_banana/
      banana/demos.h5

  tfds/
    mugs_100/
      ur3e_vla_dataset/
    mugs_250/
      ur3e_vla_dataset/
    mugs_500/
      ur3e_vla_dataset/

  models/
    ur3e_vla_mugs_100/
    ur3e_vla_mugs_250/
    ur3e_vla_mugs_500/
    runs/
    checkpoints/

  media/
    sim_mug/
    sim_banana/

  test/
    smoke/
    h5/
    media/
```

## Storage Policy

- H5 files are the source of truth for demonstrations. Keep them when possible.
- TFDS/RLDS folders can be rebuilt from H5 and may be deleted when space is tight.
- Fine-tuned models can be large. Keep them in `outputs/models/` locally or back
  them up externally, but do not commit them to Git.
- Media/debug videos are useful for inspection but can be regenerated.
- Temporary checks and smoke-test outputs go under `outputs/test/`; do not mix them with curated H5, TFDS, or model folders.

## Code Organization

- `scripts/`: executable collection, conversion, inference, and inspection tools.
- `scripts/multi_env/`: experimental vectorized collection and assembled-USD tools.
- `docs/command_reference.md`: explicit terminal commands for normal runs and custom experiments.
- `docs/`: runbooks, operational notes, and artifact policy.
- `vla_sim/`: simulation package code.
