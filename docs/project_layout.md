# Project Layout and Artifacts

The repository keeps source code, shell entry points, and documentation in Git.
Generated data and model artifacts live under `outputs/`, which is ignored by
Git. This keeps the project self-contained on disk without committing large
files.

## Artifact Layout

```text
outputs/
  h5/        Canonical H5 demonstrations.
  tfds/      Rebuilt RLDS/TFDS datasets.
  models/    Training runs, checkpoints, and exports.
  media/     Videos and inspection images.
  test/      Disposable smoke-test output.
```

Canonical H5 data uses `scene_profile=canonical_scene_v1`. Keep older mug-only
datasets separate because their objects and gripper action representation differ.

## Storage Policy

- H5 files are the source of truth for demonstrations. Keep them when possible.
- TFDS/RLDS folders can be rebuilt from H5 and may be deleted when space is tight.
- Fine-tuned models can be large. Keep them in `outputs/models/` locally or back
  them up externally, but do not commit them to Git.
- Media/debug videos are useful for inspection but can be regenerated.
- Temporary checks and smoke-test outputs go under `outputs/test/`; do not mix them with curated H5, TFDS, or model folders.

## Code Organization

- `scripts/`: common runtime entrypoints.
- `scripts/tools/`: maintenance, validation, and data-inspection utilities.
- `scripts/collect_demos_multi_env.py`: vectorized demonstration collection.
- `docs/command_reference.md`: explicit terminal commands for normal runs and custom runs.
- `docs/`: runbooks, operational notes, and artifact policy.
- `vla_sim/`: simulation package code.
