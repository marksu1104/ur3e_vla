# Project Layout and Artifacts

The repository keeps source code, assets, entry points, and documentation in
Git. Generated artifacts are ignored by Git. Use `outputs/` in a personal
checkout by default; choose the shared paths below only when another account
needs the same dataset or model. Do not copy artifacts between these layouts
during a normal collection-to-training workflow.

## Artifact Layout

```text
outputs/
  h5/<dataset>/<object>/demos.h5       Source demonstrations.
  tfds/<dataset>/                    TFDS build root (TFDS adds name/version).
  models/<run>/                      Exported model and dataset_statistics.json.
  models/runs/                       Training logs and temporary checkpoints.
  test/<purpose>/                    Disposable smoke tests, previews, videos.
  printable/                         STL parts and manifest.json.
```

For the current single-scene collection, use `pick_place_v1` as the dataset
name: `outputs/h5/pick_place_v1/red_mug/demos.h5` and
`outputs/tfds/pick_place_v1/`. These names label a *dataset*, not the Python
scene class. The H5 `scene_profile` remains `canonical_scene_v1`. Keep
MultiView, multi-env experiments, and older mug-only data in separate named
datasets until their schema, cameras, objects, and action semantics have been
verified compatible. Change the dataset name when those semantics change; do
not overwrite a dataset used for training. Pass `--output-dir` explicitly in
collection commands so the destination is visible.

For multiple accounts, use the same simple H5/TFDS split under the existing
group-writable storage:

```text
/srv/isaac_ros2/datasets/h5/<dataset>/<object>/demos.h5
/srv/isaac_ros2/datasets/tfds/<dataset>/
/srv/isaac_ros2/models/<run>/
```

Keep test outputs in each personal checkout, not shared storage. A model run
name is independent of the dataset name; record which dataset and base model
produced it. `/opt/isaac_ros2/` contains the installed runtime, never datasets
or model exports. See `docs/command_reference.md` for direct-write examples.

## Storage Policy

- H5 is the source of truth. TFDS can be rebuilt from H5; model weights cannot
  be assumed reproducible from H5 alone. Back up valuable H5/model exports.
- Keep only selected demonstration media; put exploratory images and videos in
  `outputs/test/` and remove them after review.
- Do not create an `archive/` for miscellaneous old outputs. Use an intentional,
  separately managed backup for anything that must survive cleanup.
- Direct new runs to distinct paths. Avoid `--overwrite` unless deliberately
  replacing the exact dataset or model run.

## Code Organization

- `assets/`: checked-in source USD and collision assets; not generated runs.
- `scripts/`: workflow entrypoints with CLI arguments.
- `scripts/tools/`: maintenance, validation, and data-inspection utilities.
- Workstation user/group installation and rollback scripts are private machine
  administration, not project entry points. Normal runs use an already
  installed Isaac/ROS environment.
- `scripts/collect_demos_multi_env.py`: vectorized demonstration collection.
- `vla_sim/`: reusable simulation, robot, bridge, and data code.
- `rlds_builder/`: source code that converts H5 to TFDS.
- `scripts/tools/openvla_ur3e.patch`: reviewed adapter for the pinned external
  OpenVLA checkout; weights and that checkout stay outside this repository.
- `tests/`: small automated checks that do not require full Isaac startup.
- `docs/`: runbooks and operational notes; `command_reference.md` holds normal
  commands and this file owns artifact paths.
- `vla_sim/config.py`: shared numeric scene and task values.
- `vla_sim/scene/base_scene.py`: canonical scene assembly, lights, and markers.
- `vla_sim/scene/assets.py`: robot, furniture, movable objects, and USD assets.
- `vla_sim/scene/cameras.py`: YOLO, policy, wrist, and reusable camera builders.
- `vla_sim/scene/materials.py`: render-only materials and mesh presentation.
- `vla_sim/scene/destinations.py`: remote destination and reference props.
- `vla_sim/scene/scene_options.py`: named workflow camera, fixture, and lighting selections.
- `vla_sim/simulation.py`: stepping, controllers, and state backends.
- `vla_sim/multiview/`: isolated MultiView experiment owned by its active workstream.

Avoid copying a scene class into a new entry point. Select named scene options
from the shared base scene and keep workflow-specific additions out of options
that do not use them. This keeps local lighting, camera, and prop experiments
from changing collection, VLA, synchronization, or MultiView unexpectedly.
