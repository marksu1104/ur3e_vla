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
  test/remote/<run>/                 Bridge, stream, synchronization evidence.
  test/data/<run>/                   Collection H5, previews, manual review.
  test/model/<run>/                  Training/inference smoke-test artifacts.
  test/preflight/<run>/              CPU checks and version/hash reports.
  test/full_test/                    One planned manual acceptance run.
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
- Group test outputs by purpose, not by an unlabelled miscellaneous archive.
  Keep the evidence needed for current acceptance; remove superseded smoke
  tests after review. Use a separately managed backup for valuable H5/models.
  Latest evidence is linked from `docs/acceptance.md` and `docs/remote_readiness.md`.
  Old reports are consolidated in each group's ignored `history.json`; the
  existing `relocation_manifest.json` records path changes and cleanup.
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
- `tests/`: automated regression checks that do not require full Isaac startup;
  retain these sources. `__pycache__/` is disposable generated cache.
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

## Public and Local Instructions

`docs/command_reference.md` contains complete shareable commands, with variables
for external checkouts and cell-specific connections. `docs/local_commands.md`
is an ignored private copy with this workstation's concrete paths and connection
settings; it is not copied by a new Git clone. `docs/shared_environment.md` is
also ignored and documents local runtime administration. Transfer these private
files separately when handing over this workstation, rather than publishing
them. All operational entry points and their CLI parameters remain in `scripts/`.

## Tests and Tools

`tests/` has 16 modules and 191 CPU regression tests. They protect H5/MultiView
writing, TFDS conversion, Remote and synchronization command handling, VLA
client/server/rollout, scene options, startup/shutdown, and the OpenVLA patch.
Keep these sources; run `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -q`
after loading the Isaac/ROS environment. The patch test needs a local pinned
OpenVLA Git checkout; set `UR3E_OPENVLA_SOURCE` if it is outside the known layout.

| Tools | Purpose |
| --- | --- |
| validate_h5.py, export_h5_frames.py, subset_h5_demos.py | Read-only H5 checks, previews, or a separate smaller H5 |
| acceptance_preflight.py | CPU tests, version/hash capture, optional read-only H5 check |
| test_bridge_client.py, check_vla_prediction.py | Manual bridge requests or inference comparison; bridge tasks may act on the connected server |
| export_robot_asset.py, make_usd_collision_asset.py, export_printable_assets.py | Rebuild robot/collision assets or export STL; ordinary Remote runs need none of these |
| openvla_ur3e.patch | Register UR3e data and training export in the pinned external OpenVLA source |

Generated test scripts are not project entry points. Superseded preflight and
Remote outputs can be removed after their JSON results have been retained;
keep current H5, five-view previews, Remote evidence, and the smoke-test model
while manual acceptance is pending. Model exports may contain necessary Python
implementation files and must be preserved as a complete directory.
