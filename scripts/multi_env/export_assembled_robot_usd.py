"""Export the runtime-assembled UR3e + Robotiq USD stage.

The stable simulation path assembles UR3e and the Robotiq 2F-140 at runtime with
RobotAssembler.  After assembly, Robotiq is still authored as /World/Gripper in
the USD stage even though Isaac can control the combined articulation at runtime.
For that reason this exporter writes the flattened /World subtree by default,
not only /World/Robot.  Exporting only /World/Robot silently drops the gripper.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "assets" / "ur3e_robotiq_2f140_assembled_stage.usda",
    help="Output USD path for the assembled stage asset.",
)
_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the output file if it already exists.",
)
_parser.add_argument(
    "--source-prim",
    default="/World",
    help="Flattened stage prim to export. Keep /World to preserve /World/Robot and /World/Gripper.",
)
_parser.add_argument(
    "--root-prim",
    default="/World",
    help="Root prim path inside the exported asset.",
)
_parser.add_argument(
    "--allow-missing-gripper",
    action="store_true",
    help="Do not fail if the exported USD does not contain Robotiq gripper prims.",
)
_parser.add_argument(
    "--show-gui",
    action="store_true",
    help="Run with the Isaac Sim GUI instead of headless mode.",
)
_extra_args, _ = _parser.parse_known_args()

if not _extra_args.show_gui and "--headless" not in sys.argv:
    sys.argv.append("--headless")

from vla_sim.isaac_app import boot_app, log

app = boot_app()
_app_closed = False


def close_app_once():
    global _app_closed
    if _app_closed:
        return
    _app_closed = True
    try:
        app.close(wait_for_replicator=False)
    except TypeError:
        app.close()


import traceback

import omni.kit.app
import omni.usd
from pxr import Sdf, Usd, UsdGeom

from vla_sim.scene import enable_extensions, spawn_raw_and_assemble


_GRIPPER_MARKERS = (
    "robotiq_arg2f_base_link",
    "finger_joint",
    "left_outer_knuckle",
    "right_outer_knuckle",
)


def _copy_flattened_prim(src_stage: Usd.Stage, source_prim: str, output_path: Path, root_prim: str) -> None:
    src_path = Sdf.Path(source_prim)
    dst_path = Sdf.Path(root_prim)
    if not src_path.IsAbsolutePath() or not dst_path.IsAbsolutePath():
        raise ValueError("source-prim and root-prim must be absolute USD paths")

    if not src_stage.GetPrimAtPath(src_path).IsValid():
        raise RuntimeError(f"Source prim not found after assembly: {source_prim}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    flat_layer = src_stage.Flatten()
    dst_stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(dst_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dst_stage, 1.0)

    dst_layer = dst_stage.GetRootLayer()
    copied = Sdf.CopySpec(flat_layer, src_path, dst_layer, dst_path)
    if not copied:
        raise RuntimeError(f"Failed to copy flattened prim {source_prim} -> {root_prim}")

    # Usd.Stage.Flatten() can emit instance prototypes as sibling root prims
    # such as /Flattened_Prototype_7. If only /World is copied, the exported
    # asset looks valid but referenced visuals/collisions become unresolved when
    # IsaacLab loads the asset under /World/envs/env_*/Assembled. Keep those
    # prototypes inside the default prim so the asset is self-contained.
    prototype_prefix = "Flattened_Prototype_"
    for root_spec in list(flat_layer.rootPrims):
        if root_spec.name.startswith(prototype_prefix):
            src_proto = Sdf.Path.absoluteRootPath.AppendChild(root_spec.name)
            dst_proto = dst_path.AppendChild(root_spec.name)
            if not Sdf.CopySpec(flat_layer, src_proto, dst_layer, dst_proto):
                raise RuntimeError(f"Failed to copy flattened prototype {src_proto} -> {dst_proto}")

    root = dst_stage.GetPrimAtPath(dst_path)
    if not root.IsValid():
        raise RuntimeError(f"Exported root prim is invalid: {root_prim}")
    dst_stage.SetDefaultPrim(root)
    dst_layer.Save()

    text = output_path.read_text()
    text = text.replace("</Flattened_Prototype_", f"<{root_prim}/Flattened_Prototype_")
    output_path.write_text(text)


def _validate_export(output_path: Path, require_gripper: bool) -> None:
    text = output_path.read_text(errors="ignore")
    found = [marker for marker in _GRIPPER_MARKERS if marker in text]
    if require_gripper and not found:
        raise RuntimeError(
            "Exported USD does not contain Robotiq markers. "
            "Use the default --source-prim /World, or pass --allow-missing-gripper only for debugging."
        )
    log(f"Robotiq markers found: {found if found else 'none'}")


def main() -> None:
    output_path = _extra_args.output.expanduser().resolve()
    if output_path.exists() and not _extra_args.overwrite:
        raise FileExistsError(f"Output already exists. Pass --overwrite to replace it: {output_path}")

    log("Exporting runtime-assembled UR3e + Robotiq USD stage")
    log(f"Output     : {output_path}")
    log(f"Source prim: {_extra_args.source_prim}")
    log(f"Root prim  : {_extra_args.root_prim}")

    enable_extensions()
    spawn_raw_and_assemble()

    kit = omni.kit.app.get_app()
    for _ in range(30):
        kit.update()

    stage = omni.usd.get_context().get_stage()
    _copy_flattened_prim(stage, _extra_args.source_prim, output_path, _extra_args.root_prim)
    _validate_export(output_path, require_gripper=not _extra_args.allow_missing_gripper)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log(f"Export complete: {output_path} ({size_mb:.1f} MB)")
    log("Note: this is a flattened assembled stage containing /World/Robot and /World/Gripper.")
    log("Next step: build a multi-env SceneCfg that references or clones this assembled stage safely.")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"EXCEPTION: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise
    finally:
        close_app_once()
