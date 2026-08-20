"""Export the scene's unique tabletop parts as millimetre-scale STL files."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_parser.add_argument(
    "--output-dir",
    type=Path,
    default=PROJECT_ROOT / "outputs" / "printable",
)
_parser.add_argument("--overwrite", action="store_true")
_args, _ = _parser.parse_known_args()

if "--headless" not in sys.argv:
    sys.argv.append("--headless")

from vla_sim.isaac_app import args_cli, boot_app, close_app

app = boot_app()

import numpy as np
import omni.usd
import trimesh
from pxr import Gf, Usd, UsdGeom

from vla_sim.config import (
    CUTLERY_BOX_SIZE,
    CUTLERY_BOX_WALL_HEIGHT,
    CUTLERY_BOX_WALL_THICKNESS,
    MUG_COASTER_HEIGHT,
    MUG_COASTER_RADIUS,
)
from vla_sim.runtime import RuntimeOptions, SimulationRuntime


PARTS = {
    "spoon": {
        "prim": "/World/Spoon",
        "quantity": 2,
        "source": "Poly Haven wooden_spoon (CC0)",
    },
    "mug": {
        "prim": "/World/Red_mug/SmoothedVisual",
        "quantity": 2,
        "source": "YCB 025_mug via NVIDIA Isaac assets",
    },
    "bowl": {
        "prim": "/World/Bowl/SmoothedVisual",
        "quantity": 3,
        "source": "YCB 024_bowl via NVIDIA Isaac assets",
    },
    "cutlery_tray": {
        "prim": "/World/DestinationFixtures/CutleryBox",
        "quantity": 1,
        "source": "project procedural fixture",
    },
    "coaster": {
        "prim": "/World/DestinationFixtures/MugCoasters/Destination",
        "quantity": 2,
        "source": "project procedural fixture",
    },
}


def _visible_meshes(stage, root_path: str):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"printable prim not found: {root_path}")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        yield UsdGeom.Mesh(prim)


def _triangles(mesh: UsdGeom.Mesh) -> np.ndarray:
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    faces = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + int(count)]
        cursor += int(count)
        for index in range(1, len(face) - 1):
            faces.append((face[0], face[index], face[index + 1]))
    return np.asarray(faces, dtype=np.int64)


def _stage_part_mesh(stage, root_path: str) -> trimesh.Trimesh:
    """Flatten visible composed USD meshes into the upright world orientation."""
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    cache = UsdGeom.XformCache()
    vertices = []
    faces = []
    vertex_offset = 0
    for mesh in _visible_meshes(stage, root_path):
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        mesh_faces = _triangles(mesh)
        if not len(points) or not len(mesh_faces):
            continue
        transform = cache.GetLocalToWorldTransform(mesh.GetPrim())
        world = np.asarray(
            [transform.Transform(Gf.Vec3d(*point)) for point in points],
            dtype=np.float64,
        )
        # STL is unitless; millimetres are the interoperable slicer convention.
        vertices.append(world * meters_per_unit * 1000.0)
        faces.append(mesh_faces + vertex_offset)
        vertex_offset += len(points)
    if not vertices:
        raise RuntimeError(f"no visible mesh geometry found below {root_path}")

    result = trimesh.Trimesh(
        vertices=np.concatenate(vertices),
        faces=np.concatenate(faces),
        process=True,
    )
    result.remove_unreferenced_vertices()
    result.fix_normals()
    components = result.split(only_watertight=False)
    if len(components) > 1:
        components = sorted(components, key=lambda item: item.area, reverse=True)
        discarded_area = sum(item.area for item in components[1:])
        if discarded_area > components[0].area * 0.001:
            raise RuntimeError(
                f"{root_path} contains multiple substantial mesh components"
            )
        # YCB scans contain a few disconnected two-triangle remnants. They are
        # not part of the rendered object and would make an otherwise closed
        # print fail slicer manifold checks.
        result = components[0]
    # Use the scene's upright orientation and put the part on the slicer bed.
    bounds = result.bounds
    result.apply_translation(
        (-0.5 * (bounds[0, 0] + bounds[1, 0]),
         -0.5 * (bounds[0, 1] + bounds[1, 1]),
         -bounds[0, 2])
    )
    return result


def _procedural_part(name: str) -> trimesh.Trimesh | None:
    """Build exact printable forms for fixtures authored as USD primitives."""
    if name == "coaster":
        return trimesh.creation.cylinder(
            radius=MUG_COASTER_RADIUS * 1000.0,
            height=MUG_COASTER_HEIGHT * 1000.0,
            sections=128,
        )
    if name != "cutlery_tray":
        return None

    box_x, box_y, box_z = (value * 1000.0 for value in CUTLERY_BOX_SIZE)
    wall = CUTLERY_BOX_WALL_THICKNESS * 1000.0
    wall_height = CUTLERY_BOX_WALL_HEIGHT * 1000.0
    outer = (0.5 * box_x, 0.5 * box_y)
    inner = (outer[0] - wall, outer[1] - wall)

    def corners(extents, z):
        x, y = extents
        return [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]

    vertices = np.asarray(
        corners(outer, 0.0)
        + corners(outer, wall_height)
        + corners(inner, box_z)
        + corners(inner, wall_height),
        dtype=np.float64,
    )

    def quad(a, b, c, d):
        return [(a, b, c), (a, c, d)]

    faces = [(0, 2, 1), (0, 3, 2)]  # outer bottom
    for index in range(4):
        next_index = (index + 1) % 4
        faces += quad(index, next_index, 4 + next_index, 4 + index)
        faces += quad(4 + index, 4 + next_index, 12 + next_index, 12 + index)
        faces += quad(8 + index, 8 + next_index, 12 + next_index, 12 + index)
    faces += [(8, 9, 10), (8, 10, 11)]  # cavity floor
    tray = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    tray.fix_normals()
    return tray


def main() -> None:
    output_dir = _args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"{name}.stl" for name in PARTS]
    expected.append(output_dir / "manifest.json")
    existing = [path for path in expected if path.exists()]
    if existing and not _args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to replace {names}; pass --overwrite")

    runtime = SimulationRuntime(
        RuntimeOptions(scene_profile="printable", device=args_cli.device)
    ).start()
    stage = omni.usd.get_context().get_stage()
    manifest = {
        "units": "millimetres",
        "coordinate_system": "scene upright, centred in XY, minimum Z at zero",
        "parts": {},
        "notes": [
            "Quantities include movable and fixed reference copies in the remote scene.",
            "Every exported STL is reloaded and must pass a watertight mesh check.",
            "Review YCB/NVIDIA source terms before redistributing derived mug or bowl files.",
        ],
    }
    for name, specification in PARTS.items():
        mesh = _procedural_part(name)
        if mesh is None:
            mesh = _stage_part_mesh(stage, specification["prim"])
        if not mesh.is_watertight:
            raise RuntimeError(f"{name} is not a watertight printable mesh")
        destination = output_dir / f"{name}.stl"
        mesh.export(destination, file_type="stl")
        exported = trimesh.load_mesh(destination, process=True)
        if not exported.is_watertight:
            raise RuntimeError(f"exported {destination.name} failed watertight check")
        manifest["parts"][name] = {
            **specification,
            "file": destination.name,
            "size_mm": [round(float(value), 2) for value in mesh.extents],
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(exported.is_watertight),
            "body_count": int(exported.body_count),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(PARTS)} unique parts to {output_dir}")
    for name, details in manifest["parts"].items():
        print(
            f"  {name}: {details['size_mm']} mm, "
            f"watertight={details['watertight']}, quantity={details['quantity']}"
        )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        # SimulationApp shutdown can otherwise obscure the originating error.
        traceback.print_exc()
        exit_code = 1
    finally:
        close_app()
    if exit_code:
        raise SystemExit(exit_code)
