"""Create a local USD asset with collision generated from visual meshes.

Run with Isaac Lab Python:
    cd ~/IsaacLab
    ./isaaclab.sh -p ./ur3e_vla/scripts/tools/make_usd_collision_asset.py \
        --source /Props/YCB/Axis_Aligned/025_mug.usd \
        --output ./ur3e_vla/assets/025_mug_collision.usda \
        --approximation convexDecomposition \
        --mass 0.20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Generate a collision-enabled USD wrapper from a visual mesh USD.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--source", required=True, help="Source visual USD path, Omniverse URL, or /Props/... path.")
    parser.add_argument("--output", required=True, help="Output USD path. Local paths are created as needed.")
    parser.add_argument(
        "--root-prim",
        default="/World/Object",
        help="Prim path where the source asset is referenced in the output stage.",
    )
    parser.add_argument(
        "--approximation",
        default="convexDecomposition",
        choices=(
            "none",
            "convexHull",
            "convexDecomposition",
            "meshSimplification",
            "sdf",
            "boundingCube",
            "boundingSphere",
        ),
        help=(
            "Collision approximation. For mugs, convexDecomposition is usually "
            "the first useful choice; convexHull is faster but closes concavities."
        ),
    )
    parser.add_argument("--mass", type=float, default=0.20, help="Mass in kg applied to the root prim.")
    parser.add_argument("--contact-offset", type=float, default=0.002, help="PhysX contact offset.")
    parser.add_argument("--rest-offset", type=float, default=0.0, help="PhysX rest offset.")
    parser.add_argument(
        "--bake-first-mesh",
        action="store_true",
        help="Copy the first source mesh into the output instead of referencing the source USD.",
    )
    parser.add_argument(
        "--bottom-support-center",
        help="Optional x,y,z center of a thin box merged into a baked mesh, in meters.",
    )
    parser.add_argument(
        "--bottom-support-size",
        help="Optional x,y,z size of a box merged into a baked mesh, in meters.",
    )
    parser.add_argument(
        "--no-rigid-body",
        action="store_true",
        help="Only add mesh collisions; do not add rigid body and mass APIs.",
    )
    return parser


args_cli, _unknown = build_parser().parse_known_args()
args_cli.headless = True
args_cli.experience = "isaacsim.exp.full.kit"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def log(message: str) -> None:
    print(f"[collision-asset] {message}", flush=True)


def configure_asset_root() -> None:
    import carb

    settings = carb.settings.get_settings()
    asset_url = os.environ.get(
        "ISAAC_ASSET_ROOT_URL",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )
    for key in (
        "/persistent/isaac/asset_root/cloud",
        "/persistent/isaac/asset_root/default",
        "/persistent/isaac/asset_root/nvidia",
    ):
        if not settings.get(key):
            settings.set(key, asset_url)
    log(f"asset root: {settings.get('/persistent/isaac/asset_root/cloud')}")


def get_isaac_nucleus_dir() -> str | None:
    env_path = os.environ.get("ISAAC_NUCLEUS_DIR")
    if env_path:
        return env_path.rstrip("/")

    try:
        import carb
    except ImportError:
        return None

    asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    if not asset_root:
        return None
    return f"{asset_root.rstrip('/')}/Isaac"


def resolve_source_path(source: str) -> str:
    if source.startswith("$ISAAC_NUCLEUS_DIR/"):
        source = source.replace("$ISAAC_NUCLEUS_DIR/", "", 1)

    if source.startswith("/Props/") or source.startswith("Props/"):
        isaac_nucleus_dir = get_isaac_nucleus_dir()
        if not isaac_nucleus_dir:
            raise RuntimeError(
                "--source looks like a Nucleus-relative path, but the Isaac asset "
                "root could not be resolved. Export ISAAC_NUCLEUS_DIR or pass a full USD path."
            )
        return f"{isaac_nucleus_dir}/{source.lstrip('/')}"

    return source


def ensure_parent(path: str) -> None:
    if "://" in path:
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def normalized_stage_path(path: str) -> str:
    if "://" in path:
        return path
    return str(Path(path).expanduser().resolve())


def parse_vec3(value: str | None, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    parts = tuple(float(part.strip()) for part in value.split(","))
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values")
    return parts


def apply_collision_to_mesh(mesh_prim, approximation: str, contact_offset: float, rest_offset: float) -> bool:
    from pxr import UsdPhysics

    if not mesh_prim.IsValid():
        return False

    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    if approximation != "none":
        mesh_collision.CreateApproximationAttr().Set(approximation)

    try:
        from pxr import PhysxSchema
    except ImportError:
        return True

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_collision.CreateContactOffsetAttr().Set(float(contact_offset))
    physx_collision.CreateRestOffsetAttr().Set(float(rest_offset))
    return True


def main() -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    configure_asset_root()
    source = resolve_source_path(args_cli.source)
    output = normalized_stage_path(args_cli.output)
    ensure_parent(output)

    log(f"source: {source}")
    log(f"output: {output}")

    stage = Usd.Stage.CreateNew(output)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, args_cli.root_prim).GetPrim()
    stage.SetDefaultPrim(root)
    support_center = parse_vec3(args_cli.bottom_support_center, "--bottom-support-center")
    support_size = parse_vec3(args_cli.bottom_support_size, "--bottom-support-size")
    if (support_center is None) != (support_size is None):
        raise ValueError("--bottom-support-center and --bottom-support-size must be used together")
    if support_center is not None and not args_cli.bake_first_mesh:
        raise ValueError("bottom support requires --bake-first-mesh")

    if args_cli.bake_first_mesh:
        source_stage = Usd.Stage.Open(source)
        if source_stage is None:
            raise RuntimeError(f"Could not open source stage: {source}")
        source_root = source_stage.GetDefaultPrim()
        source_mesh = next(
            (prim for prim in Usd.PrimRange(source_root) if prim.IsA(UsdGeom.Mesh)),
            None,
        )
        if source_mesh is None:
            raise RuntimeError(f"No source mesh found to bake under: {source_root.GetPath()}")
        source_geom = UsdGeom.Mesh(source_mesh)
        source_units = float(UsdGeom.GetStageMetersPerUnit(source_stage))
        source_to_world = UsdGeom.XformCache().GetLocalToWorldTransform(source_mesh)
        points = [
            Gf.Vec3f(*(float(value) * source_units for value in source_to_world.Transform(point)))
            for point in source_geom.GetPointsAttr().Get()
        ]
        counts = list(source_geom.GetFaceVertexCountsAttr().Get())
        indices = list(source_geom.GetFaceVertexIndicesAttr().Get())

        if support_center is not None and support_size is not None:
            cx, cy, cz = support_center
            hx, hy, hz = (0.5 * value for value in support_size)
            offset = len(points)
            points.extend(
                Gf.Vec3f(x, y, z)
                for x, y, z in (
                    (cx - hx, cy - hy, cz - hz),
                    (cx + hx, cy - hy, cz - hz),
                    (cx + hx, cy - hy, cz + hz),
                    (cx - hx, cy - hy, cz + hz),
                    (cx - hx, cy + hy, cz - hz),
                    (cx + hx, cy + hy, cz - hz),
                    (cx + hx, cy + hy, cz + hz),
                    (cx - hx, cy + hy, cz + hz),
                )
            )
            counts.extend([4, 4, 4, 4, 4, 4])
            indices.extend(
                offset + index
                for index in (0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 4, 0, 3, 7)
            )

        baked_mesh = UsdGeom.Mesh.Define(stage, f"{args_cli.root_prim}/CollisionMesh")
        baked_mesh.CreatePointsAttr(points)
        baked_mesh.CreateFaceVertexCountsAttr(counts)
        baked_mesh.CreateFaceVertexIndicesAttr(indices)
        baked_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    else:
        root.GetReferences().AddReference(source)
        # Resolve the referenced asset before traversing composed child meshes.
        stage.Load(args_cli.root_prim)

    mesh_count = 0
    mesh_paths = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            if apply_collision_to_mesh(
                prim,
                args_cli.approximation,
                args_cli.contact_offset,
                args_cli.rest_offset,
            ):
                mesh_count += 1
                mesh_paths.append(str(prim.GetPath()))

    if mesh_count == 0:
        raise RuntimeError(
            f"No UsdGeom.Mesh prims found under {args_cli.root_prim}. "
            "Check that --source points to a USD containing visual meshes."
        )

    if not args_cli.no_rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass_api = UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr().Set(float(args_cli.mass))

        try:
            from pxr import PhysxSchema
        except ImportError:
            pass
        else:
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
            physx_rigid.CreateDisableGravityAttr().Set(False)

    stage.GetRootLayer().Save()
    log(f"meshes: {mesh_count}")
    for mesh_path in mesh_paths[:10]:
        log(f"mesh: {mesh_path}")
    if len(mesh_paths) > 10:
        log(f"mesh: ... {len(mesh_paths) - 10} more")
    log(f"approximation: {args_cli.approximation}")
    mass = args_cli.mass if not args_cli.no_rigid_body else "not applied"
    log(f"mass: {mass}")
    log("done")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
