"""Visual-only landmark cube attached near the gripper.

Pure render aid for tracking gripper location in footage; participates in
zero physics: no UsdPhysics.CollisionAPI, no RigidBodyAPI are ever applied
to it, so it can never collide with anything and PhysX never simulates it.
It rigidly follows whatever prim it's parented under purely through the
USD parent-child transform hierarchy -- the same mechanism any other
render-only child mesh on a link already uses (see
vla_sim.scene.bind_gripper_pad_visuals for a similar existing pattern).
"""

from __future__ import annotations

GRIPPER_MARKER_COLOR = (0.0, 0.9, 0.0)  # bright green
GRIPPER_MARKER_SIZE = 1.0  # 1 m cube edge length
GRIPPER_MARKER_SHAPE = (0.05, 0.025, 0.025) # cube transorm
# Local offset from the parent prim's origin, in the parent's local frame.
# Positive Z here pushes the marker out from wrist_3_link toward where the
# gripper fingers actually are (the gripper itself isn't a fixed offset in
# config.py, so this is a starting guess -- nudge and re-check visually).
GRIPPER_MARKER_LOCAL_OFFSET = (0.00, 0.0, 0.05)


def attach_gripper_marker(
    stage,
    parent_prim_path: str,
    *,
    color: tuple[float, float, float] = GRIPPER_MARKER_COLOR,
    size: float = GRIPPER_MARKER_SIZE,
    shape: tuple[float, float, float] = GRIPPER_MARKER_SHAPE,
    local_offset: tuple[float, float, float] = GRIPPER_MARKER_LOCAL_OFFSET,
    marker_name: str = "GripperMarker",
) -> str:
    """Create a bright, collision-free cube rigidly parented under parent_prim_path.

    Returns the new prim's path.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    marker_path = f"{parent_prim_path}/{marker_name}"
    cube = UsdGeom.Cube.Define(stage, marker_path)
    cube.GetSizeAttr().Set(size)

    # Local translate only -- no physics ops of any kind are applied to
    # this prim, so it is purely along for the ride on the parent's pose.
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*local_offset))
    xform.AddScaleOp().Set(Gf.Vec3f(*shape))

    material_path = "/World/Looks/GripperMarkerMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    # Emissive so it stays bright/visible regardless of scene lighting.
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    cube.CreateDisplayColorAttr([color])
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    return marker_path
