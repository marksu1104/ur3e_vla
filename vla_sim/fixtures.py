"""Recognizable destination fixtures used by the remote demonstration."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from vla_sim.config import (
    BOWL_STACK_Z_OFFSETS,
    CUTLERY_BOX_CENTER_X_OFFSET,
    CUTLERY_BOX_CENTER_Y_OFFSET,
    CUTLERY_BOX_REFERENCE_SPOON_OFFSET,
    CUTLERY_BOX_SIZE,
    CUTLERY_BOX_SPOON_SCALE,
    CUTLERY_BOX_WALL_HEIGHT,
    CUTLERY_BOX_WALL_THICKNESS,
    DESTINATION_PROP_COLORS,
    MUG_COASTER_HEIGHT,
    MUG_COASTER_RADIUS,
    PLACE_POSITIONS,
    REFERENCE_COASTER_OFFSET,
    REFERENCE_MUG_Z_OFFSET,
    TABLE_MAT_A_POS,
    TABLE_MAT_SIZE,
    TARGETS,
)
from vla_sim.scene import quat_wxyz_to_isaac
from vla_sim.visuals import set_plastic_material


ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


def prepare_destination_fixtures(stage) -> None:
    """Create recognizable, fixed goal fixtures for the remote demonstration."""
    from pxr import UsdGeom

    root_path = "/World/DestinationFixtures"
    group_paths = {
        "box": f"{root_path}/CutleryBox",
        "box_inside": f"{root_path}/CutleryBoxInside",
        "spoons": f"{root_path}/CutlerySpoons",
        "coasters": f"{root_path}/MugCoasters",
        "reference_mug": f"{root_path}/ReferenceMug",
        "stack_bowl": f"{root_path}/StackBowls",
    }
    bowl_collider_path = f"{root_path}/StackBowlCollider"
    UsdGeom.Xform.Define(stage, root_path)
    for path in group_paths.values():
        UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xform.Define(stage, bowl_collider_path)

    table_z = TABLE_MAT_A_POS[2] + 0.5 * TABLE_MAT_SIZE[2]
    cutlery_x, cutlery_y = PLACE_POSITIONS[0]
    mug_goal_x, mug_goal_y = PLACE_POSITIONS[1]
    bowl_x, bowl_y = PLACE_POSITIONS[2]
    box_x_center = cutlery_x + CUTLERY_BOX_CENTER_X_OFFSET
    box_y_center = cutlery_y + CUTLERY_BOX_CENTER_Y_OFFSET

    wall = CUTLERY_BOX_WALL_THICKNESS
    wall_height = CUTLERY_BOX_WALL_HEIGHT
    box_x, box_y, box_z = CUTLERY_BOX_SIZE
    cuboids = (
        (
            group_paths["box"] + "/Base",
            CUTLERY_BOX_SIZE,
            (box_x_center, box_y_center, table_z + 0.5 * box_z),
        ),
        (
            group_paths["box_inside"] + "/Floor",
            (box_x - 2 * wall, box_y - 2 * wall, 0.001),
            (box_x_center, box_y_center, table_z + box_z + 0.0005),
        ),
        (
            group_paths["box"] + "/WallFront",
            (box_x, wall, wall_height),
            (
                box_x_center,
                box_y_center - 0.5 * (box_y - wall),
                table_z + 0.5 * wall_height,
            ),
        ),
        (
            group_paths["box"] + "/WallBack",
            (box_x, wall, wall_height),
            (
                box_x_center,
                box_y_center + 0.5 * (box_y - wall),
                table_z + 0.5 * wall_height,
            ),
        ),
        (
            group_paths["box"] + "/WallLeft",
            (wall, box_y, wall_height),
            (
                box_x_center - 0.5 * (box_x - wall),
                box_y_center,
                table_z + 0.5 * wall_height,
            ),
        ),
        (
            group_paths["box"] + "/WallRight",
            (wall, box_y, wall_height),
            (
                box_x_center + 0.5 * (box_x - wall),
                box_y_center,
                table_z + 0.5 * wall_height,
            ),
        ),
    )
    fixed_rigid = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=True,
        disable_gravity=True,
    )
    fixed_collision = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.002,
        rest_offset=0.0,
    )

    for path, size, position in cuboids:
        # The inset floor is a thin color layer over the solid base. Giving
        # both overlapping surfaces a collider would create duplicate contact
        # planes, so only the base and four walls carry physics.
        is_inset_floor = path.endswith("/Floor")
        cfg = sim_utils.CuboidCfg(
            size=size,
            rigid_props=None if is_inset_floor else fixed_rigid,
            collision_props=None if is_inset_floor else fixed_collision,
        )
        cfg.func(path, cfg, translation=position)

    spoon_scale = CUTLERY_BOX_SPOON_SCALE
    spoon_cfg = sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / TARGETS["spoon"]["collision_usd"]),
        scale=(spoon_scale, spoon_scale, spoon_scale),
        rigid_props=fixed_rigid,
        collision_props=fixed_collision,
    )
    # One reference spoon occupies the rear slot. It faces the opposite way so
    # the matching spoon can nest head-to-tail in a more compact box.
    reference_spoon_x, reference_spoon_y = CUTLERY_BOX_REFERENCE_SPOON_OFFSET
    spoon_cfg.func(
        group_paths["spoons"] + "/Spoon",
        spoon_cfg,
        translation=(
            box_x_center + reference_spoon_x,
            box_y_center + reference_spoon_y,
            table_z + 0.017,
        ),
        orientation=quat_wxyz_to_isaac((0.7071068, 0.0, 0.0, 0.7071068)),
    )

    mug = TARGETS["red_mug"]
    reference_coaster_position = (
        mug_goal_x + REFERENCE_COASTER_OFFSET[0],
        mug_goal_y + REFERENCE_COASTER_OFFSET[1],
    )
    # The YCB origin includes the handle. Offset the root so the cylindrical
    # mug body, rather than the full-mesh centroid, sits on the coaster center.
    reference_mug_position = (
        reference_coaster_position[0] - float(mug.get("x_nudge", 0.0)),
        reference_coaster_position[1] - float(mug.get("y_nudge", 0.0)),
        table_z + REFERENCE_MUG_Z_OFFSET,
    )
    coaster_cfg = sim_utils.MeshCylinderCfg(
        radius=MUG_COASTER_RADIUS,
        height=MUG_COASTER_HEIGHT,
        rigid_props=fixed_rigid,
        collision_props=fixed_collision,
    )
    for name, x, y in (
        ("Destination", mug_goal_x, mug_goal_y),
        ("Reference", *reference_coaster_position),
    ):
        coaster_cfg.func(
            group_paths["coasters"] + f"/{name}",
            coaster_cfg,
            translation=(x, y, table_z + 0.5 * MUG_COASTER_HEIGHT),
        )

    mug_cfg = sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / mug["collision_usd"]),
        rigid_props=fixed_rigid,
        collision_props=fixed_collision,
    )
    mug_cfg.func(
        group_paths["reference_mug"] + "/Mug",
        mug_cfg,
        translation=reference_mug_position,
        orientation=quat_wxyz_to_isaac(mug["spawn_rot"]),
    )

    bowl = TARGETS["bowl"]
    bowl_scale = float(bowl["scale"])
    bowl_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/{bowl['usd_relative']}",
        scale=(bowl_scale,) * 3,
    )
    bowl_collision = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.003,
        rest_offset=0.001,
    )
    bowl_collider_cfg = sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / bowl["collision_usd"]),
        scale=(bowl_scale,) * 3,
        rigid_props=fixed_rigid,
        collision_props=bowl_collision,
    )
    # Two identical physical bowls start neatly nested. Contact margins make
    # identical rim shells meet too early, so the upper receiving shell is
    # inset to the bowl's usable interior depth. This produces the same 20 mm
    # spacing as the visible pair while still stopping the third bowl before
    # their rendered surfaces intersect.
    receiving_shell_inset = -0.013
    for index, z_offset in enumerate(BOWL_STACK_Z_OFFSETS):
        bowl_cfg.func(
            group_paths["stack_bowl"] + f"/Bowl{index}",
            bowl_cfg,
            translation=(bowl_x, bowl_y, table_z + z_offset),
            orientation=quat_wxyz_to_isaac(bowl["spawn_rot"]),
        )
        bowl_collider_cfg.func(
            bowl_collider_path + f"/Bowl{index}",
            bowl_collider_cfg,
            translation=(
                bowl_x,
                bowl_y,
                table_z + z_offset + (receiving_shell_inset if index else 0.0),
            ),
            orientation=quat_wxyz_to_isaac(bowl["spawn_rot"]),
        )

    sim_utils.make_uninstanceable(root_path, stage)
    for name, path in group_paths.items():
        set_plastic_material(stage, path, DESTINATION_PROP_COLORS[name])
    UsdGeom.Imageable(stage.GetPrimAtPath(bowl_collider_path)).GetVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )
