"""Studio lighting rig + ground plane for photoreal tabletop renders.

The HDRI world (see _materials.setup_world_hdri) gives soft ambient fill and the
reflections that sell glass/metal/ceramic. On top of that we add a classic studio
3-light rig of large area "softboxes" (key/fill/rim) for shaping and a believable
primary shadow, plus a ground plane so the table stands on a surface and casts
contact shadows instead of floating. Together these are the biggest photorealism
wins available without touching geometry.

bpy + stdlib only. The ground plane + lights must be hidden during the
segmentation ID pass (they are not tracked objects) -- see hide_for_id_pass().
"""

from __future__ import annotations

import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _materials  # noqa: E402

_LIGHT_NAMES = ("studio_key", "studio_fill", "studio_rim")
GROUND_NAME = "ground_plane"

# Real tile PBR set for the floor (box-projected, no UVs). Clean uniform tile grid,
# lightened (gamma<1) from near-black to a calm light gray so it doesn't compete
# with the objects. See _materials.build_textured_material.
FLOOR_TEX = {
    "set": "Tiles108",
    "prefix": "Tiles108_4K-JPG",
    "scale": 0.45,
    "bump": 0.4,
    "coat": 0.15,
    "gamma": 0.4,
}


def _tile_floor_material(scale: float = 9.0) -> bpy.types.Material:
    """A procedural square-tile stone floor: aligned tile grid + recessed grout.

    Tiles read as a real kitchen/dining floor and are far easier to make
    convincing than procedural wood. Uses a Brick texture (offset=0 -> grid) on
    Generated coords; the mortar mask drives both the grout color and a groove bump.
    """
    mat = bpy.data.materials.new("floor_tile")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texco = nt.nodes.new("ShaderNodeTexCoord")
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.0  # aligned grid, not running-bond brick
    brick.inputs["Scale"].default_value = scale
    brick.inputs["Brick Width"].default_value = 0.5
    brick.inputs["Row Height"].default_value = 0.5  # == width -> square tiles
    brick.inputs["Mortar Size"].default_value = 0.012
    brick.inputs["Mortar Smooth"].default_value = 0.1
    # warm light stone tiles, slight tile-to-tile variation; dark grout
    brick.inputs["Color1"].default_value = (0.60, 0.58, 0.54, 1.0)
    brick.inputs["Color2"].default_value = (0.54, 0.52, 0.48, 1.0)
    brick.inputs["Mortar"].default_value = (0.16, 0.15, 0.14, 1.0)
    nt.links.new(texco.outputs["Generated"], brick.inputs["Vector"])
    nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.30  # polished stone/ceramic tile
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.2

    # Grout recessed: Fac is 1 at mortar, so feed (1 - Fac) as height -> tiles high.
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(brick.outputs["Fac"], inv.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(inv.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def add_ground_plane(z: float, size: float = 14.0) -> bpy.types.Object:
    """A tiled stone floor at world-z `z` (catches shadows, grounds the scene).

    Uses a real CC0 tile PBR set box-projected; `_tile_floor_material` remains as a
    no-asset procedural fallback.
    """
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z))
    plane = bpy.context.active_object
    plane.name = GROUND_NAME
    plane.data.materials.append(_materials.build_textured_material(FLOOR_TEX))
    return plane


def _area_light(
    name, location, target, energy, size, color=(1.0, 1.0, 1.0)
) -> bpy.types.Object:
    ld = bpy.data.lights.new(name, type="AREA")
    ld.shape = "RECTANGLE"
    ld.size = size
    ld.size_y = size
    ld.energy = energy
    ld.color = color
    obj = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    direction = Vector(target) - Vector(location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return obj


def setup_studio_lighting(
    center=(0.0, 0.0, 0.75), key=220.0, fill=40.0, rim=90.0
) -> list[bpy.types.Object]:
    """Soft key (front-left), gentle fill (front-right), rim (back) aimed at `center`.

    Key-dominant with weak fill so objects get real shaping + soft shadows rather
    than flat, overexposed product lighting. The HDRI supplies reflections + a
    little ambient on top (see render_scene world_strength)."""
    return [
        _area_light(
            _LIGHT_NAMES[0],
            (-1.3, -1.5, 2.1),
            center,
            key,
            size=2.0,
            color=(1.0, 0.98, 0.95),
        ),  # slightly warm key
        _area_light(
            _LIGHT_NAMES[1],
            (1.7, -1.1, 1.2),
            center,
            fill,
            size=2.6,
            color=(0.96, 0.98, 1.0),
        ),  # slightly cool fill
        _area_light(_LIGHT_NAMES[2], (0.6, 1.7, 1.9), center, rim, size=1.6),
    ]


def hide_for_id_pass() -> None:
    """Hide the ground plane and rig from rendering so the segmentation pass shows
    only the tracked (emission-shaded) objects on a black background."""
    for name in (GROUND_NAME, *_LIGHT_NAMES):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = True
