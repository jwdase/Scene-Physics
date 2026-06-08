"""Studio lighting rig + ground plane + backdrop wall for photoreal tabletop renders.

The HDRI world (see _materials.setup_world_hdri) gives soft ambient fill and the
reflections that sell glass/metal/ceramic. On top of that we add a classic studio
3-light rig of large area "softboxes" (key/fill/rim) for shaping and a believable
primary shadow, a ground plane so the table stands on a surface and casts contact
shadows, and a near backdrop wall standing on that ground plane so the visible
background is a real surface meeting the floor at a clean corner -- not the
infinite HDRI room floating behind the scene. Together these are the biggest
photorealism wins available without touching geometry.

bpy + stdlib only. The ground plane, wall, and lights must be hidden during the
segmentation ID pass (they are not tracked objects) -- see hide_for_id_pass().
"""

from __future__ import annotations

import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _materials  # noqa: E402

_LIGHT_NAMES = ("studio_key", "studio_fill", "studio_rim")
GROUND_NAME = "ground_plane"
WALL_NAME = "back_wall"

# Real tile PBR set for the floor (box-projected, no UVs). Clean uniform tile grid,
# lightened (gamma<1) from near-black to a calm light gray so it doesn't compete
# with the objects. See _materials.build_textured_material.
FLOOR_TEX = {
    "set": "Tiles108",
    "prefix": "Tiles108_4K-JPG",
    "scale": 0.45,
    "bump": 0.4,
    "coat": 0.0,
    "rough_floor": 0.45,  # matte tile, no specular glare competing with the objects
    "gamma": 0.4,
}

# Backdrop wall: a neutral warm-grey matte surface, brought in close behind the
# table and standing ON the ground plane. Kept desaturated/matte so it never
# competes with the (now vividly, uniquely colored) dataset objects.
WALL_COLOR = (0.62, 0.60, 0.57)
WALL_ROUGHNESS = 0.9
WALL_Y = 2.2  # distance behind world origin -> "wall brought closer", not at infinity


def add_ground_plane(z: float, size: float = 60.0) -> bpy.types.Object:
    """A tiled stone floor at world-z `z` (catches shadows, grounds the scene).

    Uses a real CC0 tile PBR set (FLOOR_TEX) box-projected. The plane is large
    enough (60 u) that its far edge reaches the camera's horizon, so the *actual*
    tile floor -- not the HDRI room's own baked floor -- fills the frame up to the
    horizon line. That removes the seam where a small plane used to end and the
    HDRI floor showed through at a different tone/height. Tile size is fixed in
    world units (Object-coord mapping), so enlarging tiles more, never stretches.
    """
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z))
    plane = bpy.context.active_object
    plane.name = GROUND_NAME
    plane.data.materials.append(_materials.build_textured_material(FLOOR_TEX))
    return plane


def add_back_wall(z: float, y: float = WALL_Y, size: float = 24.0) -> bpy.types.Object:
    """A large vertical matte wall standing ON the ground plane and aligned with it.

    The wall's bottom edge sits exactly at world-z `z` (the floor height), a short
    distance `y` behind the scene, so floor + wall meet at one clean corner line and
    the visible background is a near wall instead of the infinite HDRI room. The
    plane (size `size`) is big enough to overflow the frame in width and height; the
    HDRI still lights the wall but is occluded from view. Hidden during the ID pass.
    """
    # A default plane lies in XY (normal +Z). Stand it up by rotating +90 deg about
    # X: the normal becomes -Y (faces the camera) and the plane's local-Y maps to
    # world +Z. Placing the center at z + size/2 puts the bottom edge exactly at z.
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, y, z + size / 2.0))
    wall = bpy.context.active_object
    wall.name = WALL_NAME
    wall.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    mat = bpy.data.materials.new("back_wall")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:  # name varies across versions; fall back to type
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    _materials._set(bsdf, ["Base Color"], (*WALL_COLOR, 1.0))
    _materials._set(bsdf, ["Roughness"], WALL_ROUGHNESS)
    _materials._set(bsdf, ["Specular IOR Level", "Specular"], 0.2)  # no wall glare
    wall.data.materials.append(mat)
    return wall


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
    """Hide the ground plane, backdrop wall, and rig from rendering so the
    segmentation pass shows only the tracked (emission-shaded) objects on a black
    background."""
    for name in (GROUND_NAME, WALL_NAME, *_LIGHT_NAMES):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = True
