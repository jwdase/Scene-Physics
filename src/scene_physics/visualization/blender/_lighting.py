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

import bpy
from mathutils import Vector

_LIGHT_NAMES = ("studio_key", "studio_fill", "studio_rim")
GROUND_NAME = "ground_plane"


def add_ground_plane(
    z: float, size: float = 30.0, color=(0.62, 0.62, 0.62, 1.0), roughness: float = 0.55
) -> bpy.types.Object:
    """A large neutral floor at world-z `z` (catches shadows, grounds the scene)."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z))
    plane = bpy.context.active_object
    plane.name = GROUND_NAME

    mat = bpy.data.materials.new("ground_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    plane.data.materials.append(mat)
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
    center=(0.0, 0.0, 0.75), key=450.0, fill=60.0, rim=180.0
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
