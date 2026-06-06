"""Segmentation pass: flat per-object emission colors -> a clean ID render.

Rather than route an Object-Index EXR pass (the uv env has no EXR reader), we
re-shade every object with a pure Emission of a unique, well-separated hue, kill
all world lighting, and render one crisp sample per pixel. The uv side then
classifies each pixel to the nearest palette color (see visualization/
segmentation.py) -- robust because the hues are far apart in color space.

bpy + stdlib only.
"""

from __future__ import annotations

import colorsys

import bpy


def palette(n: int) -> list[tuple[float, float, float]]:
    """`n` maximally-spaced saturated hues (linear RGB)."""
    return [colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 1.0) for i in range(n)]


def apply_id_materials(mapping: dict, name_to_id: dict, pal: list) -> None:
    for name, obj in mapping.items():
        col = pal[name_to_id[name] - 1]
        mat = bpy.data.materials.new(f"id_{name}")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (col[0], col[1], col[2], 1.0)
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def configure_id_render(scene) -> None:
    """One crisp sample/pixel, no denoise, black background, no tone-mapping."""
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    bpy.context.view_layer.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.render.film_transparent = False

    world = scene.world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Standard view transform: emission color c -> sRGB(c) in the 8-bit PNG, which
    # the uv decoder mirrors. (AgX would distort the flat ID colors.)
    scene.view_settings.view_transform = "Standard"
