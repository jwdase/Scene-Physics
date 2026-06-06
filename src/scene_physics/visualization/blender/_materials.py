"""Procedural PBR materials for the dataset objects + studio HDRI world.

The source meshes have NO UVs (see project notes), so every material is built
from a Principled BSDF plus, where useful, a procedural bump driven by *Generated*
texture coordinates (bounding-box derived, no UV needed). This yields convincing
surfaces for generic items (wood, ceramic, glass, metal, fruit, stone) but cannot
reproduce branded/figurative decals -- that was an accepted limitation.

Specs are intentionally simple and hand-tuned; the look is meant to be reviewed
visually and iterated. Socket names follow Blender 4.x Principled (e.g.
"Transmission Weight", "Subsurface Weight"); `_set` tolerates older names too.

bpy + stdlib only.
"""

from __future__ import annotations

import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_specs import DEFAULT as _DEFAULT  # noqa: E402
from material_specs import MATERIAL_SPECS  # noqa: E402

# Global gain on bump strength so procedural relief actually reads under the soft
# studio lighting. Per-material albedo/roughness/contrast knobs have defaults below.
BUMP_GAIN = 2.5

# Real CC0 PBR texture sets live here (repo/resources/textures/<set>/). Resolved
# relative to this file so it works regardless of cwd / worktree.
_TEX_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "resources",
        "textures",
    )
)


def _set(bsdf, names, value) -> None:
    for n in names:
        if n in bsdf.inputs:
            bsdf.inputs[n].default_value = value
            return


def _box_image(nt, mapping, path, non_color, blend):
    """An Image Texture node BOX-projected (triplanar) -- needs no UVs."""
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(path)
    node.projection = "BOX"
    node.projection_blend = blend
    if non_color:
        node.image.colorspace_settings.name = "Non-Color"
    nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    return node


def build_textured_material(tex: dict) -> bpy.types.Material:
    """Real PBR maps (Color/Roughness/Displacement) box-projected onto a UV-less
    mesh. tex = {"set","prefix","scale","blend","bump","coat"}. Displacement drives
    a Bump (not a tangent-space normal map, which is unreliable without UVs)."""
    folder = os.path.join(_TEX_ROOT, tex["set"])
    prefix = tex["prefix"]
    blend = tex.get("blend", 0.3)
    s = tex.get("scale", 1.0)

    mat = bpy.data.materials.new(f"tex_{tex['set']}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texco = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (s, s, s)
    nt.links.new(texco.outputs["Object"], mapping.inputs["Vector"])  # real-world scale

    col = _box_image(
        nt, mapping, os.path.join(folder, f"{prefix}_Color.jpg"), False, blend
    )
    gamma = tex.get(
        "gamma", 1.0
    )  # <1 lightens (e.g. recolor a dark tile to light gray)
    if gamma != 1.0:
        g = nt.nodes.new("ShaderNodeGamma")
        g.inputs["Gamma"].default_value = gamma
        nt.links.new(col.outputs["Color"], g.inputs["Color"])
        nt.links.new(g.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    rough = _box_image(
        nt, mapping, os.path.join(folder, f"{prefix}_Roughness.jpg"), True, blend
    )
    nt.links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    disp = _box_image(
        nt, mapping, os.path.join(folder, f"{prefix}_Displacement.jpg"), True, blend
    )
    bumpn = nt.nodes.new("ShaderNodeBump")
    bumpn.inputs["Strength"].default_value = tex.get("bump", 0.4)
    nt.links.new(disp.outputs["Color"], bumpn.inputs["Height"])
    nt.links.new(bumpn.outputs["Normal"], bsdf.inputs["Normal"])
    _set(bsdf, ["Coat Weight", "Clearcoat"], tex.get("coat", 0.0))
    return mat


def build_material(name: str, spec: dict) -> bpy.types.Material:
    if spec.get("textures"):
        return build_textured_material(spec["textures"])

    mat = bpy.data.materials.new(name=f"mat_{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    _set(bsdf, ["Base Color"], spec.get("base_color", _DEFAULT["base_color"]))
    _set(bsdf, ["Metallic"], spec.get("metallic", 0.0))
    _set(bsdf, ["Roughness"], spec.get("roughness", 0.5))
    _set(bsdf, ["IOR"], spec.get("ior", 1.45))
    _set(bsdf, ["Transmission Weight", "Transmission"], spec.get("transmission", 0.0))
    _set(bsdf, ["Subsurface Weight", "Subsurface"], spec.get("subsurface", 0.0))
    _set(bsdf, ["Coat Weight", "Clearcoat"], spec.get("coat", 0.0))
    _set(bsdf, ["Specular IOR Level", "Specular"], spec.get("spec", 0.5))

    bump = spec.get("bump")
    if bump:
        kind, scale, strength = bump
        base = spec.get("base_color", _DEFAULT["base_color"])
        rough = spec.get("roughness", 0.5)
        contrast = spec.get("tex_contrast", 0.5)  # 0..1, pattern sharpness
        albedo_var = spec.get("albedo_var", 0.55)  # how much darker the valleys get
        rough_var = spec.get("rough_var", 0.18)  # +/- roughness across the pattern

        texco = nt.nodes.new("ShaderNodeTexCoord")
        coord = texco.outputs["Generated"]
        if kind == "wood":
            # Natural wood grain without UVs: stretch the coords hard along the grain
            # direction so a fractal noise reads as long wavy streaks, not bands.
            mapping = nt.nodes.new("ShaderNodeMapping")
            aniso = spec.get("wood_aniso", 0.12)  # lower = longer streaks
            mapping.inputs["Scale"].default_value = (1.0, aniso, 1.0)
            nt.links.new(coord, mapping.inputs["Vector"])
            tex = nt.nodes.new("ShaderNodeTexNoise")
            tex.inputs["Scale"].default_value = scale
            tex.inputs["Detail"].default_value = 5.0
            tex.inputs["Roughness"].default_value = 0.6
            nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        elif kind == "voronoi":
            tex = nt.nodes.new("ShaderNodeTexVoronoi")
            tex.inputs["Scale"].default_value = scale
            nt.links.new(coord, tex.inputs["Vector"])
        else:
            tex = nt.nodes.new("ShaderNodeTexNoise")
            tex.inputs["Scale"].default_value = scale
            tex.inputs["Detail"].default_value = 4.0
            nt.links.new(coord, tex.inputs["Vector"])
        fac = tex.outputs["Fac"] if "Fac" in tex.outputs else tex.outputs[0]

        # Albedo contrast: darken the pattern's valleys. The ColorRamp stops double
        # as a contrast knob -- the closer together, the punchier the transition.
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        cr = ramp.color_ramp
        half = 0.5 * (1.0 - contrast)
        cr.elements[0].position = max(0.0, 0.5 - half)
        cr.elements[1].position = min(1.0, 0.5 + half + 1e-3)
        cr.elements[0].color = (*(c * (1.0 - albedo_var) for c in base[:3]), 1.0)
        cr.elements[1].color = (base[0], base[1], base[2], 1.0)
        nt.links.new(fac, ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        # Roughness variation: gloss differs across peaks/valleys -> specular contrast.
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = max(0.0, rough - rough_var)
        mr.inputs["To Max"].default_value = min(1.0, rough + rough_var)
        nt.links.new(fac, mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

        # Stronger relief so the texture is visible under flat light.
        bumpn = nt.nodes.new("ShaderNodeBump")
        bumpn.inputs["Strength"].default_value = min(1.0, strength * BUMP_GAIN)
        nt.links.new(fac, bumpn.inputs["Height"])
        nt.links.new(bumpn.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def has_spec(name: str) -> bool:
    return name in MATERIAL_SPECS


def assign_material(obj: bpy.types.Object, name: str) -> None:
    spec = MATERIAL_SPECS.get(name, _DEFAULT)
    mat = build_material(name, spec)
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_world_hdri(
    hdri_path: str, strength: float = 1.0, rotation_deg: float = 0.0
) -> None:
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path)
    bg.inputs["Strength"].default_value = strength

    # Rotation control so the nicest part of the room sits behind the table.
    texco = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
    nt.links.new(texco.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
