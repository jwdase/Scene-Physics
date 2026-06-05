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

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_specs import DEFAULT as _DEFAULT  # noqa: E402
from material_specs import MATERIAL_SPECS  # noqa: E402


def _set(bsdf, names, value) -> None:
    for n in names:
        if n in bsdf.inputs:
            bsdf.inputs[n].default_value = value
            return


def build_material(name: str, spec: dict) -> bpy.types.Material:
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
        texco = nt.nodes.new("ShaderNodeTexCoord")
        if kind == "wood":
            tex = nt.nodes.new("ShaderNodeTexWave")
            tex.inputs["Scale"].default_value = scale
            tex.inputs["Distortion"].default_value = 2.0
        elif kind == "voronoi":
            tex = nt.nodes.new("ShaderNodeTexVoronoi")
            tex.inputs["Scale"].default_value = scale
        else:
            tex = nt.nodes.new("ShaderNodeTexNoise")
            tex.inputs["Scale"].default_value = scale
        nt.links.new(texco.outputs["Generated"], tex.inputs["Vector"])
        bumpn = nt.nodes.new("ShaderNodeBump")
        bumpn.inputs["Strength"].default_value = strength
        height = tex.outputs["Fac"] if "Fac" in tex.outputs else tex.outputs[0]
        nt.links.new(height, bumpn.inputs["Height"])
        nt.links.new(bumpn.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def has_spec(name: str) -> bool:
    return name in MATERIAL_SPECS


def assign_material(obj: bpy.types.Object, name: str) -> None:
    spec = MATERIAL_SPECS.get(name, _DEFAULT)
    mat = build_material(name, spec)
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_world_hdri(hdri_path: str, strength: float = 1.0) -> None:
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
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
