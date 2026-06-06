"""Shared Blender-side scene helpers: import a generated scene USD and recover the
{object_name: blender_object} mapping. bpy + stdlib only (no scene_physics).

The generated USDs author `/root/<safe>/<safe>` (Xform holding the settled pose ->
Mesh). Blender's USD importer bakes the pose into the mesh object's world matrix
and names objects after the prim leaf (deduping a name clash between the Xform and
its child Mesh with a `.NNN` suffix). We recover the original object name by
stripping that suffix and matching against the known scene names.
"""

from __future__ import annotations

import re

import bpy


def safe_usd_name(name: str) -> str:
    """Mirror of data_gen/usd_export.py::safe_usd_name (kept in sync by hand)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not s or not re.match(r"[A-Za-z_]", s[0]):
        s = "_" + s
    return s


def reset_scene() -> None:
    """Empty the .blend so repeated imports don't accumulate."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_scene_usd(usd_path: str, expected_names: list[str]) -> dict:
    """Import the USD and return {original_name: mesh object} for expected names."""
    bpy.ops.wm.usd_import(filepath=usd_path)

    safe_to_orig = {safe_usd_name(n): n for n in expected_names}
    mapping: dict[str, bpy.types.Object] = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        leaf = obj.name.split(".")[0]  # strip Blender's .001 dedup suffix
        orig = safe_to_orig.get(leaf)
        if orig is not None and orig not in mapping:
            mapping[orig] = obj
    return mapping
