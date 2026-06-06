"""Blender entry-point: render ONE scene from many camera positions (beauty only).

For exploring natural viewpoints of the table. Imports the scene + builds
materials/lighting/floor once, then renders from each camera in `views`.

    blender --background --python render_views.py -- <job.json>

job.json: like render_scene's, but with "views": [intrinsics_dict, ...] instead of
a single camera, and no segmentation pass. Writes <out_dir>/view_NN.png.
"""

from __future__ import annotations

import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lighting  # noqa: E402
import _materials  # noqa: E402
from _camera import setup_camera  # noqa: E402
from _scene import import_scene_usd, reset_scene  # noqa: E402
from render_scene import (  # noqa: E402
    _scene_min_z,
    _shade_auto,
    configure_cycles,
    render_png,
)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:]
    with open(argv[0]) as f:
        job = json.load(f)

    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    reset_scene()
    mapping = import_scene_usd(job["usd"], job["names"])
    for name, obj in mapping.items():
        _shade_auto(obj)
        _materials.assign_material(obj, name)

    _materials.setup_world_hdri(
        job["hdri"], job.get("world_strength", 1.0), job.get("hdri_rotation", 0.0)
    )
    # Bigger floor so low/eye-level angles don't see the plane edge.
    _lighting.add_ground_plane(_scene_min_z(mapping.values()), size=40.0)
    _lighting.setup_studio_lighting()

    scene = bpy.context.scene
    configure_cycles(scene, int(job.get("samples", 160)), job.get("device", "GPU"))
    scene.view_settings.view_transform = job.get("view_transform", "AgX")

    names = job.get("view_names")
    for i, intr in enumerate(job["views"]):
        setup_camera(intr)
        base = names[i] if names else f"view_{i + 1:02d}"
        render_png(scene, os.path.join(out_dir, f"{base}.png"))


if __name__ == "__main__":
    main()
