"""Blender entry-point: render a set of axis-aligned boxes to an RGBA PNG.

Used only by the camera-equivalence IoU test -- it produces a silhouette (via the
alpha channel of a transparent film) to compare against the Newton sensor mask.

    blender --background --python render_boxes.py -- <job.json> <out.png>

job.json: {"intrinsics": {...},
           "boxes": [{"pos":[x,y,z], "half":[hx,hy,hz]}, ...]}
"""

from __future__ import annotations

import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _camera import setup_camera  # noqa: E402


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.cameras):
        for block in list(coll):
            coll.remove(block)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    job_path, out_png = argv[0], argv[1]
    with open(job_path) as f:
        job = json.load(f)

    _clear_scene()
    setup_camera(job["intrinsics"])

    for b in job["boxes"]:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=tuple(b["pos"]))
        obj = bpy.context.active_object
        obj.scale = tuple(b["half"])  # size=2 cube has half-extent 1 -> scale == half

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False  # this Blender build has no OpenImageDenoiser
    bpy.context.view_layer.cycles.use_denoising = False
    scene.render.film_transparent = True  # geometry alpha=1, background alpha=0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
