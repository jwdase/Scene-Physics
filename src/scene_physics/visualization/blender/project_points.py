"""Blender entry-point: project world points to pixels via the Blender camera.

Used only by the camera-equivalence test. Reads a JSON job from a path given
after `--`, sets up the matched camera, and writes pixel coordinates (in the
*same* pixel-center convention as newton_projection.project_points) back out.

    blender --background --python project_points.py -- <job.json> <out.json>

job.json:  {"intrinsics": {...}, "points": [[x,y,z], ...]}
out.json:  {"pixels": [[px,py] or null, ...]}   # null = behind camera / off
"""

from __future__ import annotations

import json
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _camera import setup_camera  # noqa: E402


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    job_path, out_path = argv[0], argv[1]
    with open(job_path) as f:
        job = json.load(f)

    intr = job["intrinsics"]
    cam = setup_camera(intr)
    scene = bpy.context.scene
    W, H = int(intr["width"]), int(intr["height"])

    pixels = []
    for p in job["points"]:
        co = world_to_camera_view(scene, cam, Vector(p))
        # world_to_camera_view: x,y in [0,1] (x:left->right, y:bottom->top),
        # z = signed distance along view dir (>0 in front). Convert to the
        # pixel-center index convention used by the numpy model.
        if co.z <= 0.0:
            pixels.append(None)
            continue
        px = co.x * W - 0.5
        py = (1.0 - co.y) * H - 0.5
        pixels.append([px, py])

    with open(out_path, "w") as f:
        json.dump({"pixels": pixels}, f)


if __name__ == "__main__":
    main()
