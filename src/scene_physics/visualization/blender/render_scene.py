"""Blender entry-point: render one generated scene -> beauty PNG + ID-pass PNG.

    blender --background --python render_scene.py -- <job.json>

job.json:
    {"usd": "<scene_physics.usdc>",
     "names": ["obj1", ...],            # objects to import/material/segment
     "intrinsics": {...},               # default_camera fields
     "hdri": "<studio.hdr>",
     "out_dir": "<dir>",
     "samples": 128, "device": "GPU", "world_strength": 0.5,
     "ground": true, "studio_lights": true,   # ground plane + key/fill/rim rig
     "view_transform": "AgX"}

Writes <out_dir>/{render.png, seg_raw.png, seg_labels.json}. The uv side
(visualization/segmentation.py) turns seg_raw.png into the integer map + overlay.
"""

from __future__ import annotations

import json
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lighting  # noqa: E402
import _materials  # noqa: E402
import _seg  # noqa: E402
from _camera import setup_camera  # noqa: E402
from _scene import import_scene_usd, reset_scene  # noqa: E402
from mathutils import Vector  # noqa: E402


def _scene_min_z(objs) -> float:
    """Lowest world-space z over the given objects (the floor height)."""
    zmin = float("inf")
    for obj in objs:
        for corner in obj.bound_box:  # 8 local-space bounding-box corners
            zmin = min(zmin, (obj.matrix_world @ Vector(corner)).z)
    return zmin if zmin != float("inf") else 0.0


def _shade_auto(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(50))
    except Exception:
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            pass


def configure_cycles(scene, samples: int, device: str) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    bpy.context.view_layer.cycles.use_denoising = True

    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device == "GPU":
        chosen = None
        for dt in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dt
                prefs.refresh_devices()
                if any(d.type == dt for d in prefs.devices):
                    chosen = dt
                    break
            except Exception:
                pass
        if chosen:
            for d in prefs.devices:
                d.use = d.type == chosen
            scene.cycles.device = "GPU"
        else:
            scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "CPU"


def render_png(scene, path: str) -> None:
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    with open(argv[0]) as f:
        job = json.load(f)

    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    reset_scene()
    mapping = import_scene_usd(job["usd"], job["names"])
    missing = set(job["names"]) - set(mapping)
    if missing:
        print(f"WARNING: USD missing expected objects: {sorted(missing)}")

    for name, obj in mapping.items():
        _shade_auto(obj)
        _materials.assign_material(obj, name)

    # Environment: HDRI for soft fill + reflections, a studio area-light rig for
    # shaping/shadows, and a ground plane so the table is grounded (and casts
    # contact shadows) rather than floating in the HDRI void.
    _materials.setup_world_hdri(
        job["hdri"], job.get("world_strength", 1.0), job.get("hdri_rotation", 0.0)
    )
    if job.get("ground", True):
        floor_z = _scene_min_z(mapping.values())
        _lighting.add_ground_plane(floor_z)
        _lighting.add_back_wall(floor_z)  # near wall standing on the floor (aligned)
    if job.get("studio_lights", True):
        _lighting.setup_studio_lighting()
    setup_camera(job["intrinsics"])

    scene = bpy.context.scene
    configure_cycles(scene, int(job.get("samples", 128)), job.get("device", "GPU"))
    scene.view_settings.view_transform = job.get("view_transform", "AgX")
    render_png(scene, os.path.join(out_dir, "render.png"))

    # --- segmentation ID pass (same camera/poses) ---
    name_to_id = {name: i + 1 for i, name in enumerate(sorted(mapping))}
    pal = _seg.palette(len(name_to_id))
    _seg.apply_id_materials(mapping, name_to_id, pal)
    _lighting.hide_for_id_pass()  # drop ground + rig: only tracked objects get labels
    _seg.configure_id_render(scene)
    render_png(scene, os.path.join(out_dir, "seg_raw.png"))

    with open(os.path.join(out_dir, "seg_labels.json"), "w") as f:
        json.dump(
            {
                "name_to_id": name_to_id,
                "palette": {
                    str(name_to_id[n]): list(pal[name_to_id[n] - 1]) for n in name_to_id
                },
            },
            f,
        )


if __name__ == "__main__":
    main()
