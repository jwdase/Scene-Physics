"""Blender-side camera setup that reproduces Newton's `default_camera`.

Runs inside Blender's bundled Python (bpy + mathutils + stdlib only) -- it must
NOT import the scene_physics package (that pulls in warp/newton, absent here).

The look-at basis is reconstructed independently (not injected from the numpy
side) so the camera-equivalence test actually exercises the convention rather
than tautologically copying a matrix. Intrinsics arrive as a plain dict:

    {"eye": [x,y,z], "target": [x,y,z], "up": [x,y,z],
     "fov_degree": 60.0, "width": 640, "height": 480}

Convention match with Newton (`visualization/camera.py`, warp_raytrace/utils.py):
  * Z-up world (Blender's native up); camera looks down local -Z, +Y up.
  * Vertical FOV -> sensor_fit='VERTICAL', camera.data.angle = radians(fov).
  * Square pixels, resolution = (width, height).
"""

from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector


def look_at_matrix(eye, target, up) -> Matrix:
    """Camera->world transform; columns of the 3x3 block are right, up, -forward.

    Mirrors scene_physics.visualization.newton_projection.look_at_rotation and
    visualization/camera.py::look_at_transform.
    """
    eye = Vector(eye)
    forward = (Vector(target) - eye).normalized()
    right = forward.cross(Vector(up)).normalized()
    cam_up = right.cross(forward)
    nf = -forward
    return Matrix(
        (
            (right.x, cam_up.x, nf.x, eye.x),
            (right.y, cam_up.y, nf.y, eye.y),
            (right.z, cam_up.z, nf.z, eye.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def setup_camera(intr: dict) -> bpy.types.Object:
    """Create (or reuse) a camera object matching `intr` and make it active.

    Returns the camera object.
    """
    scene = bpy.context.scene

    # Render frame: exact pixel grid, square pixels.
    scene.render.resolution_x = int(intr["width"])
    scene.render.resolution_y = int(intr["height"])
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    cam_data = bpy.data.cameras.new("MatchCam")
    cam_data.type = "PERSP"
    cam_data.lens_unit = "FOV"
    # Vertical FOV: with sensor_fit='VERTICAL', `.angle` is the vertical angle.
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle = math.radians(float(intr["fov_degree"]))

    cam_obj = bpy.data.objects.new("MatchCam", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.matrix_world = look_at_matrix(intr["eye"], intr["target"], intr["up"])

    scene.camera = cam_obj
    return cam_obj
