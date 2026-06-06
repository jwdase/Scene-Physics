"""Pure-numpy reimplementation of the Newton tiled-camera pinhole projection.

This is the *uv-env-side* model of exactly what `SensorTiledCamera` does, derived
from `newton/_src/sensors/warp_raytrace/utils.py::compute_pinhole_camera_rays`
and the look-at convention in `visualization/camera.py::look_at_transform`.

It exists so the Blender renderer can be checked against the real sensor without
importing Newton into Blender's Python. Two independent things consume it:

  * the camera-equivalence tests (numpy model vs the real Newton sensor, and
    numpy model vs Blender's `world_to_camera_view`);
  * documentation of the precise pixel convention the rest of the stack assumes.

Conventions (must match the sensor):
  * Camera looks down its local -Z, +Y up, +X right (Newton == Blender camera).
  * Vertical FOV: at the top/bottom image edge the ray's y-component is
    ``tan(fov/2)``; the horizontal extent is scaled by ``aspect = width/height``.
  * Returned pixel coordinates are *pixel-center indices*: the center of pixel
    ``(0,0)`` is ``(0.0, 0.0)``; a point on the optical axis lands at
    ``((width-1)/2, (height-1)/2)``.
"""

from __future__ import annotations

import numpy as np

from scene_physics.configs.camera import CameraIntrinsics


def look_at_rotation(eye, target, up) -> np.ndarray:
    """Camera->world rotation matrix (columns = right, up, -forward).

    Mirrors `visualization/camera.py::look_at_transform` exactly so the numpy
    model and the live sensor share one extrinsic convention.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    # Columns are the camera basis expressed in world space.
    return np.column_stack([right, cam_up, -forward])


def world_to_camera(points, eye, target, up) -> np.ndarray:
    """World points -> camera-space points (camera looks down -Z)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    eye = np.asarray(eye, dtype=np.float64)
    R = look_at_rotation(eye, target, up)
    # p_cam = R^T (p_world - eye)
    return (points - eye) @ R


def project_points(points, intr: CameraIntrinsics) -> np.ndarray:
    """Project world points to pixel-center indices ``(px, py)``.

    Returns an ``(N, 2)`` float array. Points behind the camera (camera-space
    ``z >= 0``) yield NaN rows, matching "would not be seen by the sensor".
    """
    cam = world_to_camera(points, intr.eye, intr.target, intr.up)
    Xc, Yc, Zc = cam[:, 0], cam[:, 1], cam[:, 2]

    h = np.tan(intr.fov_rad / 2.0)
    aspect = intr.width / intr.height

    px = np.full(Xc.shape, np.nan)
    py = np.full(Xc.shape, np.nan)

    front = Zc < 0.0  # in front of the camera (looks down -Z)
    # ray_dir = (u*2h*aspect, -v*2h, -1) ∝ (Xc, Yc, Zc), with λ = -Zc > 0:
    #   u = Xc / (-Zc * 2h * aspect),   v = Yc / (Zc * 2h)
    u = Xc[front] / (-Zc[front] * 2.0 * h * aspect)
    v = Yc[front] / (Zc[front] * 2.0 * h)

    px[front] = (u + 0.5) * intr.width - 0.5
    py[front] = (v + 0.5) * intr.height - 0.5

    return np.column_stack([px, py])


def camera_matrix_world(eye, target, up) -> np.ndarray:
    """4x4 camera->world transform (for reference / direct injection if needed)."""
    M = np.eye(4)
    M[:3, :3] = look_at_rotation(eye, target, up)
    M[:3, 3] = np.asarray(eye, dtype=np.float64)
    return M
