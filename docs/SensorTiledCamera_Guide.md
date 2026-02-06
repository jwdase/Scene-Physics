# SensorTiledCamera Guide

This document explains how to use Newton's `SensorTiledCamera` for GPU-accelerated rendering and depth-to-point-cloud conversion.

## Overview

`SensorTiledCamera` is a Warp-based raytracing camera sensor that renders:
- Color images
- Depth images (ray distance, not Z-buffer)
- Normal images
- Shape index images (for segmentation)
- Albedo images

## Import

```python
from newton._src.sensors import SensorTiledCamera
# or
from newton.sensors import SensorTiledCamera
```

## Basic Setup

```python
import warp as wp
import numpy as np
from newton.sensors import SensorTiledCamera

# Create sensor from a Newton Model
sensor = SensorTiledCamera(
    model=model,           # Newton Model containing shapes
    num_cameras=1,         # Number of cameras per world
    width=640,             # Image width in pixels
    height=480,            # Image height in pixels
    options=SensorTiledCamera.Options(
        default_light=True,
        default_light_shadows=True,
        colors_per_shape=True,
        checkerboard_texture=False,
        backface_culling=True,
    ),
)
```

## Camera Rays

Camera rays are generated in camera space using a pinhole model:

```python
# FOV in radians
fov_radians = np.radians(60.0)
camera_rays = sensor.compute_pinhole_camera_rays(fov_radians)
# Shape: (num_cameras, height, width, 2)
# Index [..., 0] = ray origin (always 0,0,0 in camera space)
# Index [..., 1] = normalized ray direction
```

**Camera space convention:**
- Camera is at origin `(0, 0, 0)`
- Camera looks down **-Z axis**
- **+Y is up**, **+X is right**

## Camera Transforms

The `camera_transforms` array specifies camera poses in world space:

```python
# Shape: (num_cameras, num_worlds)
# Each element is a wp.transformf (position + quaternion XYZW)

camera_transforms = wp.array(
    [[camera_transform]],  # Shape: (1, 1) for 1 camera, 1 world
    dtype=wp.transformf,
    ndim=2
)
```

### Creating a Look-At Transform

```python
from scipy.spatial.transform import Rotation

def look_at_transform(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 1, 0])):
    """Create a wp.transformf that places camera at 'eye' looking at 'target'."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    actual_up = np.cross(right, forward)

    # Build rotation matrix (camera looks down -Z)
    rot_matrix = np.stack([right, actual_up, -forward], axis=1)
    rot = Rotation.from_matrix(rot_matrix)
    quat = rot.as_quat()  # xyzw format

    return wp.transform(
        wp.vec3(eye[0], eye[1], eye[2]),
        wp.quat(quat[0], quat[1], quat[2], quat[3])
    )

# Example usage:
camera_pos = np.array([5.0, 3.0, 5.0])
look_at_point = np.array([0.0, 0.0, 0.0])
cam_transform = look_at_transform(camera_pos, look_at_point)
```

## Rendering

```python
# Create output buffers
color_image = sensor.create_color_image_output()   # (num_worlds, num_cameras, width*height) uint32
depth_image = sensor.create_depth_image_output()   # (num_worlds, num_cameras, width*height) float32
normal_image = sensor.create_normal_image_output() # (num_worlds, num_cameras, width*height) vec3f
shape_index_image = sensor.create_shape_index_image_output()  # uint32

# Render
sensor.render(
    state,                    # Newton State (or None for static scenes)
    camera_transforms,        # (num_cameras, num_worlds)
    camera_rays,              # (num_cameras, height, width, 2)
    color_image=color_image,
    depth_image=depth_image,
    normal_image=normal_image,
    shape_index_image=shape_index_image,
)

# Get numpy arrays
depth_np = depth_image.numpy()  # (num_worlds, num_cameras, width*height)
depth_2d = depth_np[0, 0].reshape(height, width)  # Reshape for world 0, camera 0
```

## Depth Calculation

The depth value is the **Euclidean distance from camera origin to hit point** along the ray:

1. Ray is cast from camera origin in ray direction
2. BVH acceleration finds closest intersection
3. `depth = distance_along_ray` (not Z-buffer depth!)

**Important:**
- Depth = 0.0 for background/no-hit
- Depth is in world units (meters)

## Depth to Point Cloud (GPU)

Since depth is ray distance, converting to 3D points is simple:

```python
@wp.kernel(enable_backward=False)
def depth_to_point_cloud(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
    camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
    width: wp.int32,
    height: wp.int32,
    max_depth: wp.float32,
    # output
    points: wp.array(dtype=wp.vec3f, ndim=3),
):
    world_idx, cam_idx, pixel_idx = wp.tid()

    depth = depth_image[world_idx, cam_idx, pixel_idx]

    # Skip invalid depths
    if depth <= 0.0 or depth >= max_depth:
        points[world_idx, cam_idx, pixel_idx] = wp.vec3f(wp.nan, wp.nan, wp.nan)
        return

    # Convert pixel index to (y, x)
    py = pixel_idx // width
    px = pixel_idx % width

    # Ray direction in camera space
    ray_dir_camera = camera_rays[cam_idx, py, px, 1]

    # Point in camera space
    point_camera = ray_dir_camera * depth

    # Transform to world space
    cam_transform = camera_transforms[cam_idx, world_idx]
    point_world = wp.transform_point(cam_transform, point_camera)

    points[world_idx, cam_idx, pixel_idx] = point_world


def get_point_cloud(sensor, depth_image, camera_rays, camera_transforms, max_depth=1000.0):
    """Convert depth image to point cloud on GPU."""
    num_worlds, num_cameras, num_pixels = depth_image.shape
    width = sensor.render_context.width
    height = sensor.render_context.height

    points = wp.empty((num_worlds, num_cameras, num_pixels), dtype=wp.vec3f)

    wp.launch(
        depth_to_point_cloud,
        dim=(num_worlds, num_cameras, num_pixels),
        inputs=[depth_image, camera_rays, camera_transforms, width, height, max_depth],
        outputs=[points],
    )

    return points


# Usage:
points = get_point_cloud(sensor, depth_image, camera_rays, camera_transforms)
points_np = points.numpy()[0, 0]  # World 0, camera 0
valid_mask = ~np.isnan(points_np[:, 0])
valid_points = points_np[valid_mask]
```

## Visualization Helpers

```python
# Flatten depth to RGBA image for visualization
rgba_buffer = sensor.flatten_depth_image_to_rgba(
    depth_image,
    out_buffer=None,           # Optional pre-allocated buffer
    num_worlds_per_row=None,   # Grid layout
    depth_range=wp.array([near, far], dtype=wp.float32),  # Optional range
)

# Flatten color to RGBA
rgba_buffer = sensor.flatten_color_image_to_rgba(color_image)

# Flatten normals to RGBA
rgba_buffer = sensor.flatten_normal_image_to_rgba(normal_image)
```

## Comparison: PyVista vs SensorTiledCamera

| Aspect | PyVista | SensorTiledCamera |
|--------|---------|-------------------|
| Depth type | Z-buffer (needs linearization) | Ray distance (linear) |
| Computation | CPU | GPU (Warp) |
| Point cloud conversion | Needs intrinsics matrix | Simple: `ray_dir * depth` |
| Multi-world support | No | Yes (batched) |
| Speed | Slower | Much faster |

## File Locations in Newton

- Sensor: `newton/_src/sensors/sensor_tiled_camera.py`
- Render context: `newton/_src/sensors/warp_raytrace/render_context.py`
- Render kernel: `newton/_src/sensors/warp_raytrace/render.py`
- Ray casting: `newton/_src/sensors/warp_raytrace/ray_cast.py`
- Utils (ray generation): `newton/_src/sensors/warp_raytrace/utils.py`
- Example: `newton/examples/sensors/example_sensor_tiled_camera.py`

## Resume Claude Code Session

To exit and resume this conversation:
- Exit: `Ctrl+C` or `/exit`
- Resume: `claude --resume` or `claude --continue`
