import warp as wp
import jax
import jax.numpy as jnp

@wp.kernel(enable_backward=False)
def depth_to_point_cloud(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
    camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
    width: wp.int32,
    height: wp.int32,
    max_depth: wp.float32,
    points: wp.array(dtype=wp.vec3f, ndim=3),
):
    world_idx, cam_idx, pixel_idx = wp.tid()

    depth = depth_image[world_idx, cam_idx, pixel_idx]

    # Skip invalid depths (background or too far)
    if depth <= 0.0 or depth >= max_depth:
        points[world_idx, cam_idx, pixel_idx] = wp.vec3f(wp.nan, wp.nan, wp.nan)
        return

    # Convert flat pixel index to (y, x)
    py = pixel_idx // width
    px = pixel_idx % width

    # Get ray direction in camera space
    ray_dir_camera = camera_rays[cam_idx, py, px, 1]

    # Point in camera space = ray_direction * depth
    point_camera = ray_dir_camera * depth

    # Transform to world space
    point_world = wp.transform_point(camera_transforms[cam_idx, world_idx], point_camera)

    points[world_idx, cam_idx, pixel_idx] = point_world


def render_point_cloud(sensor, state, camera_transforms, camera_rays, depth_image, points_gpu, height, width, max_depth):
    """Render scene and return point cloud as (H, W, 3) jnp array."""
    sensor.render(
        state,
        camera_transforms,
        camera_rays,
        depth_image=depth_image,
    )

    wp.launch(
        depth_to_point_cloud,
        dim=depth_image.shape,
        inputs=[depth_image, camera_rays, camera_transforms, width, height, max_depth],
        outputs=[points_gpu],
    )

    # GPU-direct transfer: Warp → JAX via DLPack (no CPU round-trip)
    points_jax = jnp.from_dlpack(points_gpu)
    return points_jax[0, 0].reshape(height, width, 3)


def render_point_clouds_batch(sensor, state, camera_transforms, camera_rays, depth_image, points_gpu, height, width, max_depth, num_worlds):
    """Render all worlds and return batch of point clouds as (num_worlds, H, W, 3) jnp array.

    The SensorTiledCamera already indexes by world — this function preserves
    the world dimension instead of squeezing it.

    Args:
        sensor: SensorTiledCamera configured for num_worlds
        state: Newton state containing all parallel worlds
        camera_transforms: camera transform array
        camera_rays: camera ray array
        depth_image: pre-allocated depth image buffer (num_worlds, num_cameras, num_pixels)
        points_gpu: pre-allocated points buffer matching depth_image shape
        height, width: image dimensions
        max_depth: maximum valid depth
        num_worlds: number of parallel worlds

    Returns:
        jnp.array of shape (num_worlds, H, W, 3)
    """
    sensor.render(
        state,
        camera_transforms,
        camera_rays,
        depth_image=depth_image,
    )

    wp.launch(
        depth_to_point_cloud,
        dim=depth_image.shape,
        inputs=[depth_image, camera_rays, camera_transforms, width, height, max_depth],
        outputs=[points_gpu],
    )

    # GPU-direct transfer: Warp → JAX via DLPack (no CPU round-trip)
    points_jax = jnp.from_dlpack(points_gpu)
    # Shape: (num_worlds, num_cameras, num_pixels, 3) -> (num_worlds, H, W, 3)
    return points_jax[:, 0].reshape(num_worlds, height, width, 3)
