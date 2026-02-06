import warp as wp

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
