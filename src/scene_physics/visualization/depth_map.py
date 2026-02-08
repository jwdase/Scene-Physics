import numpy as np
import warp as wp

from newton._src.sensors.sensor_tiled_camera import SensorTiledCamera
from scene_physics.visualization.camera import look_at_transform


def setup_depth_camera(model, eye, target, width, height, fov_radians=60):
    """Sets up the sensor tiled camera for use"""
    sensor = SensorTiledCamera(model=model, num_cameras=1, width=width, height=height)

    # Setup camera Defaults
    fov_radians = np.radians(60)
    camera_rays = sensor.compute_pinhole_camera_rays(fov_radians)

    camera_transform = look_at_transform(eye, target)
    camera_transforms = wp.array(
        [[camera_transform]],
        dtype=wp.transformf,
        ndim=2,
    )

    return {
        "sensor" : sensor,
        "camera_rays" : camera_rays,
        "camera_transform" : camera_transform,
        "camera_transforms" : camera_transforms,
    }

