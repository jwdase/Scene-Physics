# System import
import sys
# Package Import
import pyvista
import warp as wp
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# b3d likelihood function
from scene_physics.likelihood.likelihoods import compute_likelihood_score
from scene_physics.likelihood.intrinsics import Intrinsics, unproject_depth

# Newton Library
import newton
from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD
from newton._src.sensors.sensor_tiled_camera import SensorTiledCamera

# Files
from scene_physics.properties.shapes import MeshBody
from scene_physics.properties.material import Material
from scene_physics.visualization.scene import PyVistaVisuailzer
from scene_physics.utils.io import plot_point_maps, save_point_cloud_ply
from scene_physics.kernels.image_process import depth_to_point_cloud
from scene_physics.visualization.camera import look_at_transform



# Setup Defaults
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Add Plane
builder.add_ground_plane()

# Material
Ball_material = Material(mu=0.8, restitution=.3, contact_ke=2e5, contact_kd=5e3, density=1e3)
Ramp_material = Material(density=0.0)

# Objects
paths = [f'objects/stable_scene/{val}' for val in ['table.obj', 'rectangle.obj']]

# Placing objects
table = MeshBody(
    builder=builder,
    body=paths[0],
    solid=True,
    scale=1.0,
    position=wp.vec3(0., 0., 0.),
    mass=0.0,
    material=Ramp_material,
)

rectangle = MeshBody(
    builder=builder,
    body=paths[1],
    solid=True,
    scale=1.0,
    position=wp.vec3(0., 0., 0.),
    mass=0.0,
    material=Ramp_material,
)

# List out bodies, camera for visualizer
bodies = [table, rectangle]
pyvista_camera = [
    (1, 1.5, 3),   # eye position
    (0, 1, 0),     # look at point
    (0, 1, 0),     # up vector
]

model = builder.finalize()

sensor = SensorTiledCamera(
        model=model,
        num_cameras=1,
        width=640,
        height=480
    )

# Need to set the field of view
fov_radians = np.radians(60)
camera_rays = sensor.compute_pinhole_camera_rays(fov_radians)

## Camera Transform

eye = np.array([1.0, 1.5, 3.0])  # Match pyvista_camera
target = np.array([0., 1., 0.])

camera_transform = look_at_transform(eye, target)

## Puts camera in correct type for ray tracing
camera_transforms = wp.array(
        [[camera_transform]],
        dtype=wp.transformf,
        ndim=2,
    )

# Create output buffers
depth_image = sensor.create_depth_image_output()
color_image = sensor.create_color_image_output()

# Get state (for static scene)
state = model.state()

# Render
sensor.render(
    state,
    camera_transforms,
    camera_rays,
    depth_image=depth_image,
    color_image=color_image,
)

# Get numpy arrays
depth_np = depth_image.numpy()  # Shape: (1, 1, 640*480)
depth_2d = depth_np[0, 0].reshape(480, 640)  # (height, width)

print(f"Depth shape: {depth_2d.shape}")
print(f"Depth range: {depth_2d[depth_2d > 0].min():.2f} - {depth_2d.max():.2f} m")

# Convert depth to point cloud on GPU
num_worlds, num_cameras, num_pixels = depth_image.shape
width = sensor.render_context.width
height = sensor.render_context.height
max_depth = 5.0  # Clip points beyond 10m

# Allocate output array for points
points_gpu = wp.empty((num_worlds, num_cameras, num_pixels), dtype=wp.vec3f)

# Launch kernel
wp.launch(
    depth_to_point_cloud,
    dim=(num_worlds, num_cameras, num_pixels),
    inputs=[depth_image, camera_rays, camera_transforms, width, height, max_depth],
    outputs=[points_gpu],
)

# Get numpy array and filter out NaN (invalid) points
points_np = points_gpu.numpy()[0, 0]  # Shape: (num_pixels, 3) for world 0, camera 0
valid_mask = ~np.isnan(points_np[:, 0])
point_cloud = points_np[valid_mask]

print(f"Point cloud shape: {point_cloud.shape}")
print(f"Point cloud bounds:")
print(f"  X: {point_cloud[:, 0].min():.2f} to {point_cloud[:, 0].max():.2f}")
print(f"  Y: {point_cloud[:, 1].min():.2f} to {point_cloud[:, 1].max():.2f}")
print(f"  Z: {point_cloud[:, 2].min():.2f} to {point_cloud[:, 2].max():.2f}")

# Save point cloud to .ply file
save_point_cloud_ply(point_cloud, "recordings/point_cloud.ply")

# PyVista visualization of the scene
visualizer = PyVistaVisuailzer(bodies, pyvista_camera)
visualizer.gen_png("recordings/scene_render.png")
print("Saved PyVista render to recordings/scene_render.png")

# Generate point cloud from PyVista depth
visualizer.set_intrinsics(Intrinsics)
pyvista_point_cloud = visualizer.point_cloud(unproject_depth)
pyvista_point_cloud = np.array(pyvista_point_cloud).reshape(-1, 3)

# Filter valid points
valid_mask_pv = ~np.isnan(pyvista_point_cloud[:, 0]) & (np.abs(pyvista_point_cloud[:, 2]) < 10)
pyvista_point_cloud_valid = pyvista_point_cloud[valid_mask_pv]

print(f"PyVista point cloud shape: {pyvista_point_cloud_valid.shape}")
print(f"PyVista point cloud bounds:")
print(f"  X: {pyvista_point_cloud_valid[:, 0].min():.2f} to {pyvista_point_cloud_valid[:, 0].max():.2f}")
print(f"  Y: {pyvista_point_cloud_valid[:, 1].min():.2f} to {pyvista_point_cloud_valid[:, 1].max():.2f}")
print(f"  Z: {pyvista_point_cloud_valid[:, 2].min():.2f} to {pyvista_point_cloud_valid[:, 2].max():.2f}")

# Save PyVista point cloud
save_point_cloud_ply(pyvista_point_cloud_valid, "recordings/point_cloud_pyvista.ply")
print("Saved PyVista point cloud to recordings/point_cloud_pyvista.ply")
