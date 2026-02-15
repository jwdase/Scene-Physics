# Package Import
import warp as wp
import numpy as np

# Files
from scene_physics.kernels.image_process import render_point_cloud
from scene_physics.utils.plots import plot_location_scores
from scene_physics.visualization.camera import setup_depth_camera
from scene_physics.likelihood.likelihoods import Likelihood
from scene_physics.utils.io import  save_point_cloud_ply
from scene_physics.sampling.mh import XZ_MH_Sampler

# Run to setup values — imports builder and rectangle into this namespace
from scene_physics.utils.setup import builder, rectangle
from scene_physics.utils.io import render_bio

# Set sensor and finalize model
model = builder.finalize()
eye = np.array([1., 1., 1.])
target = np.array([0., 0., 0.])
WIDTH = 640
HEIGHT = 480
MAX_DEPTH = 5.0

# Get Camera
camera = setup_depth_camera(model, eye, target, WIDTH, HEIGHT)
sensor = camera["sensor"]
camera_transforms = camera["camera_transforms"]
camera_rays = camera["camera_rays"]

# Create output buffers
depth_image = sensor.create_depth_image_output()
color_image = sensor.create_color_image_output()
state = model.state()
sensor.render(
    state,
    camera_transforms,
    camera_rays,
    depth_image=depth_image,
    color_image=color_image,
)

depth_image = render_bio(depth_image, HEIGHT, WIDTH)

num_worlds, num_cameras, num_pixels = depth_image.shape
points_gpu = wp.empty((num_worlds, num_cameras, num_pixels), dtype=wp.vec3f)


# Generate observed point cloud at ground truth position
observed_point_cloud = render_point_cloud(
    sensor, state, camera_transforms, camera_rays, depth_image, points_gpu, HEIGHT, WIDTH, MAX_DEPTH
)
print(f"Observed point cloud shape: {observed_point_cloud.shape}")

# Create likelihood from observed point cloud
likelihood_func = Likelihood(observed_point_cloud)
print(f"Baseline likelihood: {likelihood_func.baseline_score:.2f}")

# Create render helper bound to scene resources
def render_fn(state):
    return render_point_cloud(sensor, state, camera_transforms, camera_rays, depth_image, points_gpu, HEIGHT, WIDTH, MAX_DEPTH)

# Create sampler and run
sampler = XZ_MH_Sampler(rectangle, model, likelihood_func, proposal_std=0.02, render_fn=render_fn)

print("=============")
print("Running MCMC Sampling (GPU-accelerated)")
positions, scores = sampler.run_sampling(iterations=1_000, debug=True)

# Plot results
plot_location_scores(positions, scores, save_path="recordings/mc_gpu_samples.png")
print("Saved sample plot to recordings/mc_gpu_samples.png")
