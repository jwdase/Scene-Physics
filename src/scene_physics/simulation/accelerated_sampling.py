# Package Import
import warp as wp
import numpy as np

# Files
from scene_physics.kernels.image_process import render_point_cloud
from scene_physics.utils.plots import plot_location_scores
from scene_physics.visualization.depth_map import setup_depth_camera
from scene_physics.likelihood.likelihoods import Likelihood
from scene_physics.utils.io import  save_point_cloud_ply

# Run to setup values — imports builder and rectangle into this namespace
from scene_physics.utils.setup import builder, rectangle

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

# Get numpy arrays
depth_np = depth_image.numpy()  # Shape: (1, 1, 640*480)
depth_2d = depth_np[0, 0].reshape(480, 640)  # (height, width)

print(f"Depth shape: {depth_2d.shape}")
print(f"Depth range: {depth_2d[depth_2d > 0].min():.2f} - {depth_2d.max():.2f} m")

# Allocate GPU buffers for point cloud conversion
num_worlds, num_cameras, num_pixels = depth_image.shape
points_gpu = wp.empty((num_worlds, num_cameras, num_pixels), dtype=wp.vec3f)

# Set model reference on bodies so move_position_wp works
class MHSampler:
    def __init__(self, body, model, likelihood, proposal_std, render_fn):
        self.body = body
        self.model = model
        self.likelihood = likelihood
        self.proposal_std = proposal_std
        self.render_fn = render_fn

    def initial_sample(self):
        """Sample initial position from standard normal prior."""
        x = float(np.random.normal(scale=.2))
        z = float(np.random.normal(scale=.2))
        return x, z

    def propose(self, x_curr, z_curr):
        """Propose new position via Gaussian random walk."""
        x_new = float(np.random.normal(loc=x_curr, scale=self.proposal_std))
        z_new = float(np.random.normal(loc=z_curr, scale=self.proposal_std))
        return x_new, z_new

    def run_sampling(self, iterations, debug=False):
        """ Run Metropolis-Hastings sampling"""
        state0, state1 = self.model.state(), self.model.state()

        # Initilize Prior on state
        x_curr, z_curr = self.initial_sample()
        state0.body_q = self.body.move_position_wp(state0, x_curr, z_curr)
        state1.body_q = self.body.move_position_wp(state1, x_curr, z_curr)

        # Generate likelihood
        prev_likelihood = self.likelihood.new_proposal_likelihood(
            self.render_fn(state0)
        )

        # Track histories
        positions, scores = [], []

        for i in range(iterations):
            x_prop, z_prop = self.propose(x_curr, z_curr)
            state1.body_q = self.body.move_position_wp(state0, x_prop, z_prop)
            new_likelihood = self.likelihood.new_proposal_likelihood(
                self.render_fn(state1)
            )

            log_alpha = new_likelihood - prev_likelihood
            if np.log(np.random.uniform()) < log_alpha:
                x_curr, z_curr = x_prop, z_prop
                state0.assign(state1)
                prev_likelihood = new_likelihood


            positions.append((x_curr, z_curr))
            scores.append(float(prev_likelihood))

            if debug and i % 10 == 0:
                print(f"Iteration {i}: likelihood={prev_likelihood:.2f}, pos=({x_curr:.3f}, {z_curr:.3f})")
                save_point_cloud_ply(self.render_fn(state0), f"recordings/{i}_cloud.ply")
                print(log_alpha)
                

        return positions, scores


# --- Main sampling script ---

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
sampler = MHSampler(rectangle, model, likelihood_func, proposal_std=0.02, render_fn=render_fn)

print("=============")
print("Running MCMC Sampling (GPU-accelerated)")
positions, scores = sampler.run_sampling(iterations=1_000, debug=True)

# Plot results
plot_location_scores(positions, scores, save_path="recordings/mc_gpu_samples.png")
print("Saved sample plot to recordings/mc_gpu_samples.png")
