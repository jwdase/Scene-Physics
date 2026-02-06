# System import
import sys

# Package Import
import pyvista
import warp as wp
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# b3d likelihood function
from scene_physics.likelihoods import compute_likelihood_score
from scene_physics.intrinsics import Intrinsics, unproject_depth

# Newton Library
import newton
from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD
from newton._src.sensors.sensor_tiled_camera import SensorTiledCamera

# Files
from scene_physics.properties.shapes import MeshBody
from scene_physics.properties.material import Material
from scene_physics.visualization.scene import PyVistaVisuailzer
from scene_physics.utils.io import plot_point_maps
from scene_physics.kernels.image_process import depth_to_point_cloud
from scene_physics.visualization.camera import look_at_transform

# Setup Defaults, add ground plane
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
builder.add_ground_plane()

# Setup Materials
Wedge_material = Material(
    mu=0.8, restitution=0.3, contact_ke=2e5, contact_kd=5e3, density=1e3
)
Ramp_material = Material(density=0.0)

# Objects
paths = [f"objects/stable_scene/{val}" for val in ["table.obj", "rectangle.obj"]]

# Placing objects
table = MeshBody(
    builder=builder,
    body=paths[0],
    solid=True,
    position=wp.vec3(0.0, 0.0, 0.0),
    mass=0.0,
    material=Wedge_material,
)

rectangle = MeshBody(
    builder=builder,
    body=paths[1],
    solid=True,
    position=wp.vec3(0.0, 0.0, 0.0),
    mass=0.0,
    material=Ramp_material,
)

# Set sensor and finalize model
model = builder.finalize()
sensor = SensorTiledCamera(model=model, num_cameras=1, width=640, height=480)

# Setup camera Defaults
fov_radians = np.radians(60)
camera_rays = sensor.compute_pinhole_camera_rays(fov_radians)

## Camera Transform
eye = np.array([1.0, 1.5, 3.0])  # Match pyvista_camera
target = np.array([0., 1., 0.])
camera_transform = look_at_transform(eye, target)
camera_transforms = wp.array(
        [[camera_transform]],
        dtype=wp.transformf,
        ndim=2,
    )

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



# Image dimensions for reshaping
WIDTH = 640
HEIGHT = 480
MAX_DEPTH = 5.0

# Allocate GPU buffers for point cloud conversion
num_worlds, num_cameras, num_pixels = depth_image.shape
points_gpu = wp.empty((num_worlds, num_cameras, num_pixels), dtype=wp.vec3f)

# Set model reference on bodies so move_position_wp works
table.model = model
rectangle.model = model


def render_point_cloud(sensor, state, camera_transforms, camera_rays, depth_image, points_gpu):
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
        inputs=[depth_image, camera_rays, camera_transforms, WIDTH, HEIGHT, MAX_DEPTH],
        outputs=[points_gpu],
    )

    points_np = points_gpu.numpy()[0, 0]  # (H*W, 3)
    points_3d = points_np.reshape(HEIGHT, WIDTH, 3)
    return jnp.array(points_3d)


def plot_location_scores(locations, scores, save_path=None, cmap='viridis'):
    """Plot sample locations colored by likelihood score."""
    locations = np.array(locations)
    scores = np.array(scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        locations[:, 0],
        locations[:, 1],
        c=scores,
        cmap=cmap,
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Likelihood Score')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Sample Locations with Likelihood Scores')
    ax.set_aspect('equal', adjustable='box')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


class Likelihood:
    def __init__(self, initial_point_cloud):
        self.correct_pointcloud = initial_point_cloud
        self.baseline_score = self._compute_baseline()

    def _compute_baseline(self):
        """Compute the self-comparison score used as normalization baseline."""
        return compute_likelihood_score(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=self.correct_pointcloud,
            variance=0.001,
        )

    def new_proposal_likelihood(self, proposal):
        """Returns log-ratio of proposal likelihood to baseline."""
        new_score = compute_likelihood_score(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=proposal,
            variance=0.001,
        )
        return new_score - self.baseline_score


class MHSampler:
    def __init__(self, body, likelihood, proposal_std, render_fn):
        self.body = body
        self.likelihood = likelihood
        self.proposal_std = proposal_std
        self.render_fn = render_fn

    def initial_sample(self, state):
        """Sample initial position from standard normal prior."""
        x = float(np.random.normal())
        z = float(np.random.normal())
        return x, z

    def propose(self, x_curr, z_curr):
        """Propose new position via Gaussian random walk."""
        x_new = float(np.random.normal(loc=x_curr, scale=self.proposal_std))
        z_new = float(np.random.normal(loc=z_curr, scale=self.proposal_std))
        return x_new, z_new

    def run_sampling(self, state, iterations, debug=False):
        """Run Metropolis-Hastings sampling."""
        # Initialize from prior, and more
        x_curr, z_curr = self.initial_sample(state)
        prop_state = self.body.move_position_wp(state, x_curr, z_curr)

        # Generate Likelihood function
        point_cloud = self.render_fn(prop_state)
        prev_likelihood = self.likelihood.new_proposal_likelihood(point_cloud)
        
        # Saves values through iteration
        positions, scores = [], []

        for i in range(iterations):
            # Propose new position
            x_prop, z_prop = self.propose(x_curr, z_curr)
            prop_state = self.body.move_position_wp(prop_state, x_prop, z_prop)
            point_cloud = self.render_fn(prop_state)
            new_likelihood = self.likelihood.new_proposal_likelihood(point_cloud)

            # MH accept/reject
            log_alpha = new_likelihood - prev_likelihood
            if np.log(np.random.uniform()) < log_alpha:
                x_curr, z_curr = x_prop, z_prop
                prev_likelihood = new_likelihood
            else:
                self.body.move_position_wp(state, x_curr, z_curr)

            positions.append((x_curr, z_curr))
            scores.append(float(prev_likelihood))

            if debug and i % 10 == 0:
                print(f"Iteration {i}: likelihood={prev_likelihood:.2f}, pos=({x_curr:.3f}, {z_curr:.3f})")

        return positions, scores


# --- Main sampling script ---

# Generate observed point cloud at ground truth position
observed_point_cloud = render_point_cloud(
    sensor, state, camera_transforms, camera_rays, depth_image, points_gpu
)
print(f"Observed point cloud shape: {observed_point_cloud.shape}")

# Create likelihood from observed point cloud
likelihood_func = Likelihood(observed_point_cloud)
print(f"Baseline likelihood: {likelihood_func.baseline_score:.2f}")

# Create render helper bound to scene resources
def render_fn(state):
    return render_point_cloud(sensor, state, camera_transforms, camera_rays, depth_image, points_gpu)

# Create sampler and run
sampler = MHSampler(rectangle, likelihood_func, proposal_std=0.2, render_fn=render_fn)

print("=============")
print("Running MCMC Sampling (GPU-accelerated)")
positions, scores = sampler.run_sampling(state, iterations=100, debug=True)

# Plot results
plot_location_scores(positions, scores, save_path="recordings/mc_gpu_samples.png")
print("Saved sample plot to recordings/mc_gpu_samples.png")
