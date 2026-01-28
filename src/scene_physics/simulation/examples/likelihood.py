# Package Import
import pyvista
import warp as wp
import numpy as np
import jax.numpy as jnp

# b3d likelihood function
from scene_physics.likelihoods import threedp3_likelihood_per_pixel
from scene_physics.intrinsics import Intrinsics, unproject_depth

# Newton Library
import newton
from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD

# Files
from scene_physics.properties.shapes import Sphere, Box, MeshBody, SoftMesh, StableMesh
from scene_physics.properties.material import Material
from scene_physics.visualization.scene import VideoVisualizer
from scene_physics.utils.io import plot_point_maps

# Simple Constraints
TIME = 3
FPS = 40
DT= 1.0/ FPS
NUM_FRAMES = TIME * FPS

# Setup Defaults
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Add Plane
builder.add_ground_plane()

# Material
Ball_material = Material(mu=0.8, restitution=.3, contact_ke=2e5, contact_kd=5e3, density=1e3)
Ramp_material = Material(density=0.0)

# Objects
paths = [
    f'/home/jwdase/projects/Scene-Physics/src/scene_physics/objects/stable_scene/{val}' 
    for val in ['table.obj', 'rectangle.obj']
    ]

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

# builder.particle_max_velocity = 100.0
builder.balance_inertia = True
builder.approximate_meshes(method="convex_hull")

# Finalize Model
model = builder.finalize()
model.soft_contact_ke = 1e5

# Start Simulator
state_0 = model.state()
state_1 = model.state()
control = model.control()
solver = SolverXPBD(model, rigid_contact_relaxation=0.9, iterations=100, angular_damping=.1, enable_restitution=False)

# Recorder
recorder = RecorderModelAndState()

# Simulation
for frame in range(NUM_FRAMES):
    state_0.clear_forces()
    contacts = model.collide(state_0)
    solver.step(state_0, state_1, control, contacts, DT)
    state_0, state_1 = state_1, state_0

    recorder.record(state_0)

    if frame % 100 == 0:

        print(f"Frame {frame}/{NUM_FRAMES}")

# Rendering
bodies = [table, rectangle]

camera = [
    (1, 1.5, 3),
    (0, 1, 0),
    (0, 1, 0),
]

# Create the visualizer
visualizer = VideoVisualizer(recorder, bodies, FPS, camera_position=camera)

# Render visualization
visualizer.render("recordings/initial_still.mp4")

# Set intrinsics
visualizer.set_intrinsics(Intrinsics)

# Create projection
point_cloud = visualizer.point_cloud(unproject_depth)
point_cloud2 = visualizer.point_cloud(unproject_depth, clip=False)


small_noise = np.random.normal(0, .005, point_cloud.shape)
large_noise = np.random.normal(0, .1, point_cloud.shape)

small_noisy_point_cloud = point_cloud + small_noise
large_noisy_point_cloud = point_cloud + large_noise

plot_point_maps(small_noisy_point_cloud, 'recordings/small_noise_cloud.png')
plot_point_maps(large_noisy_point_cloud, 'recordings/large_noise_cloud.png')


likelihood_control = threedp3_likelihood_per_pixel(
    observed_xyz=point_cloud,
    rendered_xyz=point_cloud,
    variance=0.001,
    outlier_prob=0.001,
    outlier_volume=1.0,
    filter_size=3,
)

likelihood_small_noise = threedp3_likelihood_per_pixel(
    observed_xyz=point_cloud,
    rendered_xyz=small_noisy_point_cloud,
    variance=0.001,
    outlier_prob=0.001,
    outlier_volume=1.0,
    filter_size=3,
)

likelihood_large_noise = threedp3_likelihood_per_pixel(
    observed_xyz=point_cloud,
    rendered_xyz=large_noisy_point_cloud,
    variance=0.001,
    outlier_prob=0.001,
    outlier_volume=1.0,
    filter_size=3,
)

def get_likelihood(result):
    likelihood = result["pix_score"]
    clean = np.array([arr[~np.isnan(arr)] for arr in likelihood])
    score = clean.sum()
    return score

baseline_score = get_likelihood(likelihood_control)
small_score = get_likelihood(likelihood_small_noise)
large_score = get_likelihood(likelihood_large_noise)

small_score = (small_score / baseline_score)
large_score = (large_score / baseline_score)

print(f"Baseline: {baseline_score}")

print(f"With small noice: {small_score}")
print(f"With large noise: {large_score}")