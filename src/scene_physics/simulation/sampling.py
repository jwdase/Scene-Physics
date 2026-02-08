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

# Files
from scene_physics.properties.shapes import Sphere, Box, MeshBody, SoftMesh, StableMesh
from scene_physics.properties.material import Material
from scene_physics.visualization.scene import PyVistaVisuailzer
from scene_physics.utils.io import plot_point_maps

# Visualizations
from scene_physics.utils.plots import plot_location_scores

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
camera = [
    (1, 1.5, 3),
    (0, 1, 0),
    (0, 1, 0),
]

# Create visualizer and set intrinsics
visualizer = PyVistaVisuailzer(bodies, camera_position=camera)
visualizer.set_intrinsics(Intrinsics)

# Generate png - for correct placing and point cloud
visualizer.gen_png('recordings/mc/initial.png')
point_cloud = visualizer.point_cloud(unproject_depth)
visualizer.plot_point_maps(point_cloud, 'recordings/mc/initial_point_cloud.png')

class Likelihood:
    def __init__(self, initial_point_cloud):
        self.correct_pointcloud = initial_point_cloud
        self.control = self.get_control()

    def get_control(self):
        """ Gets the control value"""
        return compute_likelihood_score(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=self.correct_pointcloud,
            variance=0.001,
        )

    def new_proposal_likelihood(self, proposal):
        """ Returns likelihood ration of new proposal to control"""
        new_score = compute_likelihood_score(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=proposal,
            variance=0.001,
        )

        return new_score - self.control

class MHSampler:
    def __init__(self, visualizer, body, likelihood, var):
        self.visualizer = visualizer
        self.body = body
        self.likelihood = likelihood

        self.var = var

    def initial_sample(self):
        """Sets the prior"""
        x, y = np.random.normal(), np.random.normal()
        self.body.update_position(x, y)

    def run_sampling(self, iterations, debug=False):
        # Saved for memory
        positions = []
        score = []

        # Setup the initial prior
        self.initial_sample()
        point_cloud = self.visualizer.point_cloud(unproject_depth)
        prev_likelihood = self.likelihood.new_proposal_likelihood(point_cloud)

        # Store current values
        x_curr, _, z_curr = self.body.get_location()

        # Beg in each iteration
        for i in range(iterations):
            x_prop, z_prop = self.sampler(x_curr, z_curr)
            self.body.update_position(x_prop, z_prop)
            point_cloud = self.visualizer.point_cloud(unproject_depth)

            new_likelihood = self.likelihood.new_proposal_likelihood(
                point_cloud
            )

            log_alpha = new_likelihood - prev_likelihood
            # Accept with prob p
            if np.log(np.random.uniform()) < log_alpha:
                x_curr, z_curr = x_prop, z_prop
                prev_likelihood = new_likelihood
            else:
                self.body.update_position(x_curr, z_curr)

            # Save accepted values (after accept/reject decision)
            positions.append((x_curr, z_curr))
            score.append(prev_likelihood)

            # Print result
            if debug and i % 10 == 0:
                print(f"Current likelihood on iteration {i} is: {prev_likelihood}")
                print(self.body.location())

        return positions, score

    def sampler(self, x0, z0):
        """ Generates new proposal location"""

        dx = np.random.normal(loc=x0, scale=self.var)
        dz = np.random.normal(loc=z0, scale=self.var)

        return (dx, dz)


likelihood_func = Likelihood(point_cloud)


print(f"Likelihood: {likelihood_func.new_proposal_likelihood(point_cloud)}")
print(rectangle.location())


print("=============")
print("Running Samplined")

sampler = MHSampler(visualizer, rectangle, likelihood_func, .2,)
positions, scores = sampler.run_sampling(100, True)

plot_location_scores(positions, scores, "plot.png")


# for i, movement in enumerate(np.linspace(-.5, .5, 10)):
#    rectangle.update_position(movement, 0.)
#    visualizer.gen_png(f'recordings/mc/image0{i}.png')
#    point_cloud = visualizer.point_cloud(unproject_depth)
#    likelihood = likelihood_func.new_proposal_likelihood(point_cloud)

#    print(f"At interval {i}; {rectangle.location()}")

#    print(f"Score of {i} is {likelihood}: location")



