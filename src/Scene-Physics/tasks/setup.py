# Add correct path for me
from utils.io import setup_path
setup_path('jonathan')

# Package Import
import pyvista
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# GenJax
from genjax import ChoiceMapBuilder as C
from genjax import gen, normal, pretty, uniform
from genjax import Diff, NoChange, UnknownChange

# Newton Library
import newton
from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD

# Bayes 3D Import
from b3d.chisight.dense.likelihoods.image_likelihoods import threedp3_likelihood_per_pixel_old
from b3d.camera import unproject_depth, Intrinsics
import b3d

# Files
from properties.shapes import Sphere, Box, MeshBody, SoftMesh, StableMesh
from properties.material import Material
from visualization.scene import PyVistaVisuailzer
from utils.io import plot_point_maps

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
visualizer.gen_png('recordings/initial.png')
point_cloud = visualizer.point_cloud(unproject_depth)

class Likelihood:
    def __init__(self, initial_point_cloud):
        self.correct_pointcloud = initial_point_cloud
        self.control = self.get_control()

    def get_control(self):
        """ Gets the control value"""
        likelihood_control = threedp3_likelihood_per_pixel_old(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=self.correct_pointcloud,
            variance=0.001,
            outlier_prob=0.001,
            outlier_volume=1.0,
            filter_size=3,
        )

        return self.get_likelihood(likelihood_control)

    def get_likelihood(self, result):
        """ Code to cleanup likelihood from B3D result"""
        likelihood = result["pix_score"]
        clean = np.array([arr[~np.isnan(arr)] for arr in likelihood])
        score = clean.sum()
        return score

    def new_proposal_likelihood(self, proposal):
        """ Returns likelihood of new control"""
        new_score = threedp3_likelihood_per_pixel_old(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=proposal,
            variance=0.001,
            outlier_prob=0.001,
            outlier_volume=1.0,
            filter_size=3,
        )

        # [TODO] DOUBLE CHECK IF CORRECT
        return (self.get_likelihood(score)) / self.control

likelihood_func = Likelihood(point_cloud)

@gen
def model(x):
    # Sample latent variables
    x = uniform(-3., 3.) @ 'x'
    z = uniform(-3., 3.) @ 'z'

    # Update object position, and new point cloud
    rectangle.update_position(x, z)
    point_cloud = visualizer.point_cloud(unproject_depth)

    # Calc Likelihood of Placement
    result = likelihood_func.new_proposal_likelihood(point_cloud)

    # Update log_likelihood
    log_likely = normal(0.0, 1e-6, log_prob=result) @ 'll_score'

    return result

# Unsure what to set this to
obs_physics = C["ll_score"].set(0.0)

@gen
def prop_physics(tr, *_):
    # Get current sampled values from the previous trace
    orig_x = tr.get_choices()["x"]
    orig_z = tr.get_choices()["z"]
    
    # Propose new values using a Gaussian drift (like the notebook's MH)
    # The standard deviation (e.g., 0.1) controls the MCMC step size.
    x = normal(orig_x, 0.1) @ "x"
    z = normal(orig_z, 0.1) @ "z"
    
    return x, z
