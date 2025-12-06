# Add correct path for me
from utils.io import setup_path
setup_path('jonathan')

# Package Import
import pyvista
import warp as wp
import numpy as np
import jax.numpy as jnp

# GenJax
from genjax import ChoiceMapBuilder as C
from genjax import gen, normal, pretty
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
from visualization.scene import SceneVisualizer
from utils.io import plot_point_maps

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

