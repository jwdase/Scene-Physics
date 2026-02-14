"""
This code will run a physics inforormed MH Sampling on the placement of the bowl in simulation
"""
import numpy as np

from scene_physics.likelihood.likelihoods_physics import Likelihood_Physics
from scene_physics.sampling.mh import XZ_Physics_MH_Sampler
from scene_physics.visualization.camera import setup_depth_camera
from scene_physics.utils.setup_scenes import scene01

# Importing bodies
builder, bowl, coffee, table = scene01()
model = builder.finalize()

# Setup Camera
EYE = np.array([1., 1., 1.])
TARGET = np.array([0., 0., 0.])
WIDTH = 640
HEIGHT = 480
MAX_DEPTH = 5.0
camera = setup_depth_camera(model, EYE, TARGET, WIDTH, HEIGHT)

# Build Likelihood and Sampler
likelihood_f = Likelihood_Physics(model.state(), model, camera)

sampler = XZ_Physics_MH_Sampler(bowl, model, likelihood_f, .01)

# Begin Sampling
sampler.run_sampling(100, True)
