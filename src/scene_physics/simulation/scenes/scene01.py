"""
This code will run a physics inforormed MH Sampling on the placement of the bowl in simulation
"""
import numpy as np

from scene_physics.likelihood.likelihoods_physics import Likelihood_Physics
from scene_physics.sampling.mh import XZ_Physics_MH_Sampler
from scene_physics.utils.setup_scenes import scene01

# Name of simulation
name = "Scene01"

# Importing bodies
builder, bowl, coffee, table = scene01()
bodies = [bowl, coffee, table]
model = builder.finalize()

# Setup Camera
wp_eye = np.array([1., 1.5, 1.])
wp_target = np.array([0., 0., 0.])

pv_eye = np.array([3., 3., 3.])
pv_target = np.array([0., 0., 0.])

width = 640
height = 480
max_depth = 5.0

# Build Likelihood and Sampler
likelihood_f = Likelihood_Physics(model.state(), model, wp_eye, wp_target, pv_eye, pv_target, bodies, name, max_depth=max_depth, height=height, width=width)

sampler = XZ_Physics_MH_Sampler(bowl, model, likelihood_f, .01, watch_every=10)

# Begin Sampling
sampler.run_sampling(100, True)
