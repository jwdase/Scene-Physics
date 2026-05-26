"""
Shared physics-solver defaults.

Single source of truth for the XPBD solver + contact-material settings used
across the project: the dataset generator (data_gen/scene_gen.py), the
re-simulation/viewer (simulation/simulation.py), and the sampling-time
likelihood (likelihood/likelihoods.py). Keeping them here means a re-sim or a
likelihood forward-sim integrates at the exact fidelity the dataset was settled
at.
"""

import newton


GRAVITY = -9.81
DENSITY = 1000.0  # per-body density used when adding dynamic shapes
UP_AXIS = newton.Axis.Z

# Contact material
MU = 0.8  # sliding friction
RESTITUTION = 0.0
ROLL_FRICTION = 0.1  # damps endless rolling of cylindrical objects so they settle
TORSION_FRICTION = 0.1

# XPBD integration
SOLVER_ITERS = 32
SUBSTEPS = 8
DT = 1.0 / 60.0
SUB_DT = DT / SUBSTEPS