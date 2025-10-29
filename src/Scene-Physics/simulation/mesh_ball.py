import copy
import time

import warp as wp

from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD, SolverVBD
import newton

import numpy as np
import pyvista as pv

from properties.material import Material
from properties.shapes import Sphere, Box, MeshBody, SoftMesh
from physics.kernels import apply_random_force, apply_random_force_rot
from visualization.scene import SceneVisualizer

# Simulation constants
TIME = 4
FPS = 240
DT = 1.0 / (FPS)
NUM_FRAMES = TIME * FPS


# Ball properties
RADIUS = 0.2
BIG_RADIUS = .8
HEIGHT = 6.0
x_balls = 5
y_balls = 5

# Dtypes
vec6f = wp.types.vector(length=6, dtype=float)

# Build the physics model
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Ground Plane
WoodFloor = Material(mu=0.05, restitution=0.1, contact_ke=2e5, contact_kd=5e3)
builder.add_ground_plane(cfg=WoodFloor.to_cfg(builder))

# Add the balls
RubberBall = Material(mu=0.8, restitution=1, contact_ke=2e5, contact_kd=5e3, density=1e3)  # ~water density
MediumBall = Material(mu=0.8, restitution=0.3, contact_ke=2e5, contact_kd=5e3)
ClayBall = Material(mu=0.5, restitution=0.2, contact_ke=2e5, contact_kd=5e3)

bouncy = Sphere(builder, radius=0.5, mass=.2, position=(-1, 2, 0), material=RubberBall)
medium = Sphere(builder, radius=0.5, mass=.2, position=(0, 2, 0), material=MediumBall)
dull = Sphere(builder, radius=0.5, mass=.2, position=(1, 2, 0), material=ClayBall)


# Mesh Box
rounded_cube_surface = pv.Superquadric(  
    theta_roundness=0.3,  
    phi_roundness=0.3,  
)

box = MeshBody(builder, body=rounded_cube_surface, quat=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), wp.pi * 0.25), solid=True, mass=2., position=(2, 2, 0), material=MediumBall)


# builder.particle_max_velocity = 100.0
builder.balance_inertia = True
print(builder.body_com)


# --- Finalize model ---
model = builder.finalize()

model.soft_contact_ke = 1e5

state_0 = model.state()
state_1 = model.state()
control = model.control()
solver = SolverXPBD(model, rigid_contact_relaxation=0.9, iterations=100, angular_damping=.1, enable_restitution=True)

# --- Builder Recorder ---
recorder = RecorderModelAndState()

for frame in range(NUM_FRAMES):
    state_0.clear_forces()

    # wp.launch(apply_random_force,
    #     dim=model.body_count,
    #     inputs=[state_0.body_f, 50.0, int(time.time()) + frame]
    # )

    contacts = model.collide(state_0)
    solver.step(state_0, state_1, control, contacts, DT)
    state_0, state_1 = state_1, state_0

    recorder.record(state_0)

    if frame % 100 == 0:
        print(f"Frame {frame}/{NUM_FRAMES}")

bodies = [
    bouncy,
    medium,
    dull,
    box,
]


SceneVisualizer(recorder, bodies, FPS).render("bounce.mp4")