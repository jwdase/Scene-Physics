import warp as wp
import newton

from newton.solvers import SolverXPBD

import numpy as np
import time
import pyvista as pv

from properties.material import Material
from properties.shapes import Sphere, Box, MeshBody, SoftMesh
from physics.kernels import apply_random_force, apply_random_force_rot

# Simulation constants
TIME = 10
FPS = 60
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
WoodFloor = Material(mu=0.5, restitution=0.2, contact_ke=2e5, contact_kd=5e3)
builder.add_ground_plane(cfg=WoodFloor.to_cfg(builder))

# Add the balls
RubberBall = Material(mu=0.8, restitution=1, contact_ke=2e5, contact_kd=5e3) 
MediumBall = Material(mu=0.8, restitution=0.5, contact_ke=2e5, contact_kd=5e3)
ClayBall = Material(mu=0.5, restitution=0.2, contact_ke=2e5, contact_kd=5e3)

bouncy = Sphere(builder, radius=0.5, mass=.2, position=(-1, 5, 0), material=RubberBall)
medium = Sphere(builder, radius=0.5, mass=.2, position=(0, 5, 0), material=MediumBall)
dull = Sphere(builder, radius=0.5, mass=.2, position=(1, 5, 0), material=ClayBall)
box = Box(builder, half_extends=(0.5, 0.5, 0.5), mass=2.0, position=(3, 5, 0), material=RubberBall)

# --- Finalize model ---
model = builder.finalize()
state_0 = model.state()
state_1 = model.state()
control = model.control()
solver = SolverXPBD(model, iterations=100, enable_restitution=True)

positions = []

for frame in range(NUM_FRAMES):
    state_0.clear_forces()

    wp.launch(apply_random_force_rot,
        dim=model.body_count,
        inputs=[state_0.body_f, 50.0, int(time.time()) + frame]
    )

    contacts = model.collide(state_0)
    solver.step(state_0, state_1, control, contacts, DT)
    state_0, state_1 = state_1, state_0
    pos = state_0.body_q.numpy()[:, :3]

    positions.append(pos.copy())

    if frame % 100 == 0:
        print(f"Frame {frame}/{NUM_FRAMES}")


# RENDERING
plotter = pv.Plotter(off_screen=True)
plotter.set_background("white")
plotter.add_axes()

# Ground plane
plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
plotter.add_mesh(plane, color="lightgray", opacity=0.8)

# Plotter position
plotter.camera_position = [
    (20, 20, 20),
    (0, 0, 0),
    (0, 1, 0),
]

mesh = [bouncy.to_pyvista(), medium.to_pyvista(), dull.to_pyvista(), box.to_pyvista()]
color = ['red', 'black', 'green', 'blue']

actors = []
for i in range(len(mesh)):
    actor = plotter.add_mesh(mesh[i].copy(), color=color[i], smooth_shading=True)
    actors.append(actor)

# Write Movie
output_filename = "bounce.mp4"
plotter.open_movie(output_filename, framerate=FPS, quality=9)

for frame_idx, pos in enumerate(positions):
    for i, actor in enumerate(actors):
        temp_mesh = mesh[i].copy()
        temp_mesh.translate(pos[i], inplace=True)

        actor.mapper.SetInputData(temp_mesh)

    plotter.write_frame()

    if frame_idx % 100 == 0:
        print(f"Rendered frame {frame_idx}/{NUM_FRAMES}")

plotter.close()

print("Done!")