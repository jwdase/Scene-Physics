# Add the correc paths for me
from utils.io import setup_path
setup_path('jonathan')

import pyvista as pv
import warp as wp
import newton

from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD

from properties.shapes import Sphere, Box, MeshBody, SoftMesh, StableMesh
from properties.material import Material
from utils.io import load_stimuli_start

from visualization.scene import SceneVisualizer

# Simulation Constraints
TIME = 3
FPS = 40
DT = 1.0 / (FPS)
NUM_FRAMES = TIME * FPS

# Setup Defaults
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Add Plane
builder.add_ground_plane()

# Material
Ball_material = Material(mu=0.8, restitution=.3, contact_ke=2e5, contact_kd=5e3, density=1e3)
Ramp_material = Material(density=0.0)

paths = [f'objects/scene01/{val}' for val in ['BOWL.obj', 'COFFEE.obj', 'TABLE.obj']]

# Add Object 1
obj1 = MeshBody(
    builder=builder, 
    body=paths[0], 
    solid=True, 
    scale=1.0,
    position=wp.vec3(0., 0., 0.),
    mass=0.0,
    material=Ramp_material,
    )

# Add Object 2
obj2 = MeshBody(
    builder=builder, 
    body=paths[1], 
    solid=True, 
    scale=1.0,
    position=wp.vec3(0., 0., 0.),
    mass=0.0,
    material=Ramp_material,
    )

# Add Object 3
obj3 = MeshBody(
    builder=builder, 
    body=paths[2], 
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
bodies = [obj1, obj2, obj3]
camera = [
    (0, 1, 3),
    (0, 1, 0),
    (0, 1, 0),
]

# Create the visualizer
visualizer = SceneVisualizer(recorder, bodies, FPS, camera_position=camera)

# Render visualization
visualizer.render("recordings/initial_still.mp4")

# Create projection
depths = visualizer.gen_depth()