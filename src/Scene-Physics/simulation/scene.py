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
TIME = 4
FPS = 240
DT = 1.0 / (FPS)
NUM_FRAMES = TIME * FPS

# Setup Defaults
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

builder.add_ground_plane()

# Loading data from JSON File
simulation_specifictation = (
    load_stimuli_start("objects/stimuli.json", "objects/local_models")
)
sim = next(simulation_specifictation)

# Material
Ball_material = Material(mu=0.8, restitution=.3, contact_ke=2e5, contact_kd=5e3, density=1e3)
Floor = Material(mu=sim['rampDfriction'])

# Add Ball
ball = MeshBody(
    builder=builder, 
    body=sim['ball'], 
    solid=True, 
    scale=sim['ball_scale'] * .1,
    position=sim['ball_postion'],
    mass=2.0,
    material=Ball_material,
    quat=sim['ball_rotation']
    )

# Add Ramp
Ramp = StableMesh(
    builder=builder,
    body=sim['ramp'],
    solid=True,
    scale=1.,
    position=wp.vec3(0.0, 0.0, 0.0),
    mass=0.0,
    material=Floor,
)

# builder.particle_max_velocity = 100.0
builder.balance_inertia = True

# Finalize Model
model = builder.finalize()
model.soft_contact_ke = 1e5

# Start Simulator
state_0 = model.state()
state_1 = model.state()
control = model.control()
solver = SolverXPBD(model, rigid_contact_relaxation=0.9, iterations=100, angular_damping=.1, enable_restitution=True)

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
bodies = [ball, Ramp]
camera = [
    (200, 200, 200),
    (0, 0, 0),
    (0, 1, 0),
]
SceneVisualizer(recorder, bodies, FPS, camera_position=camera).render("recordings/import_ball.mp4")