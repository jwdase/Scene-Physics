import pyvista as pv
import warp as wp
import newton

from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD

from properties.shapes import Sphere, Box, MeshBody, SoftMesh
from properties.material import Material

from visualization.scene import SceneVisualizer

# Simulation Constraints
TIME = 4
FPS = 240
DT = 1.0 / (FPS)
NUM_FRAMES = TIME * FPS

# Setup Defaults
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Material
ClayBall = Material(mu=0.8, restitution=.01, contact_ke=2e5, contact_kd=5e3, density=1e3)

# Load the ball
file1 = "objects/local_models/Ball_1.obj"
box = MeshBody(builder, body=file1, quat=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), wp.pi * 0.25), solid=True, mass=2., position=(2, 2, 0), material=ClayBall)

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
bodies = [box]
SceneVisualizer(recorder, bodies, FPS).render("recordings/import_ball.mp4")