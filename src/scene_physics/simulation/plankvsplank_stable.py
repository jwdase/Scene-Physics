import pyvista as pv
import warp as wp
import newton

from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD

from properties.shapes import Sphere, Box, MeshBody, SoftMesh, StableMesh
from properties.material import Material
from utils.io import load_stimuli_start

from visualization.scene import VideoVisualizer

# ============================================================
# SIMULATION PARAMETERS - ADJUST THESE AS NEEDED
# ============================================================

# Time and Frame Settings
TIME = 5                    # Total simulation time in seconds
FPS = 40                    # Frames per second for rendering
DT = 1.0 / FPS              # Time step (automatically calculated)
NUM_FRAMES = TIME * FPS     # Total number of frames

# Physics Settings
GRAVITY = -9.81             # Gravity acceleration (m/s^2)
ITERATIONS = 100            # Solver iterations per frame (higher = more accurate but slower)
RIGID_CONTACT_RELAXATION = 0.9  # Contact relaxation factor (0.0-1.0)
ANGULAR_DAMPING = 0.1       # Damping for rotational motion

# Material Properties for Dynamic Objects (Rectangle & Peg)
DYNAMIC_FRICTION = 1.5      # Coefficient of friction (mu) - higher = more friction
DYNAMIC_RESTITUTION = 0.3   # Bounciness (0=no bounce, 1=perfect bounce)
DYNAMIC_CONTACT_KE = 2e5    # Contact stiffness
DYNAMIC_CONTACT_KD = 5e3    # Contact damping
DYNAMIC_DENSITY = 1e3       # Material density (kg/m^3)

# Material Properties for Static Object (Table)
STATIC_FRICTION = 1.4       # Friction for table surface
STATIC_RESTITUTION = 0.1    # Bounciness of table

# Object Positions and Scales
TABLE_SCALE = 1.0           # Scale factor for table
RECTANGLE_SCALE = 1.0       # Scale factor for rectangle
PEG_SCALE = 1.0             # Scale factor for peg

# NOTE: Use (0,0,0) if objects already have correct positions in their OBJ files
# Otherwise, adjust these positions to place objects where you want them
TABLE_POSITION = (0., 0., 0.)       # (x, y, z) position
RECTANGLE_POSITION = (0., 0., 0.)   # Use OBJ file's embedded position
PEG_POSITION = (0., 0., 0.)         # Use OBJ file's embedded position

# Camera Settings for Rendering
CAMERA_POSITION = (2, 2, 3)     # Camera location (x, y, z)
CAMERA_FOCAL_POINT = (0, 0.5, 0)  # Point camera looks at
CAMERA_UP_VECTOR = (0, 1, 0)    # Up direction

# Output Settings
OUTPUT_VIDEO = "recordings/plankvsplank_stable_simulation.mp4"  # If MP4 fails, try .avi or .gif

# ============================================================
# SIMULATION SETUP
# ============================================================

# Setup builder with coordinate system and gravity
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=GRAVITY)

# Add ground plane (optional - comment out if table is the only floor)
# builder.add_ground_plane()

# Define Materials
dynamic_material = Material(
    mu=DYNAMIC_FRICTION,
    restitution=DYNAMIC_RESTITUTION,
    contact_ke=DYNAMIC_CONTACT_KE,
    contact_kd=DYNAMIC_CONTACT_KD,
    density=DYNAMIC_DENSITY
)

static_material = Material(
    mu=STATIC_FRICTION,
    restitution=STATIC_RESTITUTION,
    contact_ke=DYNAMIC_CONTACT_KE,
    contact_kd=DYNAMIC_CONTACT_KD,
    density=0.0  # Zero density makes object static
)

# Object paths
object_dir = 'objects/plankvsplank_stable/'
table_path = object_dir + 'table.obj'
rectangle_path = object_dir + 'rectangle.obj'
peg_path = object_dir + 'peg.obj'

# ============================================================
# ADD OBJECTS TO SCENE
# ============================================================

# Add Table (STATIC - mass=0.0)
table = StableMesh(
    builder=builder,
    body=table_path,
    solid=True,
    scale=TABLE_SCALE,
    position=TABLE_POSITION,
    mass=0.0,  # Static object
    material=static_material,
)

# Add Rectangle (DYNAMIC)
rectangle = MeshBody(
    builder=builder,
    body=rectangle_path,
    solid=True,
    scale=RECTANGLE_SCALE,
    position=wp.vec3(*RECTANGLE_POSITION),
    mass=1.0,  # Mass will be calculated from density and volume
    material=dynamic_material,
)

# Add Peg (DYNAMIC)
peg = MeshBody(
    builder=builder,
    body=peg_path,
    solid=True,
    scale=PEG_SCALE,
    position=wp.vec3(*PEG_POSITION),
    mass=1.0,  # Mass will be calculated from density and volume
    material=dynamic_material,
)

# ============================================================
# FINALIZE MODEL AND SETUP SOLVER
# ============================================================

builder.balance_inertia = True
builder.approximate_meshes(method="convex_hull")

# Finalize Model
model = builder.finalize()
model.soft_contact_ke = 1e5

# Initialize states
state_0 = model.state()
state_1 = model.state()
control = model.control()

# Create solver
solver = SolverXPBD(
    model,
    rigid_contact_relaxation=RIGID_CONTACT_RELAXATION,
    iterations=ITERATIONS,
    angular_damping=ANGULAR_DAMPING,
    enable_restitution=True
)

# Setup recorder
recorder = RecorderModelAndState()

# ============================================================
# RUN SIMULATION
# ============================================================

print(f"Starting simulation: {NUM_FRAMES} frames at {FPS} FPS ({TIME}s)")
print(f"Gravity: {GRAVITY} m/s^2")
print(f"Solver iterations: {ITERATIONS}")
print("-" * 60)

for frame in range(NUM_FRAMES):
    state_0.clear_forces()
    contacts = model.collide(state_0)
    solver.step(state_0, state_1, control, contacts, DT)
    state_0, state_1 = state_1, state_0

    recorder.record(state_0)

    if frame % 10 == 0:
        print(f"Frame {frame}/{NUM_FRAMES}")

print("Simulation complete!")

# ============================================================
# RENDER VISUALIZATION
# ============================================================

# List all bodies in scene
bodies = [table, rectangle, peg]

# Camera setup
camera = [
    CAMERA_POSITION,
    CAMERA_FOCAL_POINT,
    CAMERA_UP_VECTOR,
]

# Create visualizer
print(f"\nRendering video to: {OUTPUT_VIDEO}")
visualizer = VideoVisualizer(recorder, bodies, FPS, camera_position=camera)

# Render visualization
visualizer.render(OUTPUT_VIDEO)

# Generate depth maps (optional - method not available in current version)
# depths = visualizer.gen_depth()

print("\n✓ Done! Check the video at:", OUTPUT_VIDEO)
