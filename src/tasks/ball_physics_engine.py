import warp as wp
import newton
from newton.solvers import SolverSemiImplicit  # or SolverXPBD, SolverMuJoCo

wp.init()

# Build the scene (Y-up to match many Warp examples; use Axis.Z if you prefer)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Infinite ground plane at y=0
builder.add_ground_plane()

# Dynamic ball body at y=3
ball = builder.add_body(
    xform=wp.transform(wp.vec3(0.0, 3.0, 0.0), wp.quat_identity()),
    mass=1.0
)

# Attach a spherical shape (radius 0.5) to the ball
builder.add_shape_sphere(body=ball, radius=0.5)

# Finalize model and set up solver/state
model = builder.finalize()
state_0, state_1 = model.state(), model.state()
control = model.control()     # optional, but pass along
solver = SolverSemiImplicit(model)

dt = 1.0 / 60.0

for i in range(240):
    state_0.clear_forces()
    contacts = model.collide(state_0)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0

    # body_q is an array of warp 'transform' (position+rotation); '.p' is the position
    host_body_q = state_0.body_q.numpy()
    pos = host_body_q[ball, :3]
    
    print(f"Step {i}  Ball pos: {pos}")

