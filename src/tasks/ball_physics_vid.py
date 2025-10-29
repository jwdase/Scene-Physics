import warp as wp
import newton
from newton.solvers import SolverXPBD
import pyvista as pv
import numpy as np
import time

import copy 

# -------------------------
# Initialize Warp and model
# -------------------------
wp.init()

# Simulation constants
TIME = 10
FPS = 60
SUBSTEP = 1
DT = 1.0 / (FPS)
NUM_FRAMES = TIME * FPS


# Ball properties
RADIUS = 0.2
BIG_RADIUS = .8
HEIGHT = 6.0
x_balls = 5
y_balls = 5
NUM_BALLS = x_balls * y_balls

# World Properties
DRAG_COEFFICIENT = .2

# Dtypes
vec6f = wp.types.vector(length=6, dtype=float)

# Y-up world with gravity
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Add infinite ground plane

cfg1 = copy.deepcopy(builder.default_shape_cfg)
cfg1.mu = .2
cfg1.restitution = 0

cfg2 = copy.deepcopy(builder.default_shape_cfg)
cfg2.mu = .2
cfg2.restitution = 1

cfg_ground = copy.deepcopy(builder.default_shape_cfg)
cfg_ground.mu = 1.0
cfg_ground.restitution = 0.0   # non-bouncy ground
cfg_ground.contact_ke = 1e6
cfg_ground.contact_kd = 1e3

builder.add_ground_plane(cfg=cfg_ground)


i = 0
# Generating 25 balls
for x_location in np.linspace(-3, 3, x_balls):
    for z_location in np.linspace(-3, 3, y_balls):

        ball = builder.add_body(
            xform=wp.transform(wp.vec3(x_location, HEIGHT, z_location), wp.quat_identity()),
            mass=2.0,
        )

        if i <= 10:
            builder.add_shape_sphere(
                body=ball,
                radius=RADIUS,
                cfg=cfg1
            )
            
        else:
            builder.add_shape_sphere(
                body=ball,
                radius=RADIUS,
                cfg=cfg2
            )

        i += 1

box = builder.add_body(
    xform=wp.transform(wp.vec3(0, HEIGHT + 4, 0), wp.quat_identity()),
    mass=2.0,
)

builder.add_shape_box(
    body=box,
    hz=2,
    hy=2,
    hx=2,
    cfg = cfg1
)


# Adding Wind
@wp.kernel
def apply_air_drag(
    body_v: wp.array(dtype=vec6f), 
    body_f: wp.array(dtype=vec6f),
    drag_coefficient: float,
):
    i = wp.tid()

    # extract linear velocity (first 3 components of vec6)
    v = wp.vec3(body_v[i][0], body_v[i][1], body_v[i][2])
    speed = wp.length(v)

    # apply quadratic drag: F = -c_d * v * |v|
    if speed > 1e-6:
        drag_force = -drag_coefficient * v * speed

        # add to force accumulator
        f = body_f[i]
        body_f[i] = vec6f(
            f[0] + drag_force[0],
            f[1] + drag_force[1],
            f[2] + drag_force[2],
            f[3],
            f[4],
            f[5],
        )

@wp.kernel
def apply_wind(
    body_v: wp.array(dtype=vec6f), 
    body_f: wp.array(dtype=vec6f),
    wind: float
):
    i = wp.tid()

    # Randomly put wind on the balls
    f = body_f[i]
    body_f[i] = vec6f(
        f[0] + wind,
        f[1],
        f[2],
        f[3],
        f[4],
        f[5]
    )


# Random Forces
@wp.kernel
def apply_random_force(
    body_f: wp.array(dtype=vec6f),
    strength: float,
    seed: int
):
    i = wp.tid()
    # reproducible random values per body
    rng = wp.rand_init(seed, i)
    fx = (wp.randf(rng) - 0.5) * 2.0 * strength
    fy = (wp.randf(rng) - 0.5) * 2.0 * strength
    fz = (wp.randf(rng) - 0.5) * 2.0 * strength

    f = body_f[i]
    body_f[i] = vec6f(
        f[0] + fx,
        f[1] + fy,
        f[2] + fz,
        f[3],
        f[4],
        f[5]
    )


# -------------------------
# Finalize model and setup
# -------------------------
model = builder.finalize()
state_0, state_1 = model.state(), model.state()
control = model.control()
solver = SolverXPBD(model, iterations=40)

# -------------------------
# Run simulation
# -------------------------
positions = []

for frame in range(NUM_FRAMES):
    for _ in range(SUBSTEP):
        state_0.clear_forces()

        wp.launch(
            kernel=apply_air_drag,
            dim=model.body_count,
            inputs=[state_0.body_qd, state_0.body_f, DRAG_COEFFICIENT],
        )

        wind = 2

        wp.launch(
            kernel=apply_wind,
            dim=model.body_count,
            inputs=[state_0.body_qd, state_0.body_f, 2.0]
        )

        wp.launch(
            kernel=apply_random_force,
            dim=model.body_count,
            inputs=[state_0.body_f, 4.0, frame]  # strength=2N, seed=frame for variation
        )

        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0

    # Get all body positions
    q_full = state_0.body_q.numpy()          # shape: (num_bodies, 7)
    pos = q_full[:, :3]                      # xyz
    quat = q_full[:, 3:7]                    # (x, y, z, w)
    positions.append((pos.copy(), quat.copy()))  # shape: (NUM_BALLS, 3)

    if frame % 100 == 0:
        print(f"Frame {frame}/{NUM_FRAMES}")


# RENDERING
plotter = pv.Plotter(off_screen=True)
plotter.set_background("white")
plotter.add_axes()

# Ground plane
plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
plotter.add_mesh(plane, color="lightgray", opacity=0.8)

# Create ball actors
sphere_mesh = pv.Sphere(radius=RADIUS)
actors = []
for i in range(NUM_BALLS):
    actor = plotter.add_mesh(sphere_mesh.copy(), color="orange", smooth_shading=True)
    actors.append(actor)

# Create big Ball
big_mesh = pv.Cube(
    x_length=2,
    y_length=2,
    z_length=2
)
actors.append(plotter.add_mesh(big_mesh.copy(), color="green", smooth_shading=True))

# Camera
plotter.camera_position = [
    (20, 20, 20),
    (0, 0, 0),
    (0, 1, 0),
]
def quat_to_mat(q):
    # q = [x, y, z, w]  (Warp/Newton uses xyzw)
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),       2*(xz+wy)],
        [    2*(xy+wz),  1 - 2*(xx+zz),      2*(yz-wx)],
        [    2*(xz-wy),      2*(yz+wx),   1 - 2*(xx+yy)]
    ], dtype=float)
    return R


# Write movie
output_filename = "multi_balls.mp4"
plotter.open_movie(output_filename, framerate=FPS, quality=9)

for frame_idx, (pos, quat) in enumerate(positions):
    for i, actor in enumerate(actors):
        if i < NUM_BALLS:
            # spheres: translate only
            mesh = sphere_mesh.copy()
            mesh.translate(pos[i], inplace=True)
        else:
            # the box is the last body you added → same index in state arrays
            mesh = big_mesh.copy()
            R = quat_to_mat(quat[i])          # 3x3
            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = pos[i]
            mesh.transform(T, inplace=True)   # rotate then translate

        actor.mapper.SetInputData(mesh)

    plotter.write_frame()

    if frame_idx % 100 == 0:
        print(f"Rendered frame {frame_idx}/{NUM_FRAMES}")

plotter.close()

print(f"✅ Simulation complete. Saved video as {output_filename}")
