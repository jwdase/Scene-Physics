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
DT = 1.0 / FPS
NUM_FRAMES = TIME * FPS

# Ball properties
RADIUS = 0.2
HEIGHT = 6.0
x_balls, y_balls = 5, 5
NUM_BALLS = x_balls * y_balls

# Air drag
DRAG_COEFFICIENT = 0.2
vec6f = wp.types.vector(length=6, dtype=float)

# -------------------------
# Build world and materials
# -------------------------
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

cfg_soft = copy.deepcopy(builder.default_shape_cfg)
cfg_soft.mu = 0.2
cfg_soft.restitution = 1.0

cfg_bouncy = copy.deepcopy(builder.default_shape_cfg)
cfg_bouncy.mu = 0.2
cfg_bouncy.restitution = 1.0

cfg_ground = copy.deepcopy(builder.default_shape_cfg)
cfg_ground.mu = 1.0
cfg_ground.restitution = 0.0
cfg_ground.contact_ke = 1e6
cfg_ground.contact_kd = 1e3

# Ground plane
builder.add_ground_plane(cfg=cfg_ground)

# -------------------------
# Add multiple small spheres
# -------------------------
# i = 0
# for x_loc in np.linspace(-3, 3, x_balls):
#     for z_loc in np.linspace(-3, 3, y_balls):
#         ball = builder.add_body(
#             xform=wp.transform(wp.vec3(x_loc, HEIGHT, z_loc), wp.quat_identity()),
#             mass=2.0,
#         )
#         cfg = cfg_soft if i <= 10 else cfg_bouncy
#         builder.add_shape_sphere(body=ball, radius=RADIUS, cfg=cfg)
#         i += 1

# -------------------------
# Load STL mesh instead of cube
# -------------------------
# -------------------------
# Load STL mesh instead of cube
# -------------------------
mesh_path = "sphere_mesh.stl"  # change if needed
big_mesh = pv.read(mesh_path)
big_mesh.compute_normals(inplace=True)

# Optional rescale if needed
big_mesh.scale(0.5, inplace=True)

# Create a Newton mesh from STL data
verts = big_mesh.points.astype(np.float32)
faces = big_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

mesh_obj = newton.Mesh(verts, faces)

# Create a dynamic body for the STL
stl_body = builder.add_body(
    xform=wp.transform(wp.vec3(0, HEIGHT + 4, 0), wp.quat_identity()),
    mass=2.0,
)

# Attach the mesh shape to this body
builder.add_shape_mesh(
    body=stl_body,
    xform=wp.transform(),  # identity transform in body space
    mesh=mesh_obj,
    scale=wp.vec3(1.0, 1.0, 1.0),
    cfg=cfg_soft
)


# -------------------------
# Force kernels
# -------------------------
@wp.kernel
def apply_air_drag(body_v: wp.array(dtype=vec6f), body_f: wp.array(dtype=vec6f), drag_coefficient: float):
    i = wp.tid()
    v = wp.vec3(body_v[i][0], body_v[i][1], body_v[i][2])
    speed = wp.length(v)
    if speed > 1e-6:
        drag_force = -drag_coefficient * v * speed
        f = body_f[i]
        body_f[i] = vec6f(f[0] + drag_force[0], f[1] + drag_force[1], f[2] + drag_force[2], f[3], f[4], f[5])

@wp.kernel
def apply_wind(body_v: wp.array(dtype=vec6f), body_f: wp.array(dtype=vec6f), wind: float):
    i = wp.tid()
    f = body_f[i]
    body_f[i] = vec6f(f[0] + wind, f[1], f[2], f[3], f[4], f[5])

@wp.kernel
def apply_random_force(body_f: wp.array(dtype=vec6f), strength: float, seed: int):
    i = wp.tid()
    rng = wp.rand_init(seed, i)
    fx = (wp.randf(rng) - 0.5) * 2.0 * strength
    fy = (wp.randf(rng) - 0.5) * 2.0 * strength
    fz = (wp.randf(rng) - 0.5) * 2.0 * strength
    f = body_f[i]
    body_f[i] = vec6f(f[0] + fx, f[1] + fy, f[2] + fz, f[3], f[4], f[5])

# -------------------------
# Finalize model and solver
# -------------------------
model = builder.finalize()
state_0, state_1 = model.state(), model.state()
control = model.control()
solver = SolverXPBD(model, iterations=40)

# -------------------------
# Simulation loop
# -------------------------
positions = []

for frame in range(NUM_FRAMES):
    for _ in range(SUBSTEP):
        state_0.clear_forces()

        wp.launch(apply_air_drag, dim=model.body_count, inputs=[state_0.body_qd, state_0.body_f, DRAG_COEFFICIENT])
        wp.launch(apply_wind, dim=model.body_count, inputs=[state_0.body_qd, state_0.body_f, 2.0])

        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0

    q_full = state_0.body_q.numpy()
    pos = q_full[:, :3]
    quat = q_full[:, 3:7]
    positions.append((pos.copy(), quat.copy()))

    if frame % 100 == 0:
        print(f"Frame {frame}/{NUM_FRAMES}")

# -------------------------
# Rendering setup
# -------------------------
plotter = pv.Plotter(off_screen=True)
plotter.set_background("white")
plotter.add_axes()

plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
plotter.add_mesh(plane, color="lightgray", opacity=0.8)

# Small balls
sphere_mesh = pv.Sphere(radius=RADIUS)
actors = []
for _ in range(NUM_BALLS):
    actors.append(plotter.add_mesh(sphere_mesh.copy(), color="orange", smooth_shading=True))

# STL mesh actor (green)
actors.append(plotter.add_mesh(big_mesh.copy(), color="green", smooth_shading=True))

# Camera
plotter.camera_position = [(20, 20, 20), (0, 0, 0), (0, 1, 0)]

def quat_to_mat(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ], dtype=float)

# -------------------------
# Render animation
# -------------------------
output_filename = "multi_balls_with_mesh.mp4"
plotter.open_movie(output_filename, framerate=FPS, quality=9)

for frame_idx, (pos, quat) in enumerate(positions):

    mesh = big_mesh.copy()
    R = quat_to_mat(quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    mesh.transform(T, inplace=True)
    
    actor.mapper.SetInputData(mesh)

    plotter.write_frame()
    
    if frame_idx % 100 == 0:
        print(f"Rendered frame {frame_idx}/{NUM_FRAMES}")

plotter.close()
print(f"✅ Simulation complete. Saved video as {output_filename}")
