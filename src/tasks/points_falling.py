import numpy as np
import warp as wp

import pyvista as pv

# Physics Properties
GRAVITY = wp.vec3([0, 0, -9.8])
DRAG = .2

# Assigned Values
MASS = .1
FLOOR = 1.0
DT = 1E-3
RADIUS = 0.5

def gen_balls(samples, loc):
    """ Generate Random Positions """    

    rng = np.random.default_rng(2025)

    pos = rng.uniform(low=loc[0], high=loc[1], size=(samples, 3))
    return wp.array(pos, dtype=wp.vec3)


def gen_velocity(samples, speed):
    """Generate Velocities"""
    rng = np.random.default_rng(2025)

    vel = rng.normal(loc=0.0, scale=speed, size=(samples, 3))

    return wp.array(vel, dtype=wp.vec3)    

@wp.kernel
def integrate(position:wp.array(dtype=wp.vec3), velocity:wp.array(dtype=wp.vec3)):
    i = wp.tid()

    # Update Velocity
    acceleration = (-DRAG * velocity[i]) / MASS + GRAVITY
    velocity[i] = velocity[i] + acceleration * DT

    new_z = position[i][2] + velocity[i][2] * DT

    # Adds Bouncing Feature    
    if new_z < FLOOR + RADIUS:

        # New Floor
        new_z = FLOOR + RADIUS

        # Update Velocity
        velocity[i] = wp.vec3(
            velocity[i][0],
            velocity[i][1],
            -.8 * velocity[i][2]
        )


    position[i] = wp.vec3(
        position[i][0] + velocity[i][0] * DT,
        position[i][1] + velocity[i][0] * DT,
        new_z
    )


samples = 10
steps = 5000

print(f'length of simulation: {steps * DT} seconds')

positions = gen_balls(samples, (2, 10))
velocity = gen_velocity(samples, 0)

frames = []

for step in range(steps):
    wp.launch(integrate, dim=samples, inputs=[positions, velocity])

    if step % 10 == 0:
        frames.append(positions.numpy().copy())


# --- PyVista Visualization ---
plotter = pv.Plotter(off_screen=True)
plotter.open_movie("balls_simulation.mp4", framerate=100)

# Change background color
plotter.set_background("black")  # options: "white", "lightgray", "#1e1e1e", etc.

# Optional: add ambient lighting for brightness
plotter.enable_eye_dome_lighting()  # improves 3D contrast
plotter.enable_anti_aliasing()
plotter.add_light(pv.Light(position=(10, 10, 20), intensity=2.0))

# Create a floor plane
floor_plane = pv.Plane(
    center=(6, 6, FLOOR),
    direction=(0, 0, 1),
    i_size=20,
    j_size=20
)

for frame in frames:
    plotter.clear()
    plotter.add_mesh(floor_plane, color="lightgray", specular=0.3, smooth_shading=True)

    for pos in frame:
        sphere = pv.Sphere(radius=RADIUS, center=pos)
        # Bright, reflective material
        plotter.add_mesh(
            sphere,
            color="crimson",     # bright blue
            smooth_shading=True,
            specular=1.0,            # make it shiny
            specular_power=20
        )

    plotter.write_frame()

plotter.close()
print("✅ Simulation complete — saved to balls_simulation.mp4")
