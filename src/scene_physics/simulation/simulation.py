"""
Runs a physics simulation on the scene and outputs a .usdc recording of the the physics simulator
"""

import newton
from newton.viewer import ViewerGL, ViewerUSD


FPS = 60.0
DURATION = 4
VERTICAL = "Z"
SUBSTEPS = 4
SOLVER_ITERATIONS = 8

def run_simulation(scene_usd, output_path):

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_usd(scene_usd, skip_mesh_approximation=True)

    model = builder.finalize()
    state = model.state()

    solver = newton.solvers.SolverXPBD(model, iterations=SOLVER_ITERATIONS)
    contacts = model.collide(state)

    num_frames = int(DURATION * FPS)

    usd_viewer = ViewerUSD(
        output_path=output_path,
        fps=FPS,
        up_axis=VERTICAL,
        num_frames=num_frames,
    )
    gl_viewer = ViewerGL()

    usd_viewer.set_model(model)
    gl_viewer.set_model(model)

    dt = 1.0 / FPS
    sub_dt = dt / SUBSTEPS
    t = 0.0

    for _ in range(num_frames):
        usd_viewer.begin_frame(t)
        gl_viewer.begin_frame(t)

        for _ in range(SUBSTEPS):
            contacts = model.collide(state)
            state_next = model.state()
            solver.step(state, state_next, None, contacts, sub_dt)
            state = state_next

        usd_viewer.log_state(state)
        gl_viewer.log_state(state)

        usd_viewer.end_frame()
        gl_viewer.end_frame()
        t += dt

    usd_viewer.close()

    while gl_viewer.is_running():
        gl_viewer.begin_frame(t)
        gl_viewer.log_state(state)
        gl_viewer.end_frame()

    gl_viewer.close()



if __name__ == "__main__":
    scene_usd = "scene01_physics.usdc"
    output_path = "scene01_recording.usdc"

    run_simulation(scene_usd, output_path)

    

