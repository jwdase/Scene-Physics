"""
Runs a physics simulation on the scene and outputs a .usdc recording of the the physics simulator
"""

import json
from pathlib import Path

import warp as wp
import newton
from newton.viewer import ViewerGL, ViewerUSD

# Drive the re-simulation with the exact solver setup used to generate the dataset, so a
# re-sim of an exported scene matches scene_gen's settle: same gravity / up-axis, substep
# count, solver iterations, and frame/sub-step dt. Imported (not copied) to stay in sync.
from scene_physics.data_gen.scene_gen import (
    GRAVITY,
    SOLVER_ITERS,
    SUBSTEPS,
    DT,
    SUB_DT,
    MU,
    RESTITUTION,
    TABLE,
)
from scene_physics.data_gen.object_library import load_object

FPS = 1.0 / DT          # frame dt matches scene_gen.DT (1/60 s)
DURATION = 4
VERTICAL = "Z"


def _add_table_box_collider(builder, scene_usd):
    """Add scene_gen's solid box table collider (spanning the table's AABB) to `builder`.

    scene_gen settles against a solid box for the table, NOT the thin table shell — but only
    the visual mesh is written to the USD, so `add_usd` reloads the table as a thin-shell
    collider that dropped objects tunnel through. Re-create the box here so the re-sim is
    stable in the same way the generator's settle was. The table is authored at the origin,
    so its world AABB equals the library object's local AABB. The box top sits at aabb_max[2]
    (the mesh top), matching rest heights; the still-present thin mesh collider is harmless
    because the box stops objects first. Material matches scene_gen's static collider.
    """
    makeup_path = Path(str(scene_usd).replace("_physics.usdc", "_makeup.json"))
    # makeup["static"][0] is the (USD-safe) table name; for the dataset table it equals the
    # library .obj name. Fall back to scene_gen's default table if no makeup sits alongside.
    table_name = json.loads(makeup_path.read_text())["static"][0] if makeup_path.exists() else TABLE
    table = load_object(table_name)

    half = 0.5 * (table.aabb_max - table.aabb_min)
    center = 0.5 * (table.aabb_max + table.aabb_min)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(
            (float(center[0]), float(center[1]), float(center[2])), wp.quat_identity()
        ),
        hx=float(half[0]), hy=float(half[1]), hz=float(half[2]),
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU, restitution=RESTITUTION),
    )

def run_simulation_save(scene_usd, output_path):

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
    builder.add_ground_plane()
    builder.add_usd(scene_usd, skip_mesh_approximation=True)
    _add_table_box_collider(builder, scene_usd)

    model = builder.finalize()
    state = model.state()

    solver = newton.solvers.SolverXPBD(model, iterations=SOLVER_ITERS)
    control = model.control()
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

    t = 0.0

    for _ in range(num_frames):
        usd_viewer.begin_frame(t)
        gl_viewer.begin_frame(t)

        for _ in range(SUBSTEPS):
            state.clear_forces()
            contacts = model.collide(state)
            state_next = model.state()
            solver.step(state, state_next, control, contacts, SUB_DT)
            state = state_next

        usd_viewer.log_state(state)
        gl_viewer.log_state(state)

        usd_viewer.end_frame()
        gl_viewer.end_frame()
        t += DT

    usd_viewer.close()

    while gl_viewer.is_running():
        gl_viewer.begin_frame(t)
        gl_viewer.log_state(state)
        gl_viewer.end_frame()

    gl_viewer.close()


def run_simulation(scene_usd, _):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
    builder.add_ground_plane()
    builder.add_usd(scene_usd, skip_mesh_approximation=True)
    _add_table_box_collider(builder, scene_usd)

    model = builder.finalize()
    state = model.state()

    solver = newton.solvers.SolverXPBD(model, iterations=SOLVER_ITERS)
    control = model.control()
    contacts = model.collide(state)

    num_frames = int(DURATION * FPS)
    gl_viewer = ViewerGL()

    gl_viewer.set_model(model)

    t = 0.0

    for _ in range(num_frames):
        gl_viewer.begin_frame(t)

        for _ in range(SUBSTEPS):
            state.clear_forces()
            contacts = model.collide(state)
            state_next = model.state()
            solver.step(state, state_next, control, contacts, SUB_DT)
            state = state_next

        gl_viewer.log_state(state)

        gl_viewer.end_frame()
        t += DT

    while gl_viewer.is_running():
        gl_viewer.begin_frame(t)
        gl_viewer.log_state(state)
        gl_viewer.end_frame()

    gl_viewer.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        run_simulation(sys.argv[1], sys.argv[2])
    else:
        print("Usage: simulation.py <scene_usd> <output_path>")
        sys.exit(1)


