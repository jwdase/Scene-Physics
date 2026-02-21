"""
End-to-end parallel 6DOF MH sampling experiment.

Uses Newton's multi-world GPU capability to evaluate num_worlds proposals
in parallel, with sequential object placement.

Usage:
    PYTHONPATH=. python src/scene_physics/simulation/run_parallel_sampling.py
"""

import os
import numpy as np
import warp as wp
import newton

from scene_physics.properties.shapes import Parallel_Mesh, Parallel_Static_Mesh
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material
from scene_physics.simulation.parallel_builder import  allocate_worlds
from scene_physics.likelihood.likelihoods_physics import Likelihood_Physics_Parallel
from scene_physics.sampling.proposals import SixDOFProposal, linear_decay
from scene_physics.sampling.parallel_mh import ParallelPhysicsMHSampler
from newton.solvers import SolverXPBD

# ─── Configuration ───────────────────────────────────────────────────────────

NUM_WORLDS = 16
ITERATIONS_PER_OBJECT = 100
POS_STD = 0.05
ROT_STD = 0.1
WIDTH = 640
HEIGHT = 480
MAX_DEPTH = 5.0
EXPERIMENT_NAME = "parallel_6dof"

# Physics simulation
SIM_SECONDS = 4
SIM_FPS = 40
SIM_DT = 1.0 / SIM_FPS
SIM_FRAMES = SIM_SECONDS * SIM_FPS

# Camera positions
WP_EYE = np.array([1., 1.5, 1.])
WP_TARGET = np.array([0., 0., 0.])

# Object mesh paths
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_ROOT = os.path.join(PACKAGE_ROOT, "objects", "scene01")


# ─── Scene Builder Function ──────────────────────────────────────────────────
def make_scene01_world():
    """Build a single-world Scene01 (no ground plane — that's added globally).

    Returns:
        (builder, bodies_dict) where bodies_dict maps name -> Body
    """

    bowl = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/BOWL.obj",
        position=(0., 0., 0.),
        material=Dynamic_Material,
        name="bowl",
    )
    coffee = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/COFFEE.obj",
        position=(1., 1., 1.),
        material=Dynamic_Material,
        name="coffee",
    )
    table = Parallel_Static_Mesh(
        body_file=f"{SCENE_ROOT}/TABLE.obj",
        position=(2., 2., 2.),
        material=Still_Material,
        name="table",
    )

    return {"dynamic" : [bowl, coffee], "static" : [table]}


def build_worlds(worlds, stat_obj, dyn_obj):
    """
    Fills each world with meshes depending on whether they're dynamic
    meshes or not, then finalizes the model
    """

    # Insert all static objects
    for obj in stat_obj:
        print(type(obj))
        assert isinstance(obj, Parallel_Static_Mesh), "Must be static"
        obj.insert_object_static(worlds)

    # Insert all dynamic objects
    for i in range(worlds.num_worlds):
        for obj in dyn_obj:
            assert isinstance(obj, Parallel_Mesh), "Must be dynamic"
            obj.insert_object(worlds, i)

    # Finalize and assign pointers to objects
    model = worlds.finalize()
    for obj in (stat_obj + dyn_obj):
        obj.give_finalized_world(model)

    return model

# ─── Main ────────────────────────────────────────────────────────────────────

def main(x):
    worlds = allocate_worlds(x)
    obj = make_scene01_world()
    model = build_worlds(worlds, obj["static"], obj["dynamic"])

    # Initialise simulation states and solver
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    solver = SolverXPBD(
        model,
        rigid_contact_relaxation=0.9,
        iterations=32,
        angular_damping=0.1,
        enable_restitution=False,
    )

    # Pre-allocate GPU-side history buffer — avoids per-frame CPU/GPU sync
    num_bodies = len(model.body_q)
    history_gpu = wp.zeros((SIM_FRAMES, num_bodies), dtype=wp.transformf, device="cuda")

    # Forward physics simulation — 4 seconds
    for frame in range(SIM_FRAMES):
        state_0.clear_forces()
        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, SIM_DT)
        state_0, state_1 = state_1, state_0

        wp.copy(dest=history_gpu[frame], src=state_0.body_q)  # stays on GPU

        if frame % 40 == 0:
            print(f"Frame {frame}/{SIM_FRAMES}")

    # Single GPU→CPU transfer at the end
    history = history_gpu.numpy()  # [frames, total_bodies, 7]

    return model, obj, history
