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

from scene_physics.properties.shapes import Parallel_Mesh
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material
from scene_physics.simulation.parallel_builder import  allocate_worlds
from scene_physics.likelihood.likelihoods_physics import Likelihood_Physics_Parallel
from scene_physics.sampling.proposals import SixDOFProposal, linear_decay
from scene_physics.sampling.parallel_mh import ParallelPhysicsMHSampler

# ─── Configuration ───────────────────────────────────────────────────────────

NUM_WORLDS = 16
ITERATIONS_PER_OBJECT = 100
POS_STD = 0.05
ROT_STD = 0.1
WIDTH = 640
HEIGHT = 480
MAX_DEPTH = 5.0
EXPERIMENT_NAME = "parallel_6dof"

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
    table = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/TABLE.obj",
        position=(2., 2., 2.),
        material=Still_Material,
        name="table",
    )

    return [bowl, coffee, table]


def build_worlds(worlds, objects):

    # Insert all the objects
    for i in range(worlds.num_worlds):
        for obj in objects:
            obj.insert_object(worlds, i)

    # Finalize and assign pointers to objects
    model = worlds.finalize()
    for obj in objects:
        obj.give_finalized_world(model)

    return model

    




# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    worlds = allocate_worlds(10)                # Alocate our large worlds
    objects = make_scene01_world()

    model = build_worlds(worlds, objects)

    return model, objects





