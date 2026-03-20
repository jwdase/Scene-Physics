"""
Scene01 sampling experiment.

Defines scene objects and runs importance sampling via the shared pipeline.
"""

import os

from scene_physics.properties.priors import Priors, SimulationObjects
from scene_physics.properties.shapes import Parallel_Mesh, Parallel_Static_Mesh
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material
from scene_physics.simulation.sampling import run_importance_sampling


# ─── Configuration ───────────────────────────────────────────────────────────

NUM_WORLDS = 100
ITERATIONS_PER_OBJECT = 50
TOTAL_ITERATIONS = 5
DECAY = "exp"
LOCATION = "recordings/Scene01_final"
PYVISTA_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]

EXAMPLES_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_ROOT = os.path.join(EXAMPLES_ROOT, "objects", "scene01")

PRIORS = Priors(total_iter=ITERATIONS_PER_OBJECT)

# ─── Scene Builder Function ──────────────────────────────────────────────────

def make_scene01_world():
    """Build Scene01 objects."""

    bowl = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/BOWL.obj",
        position=(0., 0., 0.),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="bowl",
        priors=PRIORS,
    )
    coffee = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/COFFEE.obj",
        position=(1., 1., 1.),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="coffee",
        priors=PRIORS,
    )
    table = Parallel_Static_Mesh(
        body_file=f"{SCENE_ROOT}/TABLE.obj",
        position=(2., 2., 2.),
        target_position=(0., 0., 0.),
        material=Still_Material,
        name="table",
    )

    return SimulationObjects(observed=[bowl], static=[table], unobserved=[coffee])

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    objects = make_scene01_world()
    run_importance_sampling(
        objects,
        location=LOCATION,
        num_worlds=NUM_WORLDS,
        iter_per_obj=ITERATIONS_PER_OBJECT,
        total_iterations=TOTAL_ITERATIONS,
        decay=DECAY,
        pyvista_camera=PYVISTA_CAMERA,
    )
