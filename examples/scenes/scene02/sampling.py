"""
Scene02 sampling experiment.

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
LOCATION = "recordings/Scene02_final"
PYVISTA_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]

EXAMPLES_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_ROOT = os.path.join(EXAMPLES_ROOT, "objects", "scene02")

PRIORS = Priors(total_iter=ITERATIONS_PER_OBJECT)

# ─── Scene Builder Function ──────────────────────────────────────────────────

def make_scene02_world():
    """Build Scene02 objects."""

    battery = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/BATTERY.obj",
        position=(0., 0.5, 0.),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="battery",
        priors=PRIORS,
    )
    circle = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/CIRCULAR.obj",
        position=(0.1, 1.0, 0.1),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="circle",
        priors=PRIORS,
    )
    table = Parallel_Static_Mesh(
        body_file=f"{SCENE_ROOT}/TABLE.obj",
        position=(0., 0., 0.),
        target_position=(0., 0., 0.),
        material=Still_Material,
        name="table",
    )

    return SimulationObjects(observed=[battery], static=[table], unobserved=[circle])

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    objects = make_scene02_world()
    run_importance_sampling(
        objects,
        location=LOCATION,
        num_worlds=NUM_WORLDS,
        iter_per_obj=ITERATIONS_PER_OBJECT,
        total_iterations=TOTAL_ITERATIONS,
        decay=DECAY,
        pyvista_camera=PYVISTA_CAMERA,
    )
