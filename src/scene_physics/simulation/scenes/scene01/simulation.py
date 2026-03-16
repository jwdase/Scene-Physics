"""
Standalone physics simulation of Scene01.

Builds the scene at fixed starting positions, runs forward XPBD physics,
and renders the result as an .mp4 video. No sampling — just a physics video.
"""

import os
import numpy as np

from scene_physics.properties.shapes import Parallel_Mesh, Parallel_Static_Mesh
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material
from scene_physics.visualization.scene import PhysicsVideoVisualizer
from scene_physics.properties.priors import SimulationObjects
from scene_physics.simulation.simulation import run_physics_sim_target

# ─── Configuration ───────────────────────────────────────────────────────────
OUTPUT      = "recordings/physics/scene01.mp4"
PYVISTA_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCENE_ROOT   = os.path.join(PACKAGE_ROOT, "objects", "scene01")

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bowl = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/BOWL.obj",
        position=(0., 0.5, 0.),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="bowl",
    )
    coffee = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/COFFEE.obj",
        position=(0.1, 1.0, 0.1),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="coffee",
    )
    table = Parallel_Static_Mesh(
        body_file=f"{SCENE_ROOT}/TABLE.obj",
        position=(0., 0., 0.),
        target_position=(0., 0., 0.),
        material=Still_Material,
        name="table",
    )

    # Define Object Dataclass
    objects = SimulationObjects(observed=[bowl], unobserved=[coffee], static=[table,])

    # Run the simulation
    run_physics_sim_target(objects, OUTPUT)
