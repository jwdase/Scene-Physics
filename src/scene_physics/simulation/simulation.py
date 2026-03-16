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

# Defaults
SIM_SECONDS = 3
SIM_FPS     = 40
PYVISTA_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]

def run_physics_sim_target(objects, path, sim_seconds=SIM_SECONDS, sim_fps=SIM_FPS, camera=PYVISTA_CAMERA):
    """
    Runs a visualization of a given seen and outputs
    it to output direcory
    """
    
    # Places in finalized positions
    for obj in objects.all:
        if isinstance(obj, Parallel_Static_Mesh):
            continue
        obj.set_final_positions_to_target()

    # Ensures good directory
    path_parts = path.split("/")
    if len(path_parts) > 1:
        os.makedirs("".join(path[:-1]))

    # Rendering simulation
    print("Rendering Simulation")
    visualizer = PhysicsVideoVisualizer(objects, FPS=sim_fps, camera_pos=camera)
    visualizer.render_final_scene(path, frames=sim_fps * sim_seconds)

if __name__ == "__main__":
    pass
