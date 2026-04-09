"""
Standalone physics simulation of Scene01.

Builds the scene at fixed starting positions, runs forward XPBD physics,
and renders the result as an .mp4 video. No sampling — just a physics video.
"""

import os
import numpy as np

from scene_physics.properties.shapes import Parallel_Static_Mesh
from scene_physics.visualization.scene import PhysicsVideoVisualizer

# Defaults
SIM_SECONDS = 3
SIM_FPS     = 40
PYVISTA_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]

def run_physics_sim_target(objects, path, sim_seconds=SIM_SECONDS, sim_fps=SIM_FPS, camera=PYVISTA_CAMERA, cam_fov=None):
    """
    Runs a visualization of a given seen and outputs
    it to output direcory
    """
    
    # Places in finalized positions
    for obj in objects.all_sampled:
        if isinstance(obj, Parallel_Static_Mesh):
            continue
        obj.set_final_position_to_target()

    # Ensures good directory
    path_parts = path.split("/")
    if len(path_parts) > 1:
        os.makedirs("/".join(path_parts[:-1]), exist_ok=True)

    # Rendering simulation
    print("Rendering Simulation")
    visualizer = PhysicsVideoVisualizer(objects, FPS=sim_fps, camera_pos=camera, cam_fov = cam_fov)
    visualizer.render_final_scene(path, frames=sim_fps * sim_seconds, dt=1/sim_fps)

if __name__ == "__main__":
    pass
