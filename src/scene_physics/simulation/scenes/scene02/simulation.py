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

# ─── Configuration ───────────────────────────────────────────────────────────

SIM_SECONDS = 3
SIM_FPS     = 40
OUTPUT      = "recordings/physics/scene02.mp4"

PYVISTA_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCENE_ROOT   = os.path.join(PACKAGE_ROOT, "objects", "scene02")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():

    battery = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/BATTERY.obj",
        position=(0., 0.5, 0.),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="battery",
    )
    circle = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/CIRCULAR.obj",
        position=(0.1, 1.0, 0.1),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="circle",
    )
    table = Parallel_Static_Mesh(
        body_file=f"{SCENE_ROOT}/TABLE.obj",
        position=(0., 0., 0.),
        target_position=(0., 0., 0.),
        material=Still_Material,
        name="table",
    )

    # PhysicsVideoVisualizer starts physics from final_position.
    # Set to the starting poses above: [x, y, z, qx, qy, qz, qw].
    battery.final_position   = np.array([ 9.9889427e-02, -1.7543495e-02, -8.9898050e-02, -7.4992753e-03, -1.6852148e-04, -6.5252934e-05, 9.9997187e-01], dtype=np.float32)
    circle.final_position = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32)

    objects = {"observed": [], "unobserved": [battery], "static": [table]}

    os.makedirs("recordings", exist_ok=True)

    print("Rendering simulation...")
    visualizer = PhysicsVideoVisualizer(objects, FPS=SIM_FPS, camera_pos=PYVISTA_CAMERA)
    visualizer.render_final_scene(OUTPUT, frames=SIM_SECONDS * SIM_FPS)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
