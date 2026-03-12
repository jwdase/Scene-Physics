"""
End-to-end parallel 6DOF MH sampling experiment.

Uses Newton's multi-world GPU capability to evaluate num_worlds proposals
in parallel, with sequential object placement.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import warp as wp
import newton

from scene_physics.properties.priors import Priors
from scene_physics.utils.setup import build_worlds
from scene_physics.properties.shapes import Parallel_Mesh, Parallel_Static_Mesh
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material
from scene_physics.utils.parallel_builder import allocate_worlds
from scene_physics.likelihood.likelihoods_physics import Likelihood_Physics_Parallel
from scene_physics.sampling.parallel_mh import ImportanceSampling
from scene_physics.visualization.scene import PyVistaVisualizer, PhysicsVideoVisualizer
from newton.solvers import SolverXPBD


# ─── Configuration ───────────────────────────────────────────────────────────

NUM_OBJECTS = 2
NUM_WORLDS = 500 # (8 GB)
ITERATIONS_PER_OBJECT = 50
WIDTH = 320
HEIGHT = 240
MAX_DEPTH = 4.0
DECAY = "exp"
EXPERIMENT_NAME = "Scene01_final"
LOCATION = f"recordings/{EXPERIMENT_NAME}"
TOTAL_ITERATIONS = 50 * 2

# Physics simulation
SIM_SECONDS = 2
SIM_FPS = 40
SIM_DT = 1.0 / SIM_FPS
SIM_FRAMES = SIM_SECONDS * SIM_FPS

# Camera positions
WP_EYE = np.array([1., 1.5, 1.])
WP_TARGET = np.array([0., 0., 0.])
PYVISTA_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]

# Object mesh paths
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_ROOT = os.path.join(PACKAGE_ROOT, "objects", "scene01")

PRIORS = Priors(total_iter=ITERATIONS_PER_OBJECT)

# ─── Scene Builder Function ──────────────────────────────────────────────────

def make_scene01_world():
    """
    Build a single-world Scene01 (no ground plane — that's added globally).

    Returns:
        Dict: where bodies_dict maps name -> Body
    """

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

    return {"observed" : [bowl], "static" : [table], "unobserved": [coffee]}

# ─── Main ────────────────────────────────────────────────────────────────────

def main():

    print(f"Saving to: {LOCATION}")
    print("Building Model")
    worlds = allocate_worlds(NUM_WORLDS)
    obj = make_scene01_world()
    model, target_state = build_worlds(worlds, obj)

    print("Building Visualizer")
    visualizer = PyVistaVisualizer(obj, num_worlds=NUM_WORLDS, camera_pos=PYVISTA_CAMERA)

    print("Building Likelihood Function")
    likelihood = Likelihood_Physics_Parallel(
        target_state=target_state,
        model=model,
        wp_eye=WP_EYE,
        wp_target=WP_TARGET,
        num_worlds=NUM_WORLDS,
        name=EXPERIMENT_NAME,
        max_depth=MAX_DEPTH,
        height=HEIGHT,
        width=WIDTH,
    )

    
    print("Building Sampler")
    sampler = ImportanceSampling(model, likelihood, obj, iter_per_obj=ITERATIONS_PER_OBJECT, visualization=visualizer, name=LOCATION, decay=DECAY)

    # sampler.run_sampling_linear_print(debug=True)
    sampler.run_sampling_gibbs(iters=TOTAL_ITERATIONS, debug=False)

    # Runs Plots
    sampler.print_results()
    sampler.plot_proposal_scores()
    sampler.plot_proposal_stds()
    sampler.sampler.plot_avg_score()

    print("Saving .png Final Result")
    visualizer.show_final_scene(f"{LOCATION}/final_position.png")

    print("Filming Final Result")
    video_visualizer = PhysicsVideoVisualizer(obj, FPS=SIM_FPS, camera_pos=PYVISTA_CAMERA)
    video_visualizer.render_final_scene(f"{LOCATION}/final_state.mp4")


    

if __name__ == "__main__":
    main()



