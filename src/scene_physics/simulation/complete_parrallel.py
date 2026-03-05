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
from scene_physics.visualization.scene import PyVistaVisualizer
from newton.solvers import SolverXPBD

# ─── Configuration ───────────────────────────────────────────────────────────

NUM_WORLDS = 20
ITERATIONS_PER_OBJECT = 20
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
PYVISTA_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]

# Object mesh paths
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_ROOT = os.path.join(PACKAGE_ROOT, "objects", "scene01")


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
    )
    coffee = Parallel_Mesh(
        body_file=f"{SCENE_ROOT}/COFFEE.obj",
        position=(1., 1., 1.),
        target_position=(0., 0., 0.),
        material=Dynamic_Material,
        name="coffee",
    )
    table = Parallel_Static_Mesh(
        body_file=f"{SCENE_ROOT}/TABLE.obj",
        position=(2., 2., 2.),
        target_position=(0., 0., 0.),
        material=Still_Material,
        name="table",
    )

    return {"observed" : [bowl, coffee], "static" : [table], "unobserved": []}


def build_worlds(worlds, objects):
    """
    Fills each world with meshes depending on whether they're dynamic
    meshes or not, then finalizes the model

    Args:
        worlds: world builder
        stat_obj: objects that exist in all sim and don't move
        dyn_obj_ob: all observable objects
        dyn_obj_un: all unobservable objects

    Returns:
        model: Our finalized model
        objects: list of objects in order which they should be inserted
    """
    all_objects = objects["static"] + objects["observed"] + objects["unobserved"]

    # Insert all static objects
    for obj in objects["static"]:
        print(type(obj))
        assert isinstance(obj, Parallel_Static_Mesh), "Must be static"
        obj.insert_object_static(worlds)

    # Insert all observed dynamic objects
    for i in range(worlds.num_worlds):
        for obj in objects["observed"]:
            assert isinstance(obj, Parallel_Mesh), "Must be dynamic"
            obj.insert_object(worlds, i)


    for i in range(worlds.num_worlds):
        for obj in objects["unobserved"]:
            assert isinstance(obj, Parallel_Mesh), "Must be dynamic"
            obj.insert_object(worlds, i)

    # Finalize and assign pointers to objects
    model = worlds.finalize()
    for obj in all_objects:
        obj.give_finalized_world(model)
    
    # Take state of correct placement
    for obj in all_objects:
        obj.move_to_target()
    
    target = model.state()

    # Hide all objects that are not static
    for obj in (objects["observed"] + objects["unobserved"]):
        obj.freeze_finalized_body()
    
    return model, target


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
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
    sampler = ParallelPhysicsMHSampler(model, likelihood, obj, iter_per_obj=ITERATIONS_PER_OBJECT, visualization=visualizer, name=f"recordings/{EXPERIMENT_NAME}/")

    sampler.run_sampling_linear_print(debug=True)
    sampler.print_results()
    sampler.plot_proposal_scores()

    print("Saving Final Result")
    visualizer.show_final_scene(f"recordings/{EXPERIMENT_NAME}/final_position.png")
    

if __name__ == "__main__":
    main()



