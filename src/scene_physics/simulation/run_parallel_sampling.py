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

from scene_physics.properties.shapes import MeshBody
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material
from scene_physics.simulation.parallel_builder import build_parallel_worlds
from scene_physics.likelihood.likelihoods_physics import Likelihood_Physics_Parallel
from scene_physics.sampling.proposals import SixDOFProposal, linear_decay
from scene_physics.sampling.parallel_mh import ParallelPhysicsMHSampler

# ─── Configuration ───────────────────────────────────────────────────────────

NUM_WORLDS = 16          # Scale up on H100 (16, 32, 64, ...)
ITERATIONS_PER_OBJECT = 100
POS_STD = 0.05           # Initial position proposal std (meters)
ROT_STD = 0.1            # Initial rotation proposal std (radians)
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
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

    bowl = MeshBody(
        builder=builder,
        body=f"{SCENE_ROOT}/BOWL.obj",
        material=Dynamic_Material,
        name="bowl",
    )
    coffee = MeshBody(
        builder=builder,
        body=f"{SCENE_ROOT}/COFFEE.obj",
        material=Dynamic_Material,
        name="coffee",
    )
    table = MeshBody(
        builder=builder,
        body=f"{SCENE_ROOT}/TABLE.obj",
        material=Still_Material,
        name="table",
    )

    bodies_dict = {
        "bowl": bowl,
        "coffee": coffee,
        "table": table,
    }
    return builder, bodies_dict


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Building {NUM_WORLDS} parallel worlds...")

    # Build parallel worlds
    main_builder, world_bodies, body_index_map = build_parallel_worlds(
        base_builder_fn=make_scene01_world,
        num_worlds=NUM_WORLDS,
    )

    # Finalize model
    model = main_builder.finalize()
    print(f"Model finalized: {len(model.body_q)} total bodies across {NUM_WORLDS} worlds")

    # Create target state (ground truth at default positions)
    target_state = model.state()

    # Build parallel likelihood
    print("Setting up parallel likelihood...")
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

    # Configure 6DOF proposals with variance schedule
    proposal = SixDOFProposal(
        pos_std=POS_STD,
        rot_std=ROT_STD,
        schedule=linear_decay,
    )

    # Sequential placement order: sample dynamic objects, table is static
    placement_order = ["bowl", "coffee"]

    # Optional: initialize near ground truth for visible objects
    init_positions = {
        "bowl": (np.array([0.0, 0.0, 0.0]) + np.random.normal(0, 0.1, size=3),
                 np.array([0.0, 0.0, 0.0, 1.0])),
        "coffee": (np.array([0.0, 0.0, 0.0]) + np.random.normal(0, 0.1, size=3),
                   np.array([0.0, 0.0, 0.0, 1.0])),
    }

    # Create sampler
    sampler = ParallelPhysicsMHSampler(
        model=model,
        likelihood=likelihood,
        placement_order=placement_order,
        world_bodies=world_bodies,
        body_index_map=body_index_map,
        num_worlds=NUM_WORLDS,
        proposal=proposal,
    )

    # Run sampling
    print(f"\nRunning parallel MH sampling ({ITERATIONS_PER_OBJECT} iterations per object)...")
    print(f"  num_worlds={NUM_WORLDS}, pos_std={POS_STD}, rot_std={ROT_STD}")
    print(f"  placement_order={placement_order}")

    results = sampler.run_sampling(
        iterations_per_object=ITERATIONS_PER_OBJECT,
        init_positions=init_positions,
        debug=True,
    )

    # Save results
    os.makedirs(f"recordings/{EXPERIMENT_NAME}", exist_ok=True)
    for body_name, result in results.items():
        pos = result['position']
        quat = result['quat']
        scores = result['scores']

        print(f"\n{body_name}:")
        print(f"  Final position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        print(f"  Final quaternion: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
        print(f"  Final score: {scores[-1]:.4f}")

        np.save(f"recordings/{EXPERIMENT_NAME}/{body_name}_scores.npy", np.array(scores))

    print(f"\nResults saved to recordings/{EXPERIMENT_NAME}/")


if __name__ == "__main__":
    main()
