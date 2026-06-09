import json
import os
from dataclasses import dataclass       
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import newton
import numpy as np

from scene_physics.utils.plots import gen_save_point_cloud, build_final_world, plot_target_scene
from scene_physics.properties.structs import Scene_Makeup, object_collection
from scene_physics.likelihood.likelihoods import ParallelPhysicsLikelihood
from scene_physics.sampling.importance import ImportanceSampler
from scene_physics.configs.camera import CameraIntrinsics, default_camera
from scene_physics.visualization.camera import MultiWorldCamera

# Sampler we get to choose
from scene_physics.sampling.proposals import ExpDecayProposal

NUM_WORLDS = int(os.environ.get("NUM_WORLDS", 15))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 50))

@dataclass
class Experiment:
    iterations : int
    decay_method : str


def build_worlds(scene_usd, scene_makeup : Scene_Makeup):
    blueprint = newton
    blueprint = newton.ModelBuilder()
    blueprint.add_usd(scene_usd, skip_mesh_approximation=True)

    # Replicate Across N Worlds
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    builder.replicate(blueprint, NUM_WORLDS, spacing=(0.0, 0.0, 0.0))

    model = builder.finalize()

    # Create Object Collection and Return
    objects = object_collection(model, scene_makeup, NUM_WORLDS)
    return model, objects


def run_importance_sampling(scene_usd, prior_json, truth_json, intrinsics : CameraIntrinsics, scene_makeup : Scene_Makeup, save_dir : str, iterations=20):
    base_rng = np.random.default_rng(seed=42)

    # Step -1: Print States
    print("======")
    print(f"Scene: {scene_usd}")
    print(f"Number of Worlds: {NUM_WORLDS}")
    print(f"Number of Epochs: {NUM_EPOCHS}")

    # Step 0: Plot Target Scene
    plot_target_scene(truth_json, save_dir)

    # Get point cloud and save
    point_cloud = gen_save_point_cloud(scene_usd, intrinsics, f"{save_dir}/point_cloud.ply")

    # Step 2: Build our new world
    model, objects = build_worlds(scene_usd, scene_makeup)

    # Step 3: Create Likelihood Function
    multiCamera = MultiWorldCamera(intrinsics, model, num_worlds=NUM_WORLDS)
    likelihoodf = ParallelPhysicsLikelihood(multiCamera, point_cloud, model)

    # Step 4: Insert Priors on Shapes
    objects.assign_priors(prior_json, ExpDecayProposal, iterations, base_rng)

    # Step 5: Initialize Importance Sampler
    scene = model.state()
    sampler = ImportanceSampler(objects, likelihoodf, scene, save_dir)

    # Step 5: Run Importance Sampling
    sampler.initialize()
    sampler.gibb_sample(iterations=iterations)

    # Step 6: Feed Correct Values
    objects.assign_correct(truth_json)

    # Step 7: Generate Plots
    sampler.gen_plots(save_dir)
    objects.gen_plots(save_dir)

    # Step 8: Build Final World and Save
    build_final_world(scene_usd, objects, save_dir)

    return scene, model, likelihoodf

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Accept scene name as argument, default to scene001
    scene_name = sys.argv[1] if len(sys.argv) > 1 else "scene002"

    # Absolute path anchored to project root (three levels up from simulation/)
    project_root = Path(__file__).resolve().parents[3]   # → Scene-Physics/
    scene_dir = project_root / "resources" / "generated_scenes" / scene_name

    scene_usd  = str(scene_dir / "data" / f"{scene_name}_physics.usdc")
    priors     = str(scene_dir / "data" / f"{scene_name}_modified_priors.json")
    truth_json = str(scene_dir / "data" / f"{scene_name}_truth.json")
    folder     = str(scene_dir / "results")

    makeup_json = str(scene_dir / "data" / f"{scene_name}_makeup.json")
    with open(makeup_json) as f:
        mk = json.load(f)

    scene_makeup = Scene_Makeup(
        static=mk["static"],
        observed=mk["observed"],
        hidden=mk["hidden"],
    )

    scene, model, x = run_importance_sampling(scene_usd, priors, truth_json, default_camera, scene_makeup, folder, NUM_EPOCHS)
