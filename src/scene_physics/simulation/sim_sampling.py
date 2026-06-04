import json
import os       
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import newton
import warp as wp
import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt

from newton._src.sensors.sensor_tiled_camera import SensorTiledCamera
from scene_physics.visualization.camera import look_at_transform
from scene_physics.kernels.image_process import (
    render_point_cloud,
    render_point_clouds_batch,
)
from scene_physics.utils.io import save_point_cloud_ply
from dataclasses import dataclass, field
import numpy as np

from scene_physics.properties.shapes import (
    Scene_Makeup,
    object_collection
)

from scene_physics.likelihood.likelihoods import ParallelPhysicsLikelihood
from scene_physics.sampling.proposals import ExpDecayProposal
from scene_physics.sampling.importance import ImportanceSampler
from scene_physics.configs.camera import CameraIntrinsics, default_camera
from scene_physics.visualization.camera import SingleWorldCamera, MultiWorldCamera
from scene_physics.utils.io import plot_target_scene
from scene_physics.properties.shapes import Object_Collection, Hidden

NUM_WORLDS = int(os.environ.get("NUM_WORLDS", 15))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 3))

@dataclass
class Experiment:
    iterations : int
    decay_method : str


def gen_point_cloud(scene_usd, intrinsics : CameraIntrinsics) -> jax.Array:
    # Build the scene
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_usd(scene_usd, skip_mesh_approximation=True)
    
    # Create the model and render
    model = builder.finalize()
    renderer = SingleWorldCamera(intrinsics, model)

    # Generate the state and make point cloud
    state = model.state()
    pc = renderer.render(state)

    return jnp.array(pc)    # Deallocated from warp/Newton

def gen_save_point_cloud(scene_usd, intrinsics : CameraIntrinsics, save_location):
    point_cloud = gen_point_cloud(scene_usd, intrinsics)
    save_point_cloud_ply(point_cloud, save_location)

    return point_cloud

def build_final_world(scene_usd, object_collection : Object_Collection, save_dir : str):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_usd(scene_usd, skip_mesh_approximation=True)

    model = builder.finalize()

    for i, body_name in enumerate(model.body_key):
        assert body_name in object_collection.objects, f"Object {body_name} in USD not found in Object Collection"

        # NOTE only set final location for occluded objects, since we assume perfect observation of visible objects
        if isinstance(object_collection[body_name], Hidden):
            model.body_q[i] = object_collection[body_name].get_final_location()

    # TODO save used somehow in folder



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

    # Step 6: Generate Plots
    sampler.gen_plots(save_dir)
    objects.gen_plots(save_dir)

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
