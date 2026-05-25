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
from scene_physics.sampling.proposals import NoDecayProposal
from scene_physics.sampling.importance import ImportanceSampler

EYE = np.array([0., -1.5, 1.5])
TARGET = np.zeros(3)
UP = np.array([0, 0, 1]) 

UP_AXIS = newton.Axis.Z 
NUM_WORLDS = int(os.environ.get("NUM_WORLDS", 15))

@dataclass
class CameraIntrinsics:
    width : int
    height : int
    fov_degree: float
    max_depth : float = 4.0

    eye : np.ndarray = field(default_factory=lambda:EYE)
    target : np.ndarray = field(default_factory=lambda:TARGET)
    up : np.ndarray = field(default_factory=lambda:UP)

    @property
    def fov_rad(self):
        return np.radians(self.fov_degree)


@dataclass
class Experiment:
    iterations : int
    decay_method : str


NUM_CAMERAS = 1

class Camera:
    def __init__(self, intrinsics : CameraIntrinsics, model, num_worlds : int):
        self.intrinsics = intrinsics
        self.model = model
        self.num_worlds = num_worlds

        self.sensor = SensorTiledCamera(
                model=model, num_cameras=NUM_CAMERAS,
                width=intrinsics.width, height=intrinsics.height
                )

        self.camera_rays = self.sensor.compute_pinhole_camera_rays(intrinsics.fov_rad)

        t = look_at_transform(intrinsics.eye, intrinsics.target, intrinsics.up)
        self.camera_transforms = wp.array(
                [[t] * self.num_worlds], dtype=wp.transformf, ndim=2
                )

        self.depth_image = self.sensor.create_depth_image_output()
        self.points_gpu = wp.empty(
                self.depth_image.shape,
                dtype=wp.vec3f
            )

    def render(self, state):
        raise NotImplementedError("Can not render on Camera")


class SingleWorldCamera(Camera):
    def __init__(self, intrinsics : CameraIntrinsics, model):
        super().__init__(intrinsics, model, num_worlds=1)

    def render(self, state):
        return render_point_cloud(
            self.sensor, state, self.camera_transforms, self.camera_rays,
            self.depth_image, self.points_gpu,
            self.intrinsics.height, self.intrinsics.width, self.intrinsics.max_depth,
        )


class MultiWorldCamera(Camera):
    def __init__(self, intrinsics : CameraIntrinsics, model):
        super().__init__(intrinsics, model, num_worlds=NUM_WORLDS)

    def render(self, state):
        return render_point_clouds_batch(
            self.sensor, state, self.camera_transforms, self.camera_rays,
            self.depth_image, self.points_gpu,
            self.intrinsics.height, self.intrinsics.width, self.intrinsics.max_depth,
            self.num_worlds,
        )
    
def plot_target_scene(truth_json : str, save_dir : str):
    with open(truth_json, 'r') as f:
        truth = json.load(f)

    labels = list(truth.keys())
    positions = [truth[obj][:3] for obj in labels]

    plt.figure(figsize=(8, 6))

    for lab, pos in zip(labels, positions):
        plt.scatter([pos[0]], [pos[1]], label=lab)


    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Target Scene")
    plt.legend()
    plt.savefig(f"{save_dir}/target_scene.png")
    plt.close()

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
    multiCamera = MultiWorldCamera(intrinsics, model)
    likelihoodf = ParallelPhysicsLikelihood(multiCamera, point_cloud, model)

    # Step 4: Insert Priors on Shapes
    objects.assign_priors(prior_json, NoDecayProposal, base_rng)

    # Step 5: Initialize Importance Sampler
    scene = model.state()
    sampler = ImportanceSampler(objects, likelihoodf, scene)

    # Step 5: Run Importance Sampling
    sampler.initialize()
    sampler.gibb_sample(iterations=iterations)

    # Step 6: Generate Plots
    sampler.gen_plots(save_dir)
    objects.gen_plots(save_dir)

    

    return scene, model, likelihoodf





default_camera = CameraIntrinsics(width=640, height=480, fov_degree=60,)

if __name__ == "__main__":

    scene_usd = "scene002/data/scene002_physics.usdc"
    priors = "scene002/data/scene002_priors.json"
    truth_json = "scene002/data/scene002_truth.json"

    folder = "scene002/results"

    makeup_json = "scene002/data/scene002_makeup.json"
    with open(makeup_json) as f:
        mk = json.load(f)
    scene_makeup = Scene_Makeup(
        static=mk["static"],
        observed=mk["observed"],
        hidden=mk["hidden"],
    )

    scene, model, x = run_importance_sampling(scene_usd, priors, truth_json, default_camera, scene_makeup, folder, 10)
