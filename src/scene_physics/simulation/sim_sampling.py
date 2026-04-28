import os       
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import newton
import warp as wp
import jax 
import jax.numpy as jnp

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

EYE = np.array([0., -1.5, 1.5])
TARGET = np.zeros(3)
UP = np.array([0, 0, 1]) 

UP_AXIS = newton.Axis.Z 
NUM_WORLDS = 15

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
    objects = object_collection(model, scene_makeup)
    return model, objects


def run_importance_sampling(scene_usd, prior_json, intrinsics : CameraIntrinsics, scene_makeup : Scene_Makeup, save_dir : str):
    base_rng = np.random.default_rng(seed=42)

    # Get point cloud and save
    point_cloud = gen_save_point_cloud(scene_usd, intrinsics, f"{save_dir}/point_cloud.ply")

    # Step 2: Build our new world
    model, objects = build_worlds(scene_usd, scene_makeup)

    # Step 3: Create Likelihood Function
    multiCamera = MultiWorldCamera(intrinsics, model)
    likelihoodf = ParallelPhysicsLikelihood(multiCamera, point_cloud, model)

    # Step 4: Insert Priors on Shapes
    objects.assign_priors(prior_json, NoDecayProposal, base_rng)

    print("Done")

    # :TODO Rewrite Smapling

    # Step 5: Run Importance Sampling

    

    return model, likelihoodf





default_camera = CameraIntrinsics(width=640, height=480, fov_degree=60,)

if __name__ == "__main__":

    scene_usd = "scene01/data/scene01_physics.usdc"
    priors = "scene01/data/scene01_priors.json"

    folder = "scene01/results"

    scene_makeup = Scene_Makeup(
        static=['dining_room_table'],
        observed=['coffee_0023', 'soap_dispenser_01'],
        hidden=['f10_apple_iphone_4'],
        )

    model, x = run_importance_sampling(scene_usd, priors, default_camera, scene_makeup, folder)
