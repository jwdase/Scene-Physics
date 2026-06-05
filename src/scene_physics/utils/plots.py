import json

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import newton
import jax
import jax.numpy as jnp

from scene_physics.properties.structs import Object_Collection
from scene_physics.configs.camera import CameraIntrinsics
from scene_physics.visualization.camera import SingleWorldCamera
from scene_physics.data_gen.usd_repose import repose_usd
from scene_physics.properties.shapes import *

# Plot point cloud
def plot_point_maps(point_cloud, location):
    pts = np.array(point_cloud).reshape(-1, 3)
    pts[:,2] = -pts[:,2]
    pc = pv.PolyData(pts)
    pc.plot(
        point_size=5,
        style="points",
        screenshot=location,
    )

# Export point cloud as a .ply
def save_point_cloud_ply(point_cloud, location):
    pts = np.array(point_cloud).reshape(-1, 3)
    valid_mask = (pts[:, 2] > -10) & (pts[:, 2] < 10)
    pts_valid = pts[valid_mask]
    pc_ply = pv.PolyData(pts_valid)
    pc_ply.save(location)

    print(f"Saved to: {location}")


def render_bio(depth_image, height, width, debug=False):
    """
    Prints information about a render to the screen,
    and returns shape of depth_image
    """
    depth_np = depth_image.numpy()
    depth_2d = depth_np[0, 0].reshape(height, width)

    if debug:
        print(f"Depth shape: {depth_2d.shape}")
        print(f"Depth range: {depth_2d[depth_2d > 0].min():.2f} - {depth_2d.max():.2f} m")

    return depth_2d


def plot_target_scene(truth_json : str, save_dir : str):
    with open(truth_json, 'r') as f:
        truth = json.load(f)

    labels = list(truth.keys())
    positions = [truth[obj][:3] for obj in labels]

    for lab, pos in zip(labels, positions):
        plt.scatter([pos[0]], [pos[1]], label=lab, alpha=0.6)


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

def build_final_world(scene_usd, object_collection : Object_Collection, save_dir : str):
    # NOTE: Location update only for hidden objects

    new_positions = {
        obj.name : obj.get_final_location() 
        for obj in object_collection.objects.values() 
        if isinstance(obj, Hidden)
    }

    out_path = f"{save_dir}/final_world.usdc"

    repose_usd(scene_usd, new_positions, out_path)
