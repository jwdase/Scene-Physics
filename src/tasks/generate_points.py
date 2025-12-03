import sys
sys.path.append("/orcd/home/002/jacktuck/Scene-Physics/b3d/src")

import pyvista as pv
import numpy as np
import jax.numpy as jnp


# Import B3D libraries
from b3d.chisight.dense.likelihoods.image_likelihoods import threedp3_likelihood_per_pixel_old
from b3d.camera import unproject_depth, Intrinsics
import b3d

# ---------------------------------------------
# Create scene and render
# ---------------------------------------------
plotter = pv.Plotter(off_screen=True)
# plotter.add_axes()

# plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
# plotter.add_mesh(plane, color="lightgray")
plotter.add_mesh(pv.Sphere(center=(0, 1, 0)))


plotter.screenshot("photo.png")
zval = plotter.get_image_depth()

# ---------------------------------------------
# Compute intrinsics from an existing plotter
# ---------------------------------------------
def get_camera_intrinsics(plotter):
    """
    Return camera intrinsics from a pyvista plotter.
    """
    camera = plotter.camera
    near, far = camera.clipping_range
    width, height = plotter.render_window.GetSize()

    # Vertical field of view (degrees → radians)
    fov_deg = camera.view_angle
    fov_rad = np.radians(fov_deg)

    # Standard pinhole intrinsics
    fy = height / (2 * np.tan(fov_rad / 2))
    fx = fy * (width / height)

    cx = width / 2
    cy = height / 2

    return {
        "height": height,
        "width": width,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "near": near,
        "far": far,
    }


camera_data = get_camera_intrinsics(plotter)

intrinsic = Intrinsics(
    height=camera_data["height"],
    width=camera_data["width"],
    fx=camera_data["fx"],
    fy=camera_data["fy"],
    cx=camera_data["cx"],
    cy=camera_data["cy"],
    near=camera_data["near"],
    far=camera_data["far"],
)

# ---------------------------------------------
# Unproject depth → point cloud
# ---------------------------------------------
# ensure numpy float32 for depth
zval = np.array(zval, dtype=np.float32)
zlinear = -zval
point_cloud = unproject_depth(zlinear, intrinsic)

# ---------------------------------------------
# Helper for visualizing point clouds
# ---------------------------------------------
def plot_point_maps(point_cloud, location):
    pts = np.array(point_cloud).reshape(-1, 3)
    pts[:,2] = -pts[:,2]
    pc = pv.PolyData(pts)
    pc.save("point_cloud.ply")
    pc.plot(
        point_size=5,
        style="points",
        screenshot=location,
        # cpos=cam_location,
    )

plot_point_maps(point_cloud, "cloud1.png")

# Remove all nan
point_cloud = np.array(point_cloud)
point_cloud[np.isnan(point_cloud)] = 1_000
point_cloud = jnp.array(point_cloud)

# ---------------------------------------------
# Test similarity scores with noise
# ---------------------------------------------
noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1]

print("Testing similarity scores with white noise:")
print("-" * 70)

scores = []
for noise_std in noise_levels:
    # Add white noise to point cloud
    noise = np.random.normal(0, noise_std, point_cloud.shape)
    noisy_point_cloud = point_cloud + noise
    
    # Compute per-pixel likelihoods
    image_likelihoods = threedp3_likelihood_per_pixel_old(
        observed_xyz=noisy_point_cloud,
        rendered_xyz=point_cloud,
        variance=0.001,
        outlier_prob=0.001,
        outlier_volume=1.0,
        filter_size=3,
    )
    
    likelihood = image_likelihoods["pix_score"]
    clean = np.array([arr[~np.isnan(arr)] for arr in likelihood])
    score = clean.sum()
    scores.append(score)

baseline_score = scores[0]

for noise_std, score in zip(noise_levels, scores):
    pct_change = ((score - baseline_score) / abs(baseline_score)) * 100
    print(f"Noise std={noise_std:6.3f} -> Score: {score:12.2f} | Change: {pct_change:+6.2f}%")
