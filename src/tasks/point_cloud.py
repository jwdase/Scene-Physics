import sys
sys.path.append("/orcd/home/002/jwdase/project/Scene-Physics/b3d/src")


import pyvista as pv
import numpy as np
import 

from b3d.chisight.dense.likelihoods.image_likelihoods import threedp3_likelihood_per_pixel_old

# ------------------------------
# 1. Render scene off-screen
# ------------------------------
plotter = pv.Plotter(off_screen=True)
sphere = pv.Box()
plotter.add_mesh(sphere)

plotter.add_axes()
plotter.set_background("white")
plotter.camera_position = cam_location


# Render window size
w, h = 1024, 768

# Render and extract depth + rgb
rgb = plotter.screenshot('file2.png')
depth = plotter.get_image_depth()   # float32 depth buffer

# ------------------------------
# 2. Convert normalized depth buffer → real distances
# ------------------------------
# PyVista returns depth in [0, 1] range using near/far planes.
near = plotter.camera.clipping_range[0]
far  = plotter.camera.clipping_range[1]

# Convert normalized depth to linear depth
Z = near * far / (far - depth * (far - near))

# ------------------------------
# 3. Camera intrinsics from PyVista camera FOV
# ------------------------------
fov = plotter.camera.view_angle  # vertical FOV in degrees

fx = (w / 2) / np.tan(np.deg2rad(fov / 2))
fy = (h / 2) / np.tan(np.deg2rad(fov / 2))
cx = w / 2
cy = h / 2

# ------------------------------
# 4. Build meshgrid of pixel coordinates
# ------------------------------
u, v = np.meshgrid(np.arange(w), np.arange(h))

# ------------------------------
# 5. Apply pinhole projection to get point cloud
# ------------------------------
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy  # Note: might flip sign depending on convention
point_cloud = np.stack((X, Y, Z), axis=-1)
points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

# ------------------------------
# 6. Add RGB colors if desired
# ------------------------------
rgb_pts = rgb.reshape(-1, 3) / 255.0

# ------------------------------
# 7. Create PyVista point cloud
# ------------------------------
pc = pv.PolyData(points)
pc["rgb"] = rgb_pts

plotter = pv.Plotter(off_screen=True)
plotter.add_points(pc, scalars="rgb", rgb=True, point_size=2)
plotter.screenshot('file3.png')

# Remove all nan
point_cloud = np.array(point_cloud)
point_cloud[np.isnan(point_cloud)] = 100

image_likelihoods = threedp3_likelihood_per_pixel_old(
    observed_xyz=point_cloud,
    rendered_xyz=point_cloud,
    variance=50.,
    outlier_prob=0.001,
    outlier_volume=1.0,
    filter_size=3,
)

val = image_likelihoods['pix_score']