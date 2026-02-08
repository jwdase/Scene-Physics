import numpy as np
import pyvista as pv

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
