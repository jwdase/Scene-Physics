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
