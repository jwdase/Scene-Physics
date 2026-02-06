import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

def look_at_transform(eye, target, up=np.array([0., 1., 0.])):
    """Creates a wp.transformf that places the camera at 'eye' looking at 'target'.

    Newton camera convention: looks down -Z, +Y is up, +X is right.
    The rotation matrix columns are the camera basis vectors in world space.
    """
    # Forward direction (where camera looks)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Right vector (camera +X)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute up to ensure orthogonality (camera +Y)
    camera_up = np.cross(right, forward)

    # Build rotation matrix with columns as basis vectors:
    # Column 0: camera +X → right
    # Column 1: camera +Y → camera_up
    # Column 2: camera +Z → -forward (camera looks down -Z)
    rot_matrix = np.column_stack([right, camera_up, -forward])
    rot = Rotation.from_matrix(rot_matrix)
    quat = rot.as_quat()  # xyzw format

    return wp.transform(
            wp.vec3(eye[0], eye[1], eye[2]),
            wp.quat(quat[0], quat[1], quat[2], quat[3])
    )
