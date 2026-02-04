# Warp GPU Rendering for Faster Sampling

## Problem

The current sampling pipeline uses PyVista (CPU-based VTK) for rendering depth images. Each MCMC iteration requires a full CPU render, which is the primary bottleneck.

**Current flow:**
```
PyVista (VTK-based, CPU)
    ↓
pv.Plotter.get_image_depth()  ← CPU rasterization
    ↓
Depth buffer (numpy array)
    ↓
Unproject to point cloud
```

## Solution

Newton already includes a GPU ray-traced renderer via Warp: `SensorTiledCamera`.

**Location:** `src/newton/newton/_src/sensors/sensor_tiled_camera.py`

### Key Advantages

| Feature | PyVista | Newton/Warp |
|---------|---------|-------------|
| Execution | CPU | **GPU (CUDA)** |
| Method | Rasterization | Ray tracing with BVH |
| Batch rendering | No | **Yes** (multiple cameras/worlds) |
| Integration | Separate from physics | **Same Model/State objects** |

### Expected Speedup

- **Single render**: ~10-50x faster
- **Batch rendering**: Can evaluate multiple proposals in one GPU call

---

## Implementation Guide

### 1. Basic Usage

```python
from newton._src.sensors import SensorTiledCamera
import warp as wp
import numpy as np

# Assuming you have a Newton Model and State
sensor = SensorTiledCamera(
    model=model,
    num_cameras=1,
    width=640,
    height=480
)

# Create output buffer (on GPU)
depth_output = sensor.create_depth_image_output()

# Compute camera rays (pinhole model)
fov_radians = np.radians(60.0)  # vertical FOV
camera_rays = sensor.compute_pinhole_camera_rays(fov_radians)

# Camera transform: (num_cameras, num_worlds) array of wp.transformf
# transformf = (position, quaternion) where quaternion is (x, y, z, w)
camera_transforms = wp.array(
    [[wp.transformf(pos, quat)]],  # shape: (1, 1) for single camera, single world
    dtype=wp.transformf
)

# Render
sensor.render(
    state=state,
    camera_transforms=camera_transforms,
    camera_rays=camera_rays,
    depth_image=depth_output
)

# Get result as numpy
depth = depth_output.numpy()  # shape: (num_worlds, num_cameras, width*height)
depth = depth.reshape(height, width)
```

### 2. Camera Transform Setup

The camera transform uses `wp.transformf` which is `(translation, quaternion)`.

To match your current PyVista camera:
```python
def pyvista_camera_to_warp_transform(camera_position):
    """
    Convert PyVista camera position format to Warp transform.

    PyVista format: [(eye_x, eye_y, eye_z), (focal_x, focal_y, focal_z), (up_x, up_y, up_z)]
    """
    eye = np.array(camera_position[0])
    focal = np.array(camera_position[1])
    up = np.array(camera_position[2])

    # Compute camera basis vectors
    forward = focal - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up_corrected = np.cross(right, forward)

    # Build rotation matrix (camera looks down -Z in its local frame)
    # Columns are right, up, -forward
    R = np.column_stack([right, up_corrected, -forward])

    # Convert to quaternion (scipy uses xyzw format, same as Warp)
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(R).as_quat()  # xyzw

    return wp.transformf(
        wp.vec3f(eye[0], eye[1], eye[2]),
        wp.quatf(quat[0], quat[1], quat[2], quat[3])
    )
```

### 3. Integration with MHSampler

Modify `simulation/sampling.py`:

```python
class WarpMHSampler(MHSampler):
    def __init__(self, model, state, camera_position, width=640, height=480, var=0.2):
        self.model = model
        self.state = state
        self.var = var

        # Setup Warp sensor (once)
        self.sensor = SensorTiledCamera(model, num_cameras=1, width=width, height=height)
        self.depth_output = self.sensor.create_depth_image_output()

        fov_rad = np.radians(60.0)
        self.camera_rays = self.sensor.compute_pinhole_camera_rays(fov_rad)
        self.camera_transforms = self._setup_camera(camera_position)

        self.width = width
        self.height = height

    def render_depth(self):
        """GPU-accelerated depth rendering."""
        self.sensor.render(
            state=self.state,
            camera_transforms=self.camera_transforms,
            camera_rays=self.camera_rays,
            depth_image=self.depth_output
        )
        return self.depth_output.numpy().reshape(self.height, self.width)

    def update_position(self, body_idx, x, y, z):
        """Update object position in Newton state."""
        # Update body_q for the specified body
        body_q = self.state.body_q.numpy()
        body_q[body_idx, 0] = x  # position x
        body_q[body_idx, 1] = y  # position y
        body_q[body_idx, 2] = z  # position z
        self.state.body_q = wp.array(body_q, dtype=wp.transformf)
```

### 4. Batch Rendering (Advanced)

For even more speedup, render multiple proposals at once using Newton's multi-world feature:

```python
# Create model with multiple "worlds" (parallel scene configurations)
num_proposals = 16
model = ModelBuilder(num_worlds=num_proposals)

# ... build scene ...

sensor = SensorTiledCamera(model, num_cameras=1, width=W, height=H)

# Update each world with different object positions
# Then render ALL in one GPU call
sensor.render(state, camera_transforms, camera_rays, depth_image=depth_output)

# depth_output shape: (num_proposals, 1, W*H)
# Compute likelihoods for all proposals in parallel
```

---

## Files to Modify

1. **`visualization/scene.py`** - Add `WarpVisualizer` class alongside `PyVistaVisualizer`
2. **`simulation/sampling.py`** - Use Warp rendering in `MHSampler`
3. **`intrinsics.py`** - May need to adjust unprojection for Warp's depth format

## Key References

- `src/newton/newton/_src/sensors/sensor_tiled_camera.py` - Main sensor class
- `src/newton/newton/_src/sensors/warp_raytrace/ray_cast.py` - Ray intersection kernels
- `src/newton/newton/_src/sensors/warp_raytrace/render.py` - Render kernel

## Notes

- Warp depth is distance along ray, not Z-buffer depth
- First render may be slow due to Warp kernel compilation (subsequent renders are fast)
- Ensure CUDA is available: `wp.is_cuda_available()`
