# Bug 2: `camera_transforms` Out-of-Bounds — Garbage Point Clouds for All Worlds Except World 0

## Status
**Open** — not yet fixed.

## Symptom
Objects are placed at completely wrong locations. The sampler appears to wander randomly rather than converging toward the target position.

## Root Cause

### Where it happens
`visualization/camera.py:16–20` and `kernels/image_process.py:35`

### What goes wrong

`setup_depth_camera` creates the camera transform array as a `(1, 1)` Warp array — one camera, one world:

```python
camera_transforms = wp.array(
    [[camera_transform]],
    dtype=wp.transformf,
    ndim=2,
)
# shape: (num_cameras=1, num_worlds=1)
```

The `depth_to_point_cloud` kernel is launched with dimensions `(num_worlds, num_cameras, num_pixels)` and indexes into this array using both `cam_idx` and `world_idx`:

```python
world_idx, cam_idx, pixel_idx = wp.tid()
...
point_world = wp.transform_point(camera_transforms[cam_idx, world_idx], point_camera)
```

With `NUM_WORLDS=20`, `world_idx` runs from `0` to `19`. `camera_transforms[0, 0]` is valid; `camera_transforms[0, 1..19]` are **out of bounds** on the `(1, 1)` array.

In CUDA, out-of-bounds reads on a GPU array return adjacent device memory — likely zeros or garbage values from whatever was allocated nearby. The depth→world-space transform for worlds 1..19 is therefore corrupted. Their rendered point clouds are meaningless.

### Downstream effect

`new_proposal_likelihood_still_batch` computes scores for all `num_worlds` proposals by comparing each world's point cloud against the target. Since worlds 1..19 produce garbage point clouds, their scores are large negative noise unrelated to object position.

The rank-based softmax selection in `_rank_proposals` treats these garbage scores as real signal. With one genuinely-scoring world (world 0) and 19 garbage worlds, the sampler effectively degrades to a **1-world search**. The `NUM_WORLDS` parallelism provides no benefit. Convergence is slow and highly dependent on the random walk from world 0's initial position.

## Diagnostic

The `print(scores)` statement on `proposals.py:97` already prints every iteration's score vector. If Bug 2 is active, you will see:

- One score that varies meaningfully across iterations (world 0)
- The remaining scores clustered at large, similar negative values that don't change much as objects move

## Fix (not yet implemented)

Broadcast the single camera transform to cover all worlds. Since every world uses the same physical camera, the transform should be identical across the world dimension:

```python
# In setup_depth_camera, after building camera_transform:
camera_transforms = wp.array(
    [[camera_transform] * num_worlds],  # shape: (1, num_worlds)
    dtype=wp.transformf,
    ndim=2,
)
```

This requires `setup_depth_camera` to accept `num_worlds` as a parameter, which it currently does not.

Alternatively, change the kernel to always clamp `world_idx` to 0 for the camera lookup (since all worlds share one camera):

```python
point_world = wp.transform_point(camera_transforms[cam_idx, 0], point_camera)
```

This is a simpler one-line fix but hardcodes the assumption that there is exactly one camera shared across all worlds.

## Files to Change

| File | Change needed |
|------|---------------|
| `visualization/camera.py` | Pass `num_worlds` and build `(1, num_worlds)` transform array |
| OR `kernels/image_process.py` | Clamp `world_idx` to `0` when indexing `camera_transforms` |
| `likelihood/likelihoods_physics.py` | Update `setup_depth_camera` call to pass `num_worlds` (if taking first approach) |
