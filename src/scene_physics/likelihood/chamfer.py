"""
Bidirectional Chamfer distance likelihood for point cloud comparison.

Drop-in alternative to the 3DP3 Gaussian mixture likelihood.
"""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit)
def chamfer_distance(pc1: jnp.ndarray, pc2: jnp.ndarray) -> float:
    """
    Bidirectional Chamfer distance between two point clouds.

    Filters NaN points (invalid depth pixels) before computing distances.
    Returns negative distance for use as a log-likelihood proxy
    (higher = better match).

    Args:
        pc1: Point cloud, shape (H, W, 3)
        pc2: Point cloud, shape (H, W, 3)

    Returns:
        Negative Chamfer distance (scalar)
    """
    # Flatten to (N, 3)
    pts1 = pc1.reshape(-1, 3)
    pts2 = pc2.reshape(-1, 3)

    # Mask out NaN points
    valid1 = ~jnp.any(jnp.isnan(pts1), axis=-1)
    valid2 = ~jnp.any(jnp.isnan(pts2), axis=-1)

    # Replace NaN with large sentinel so they don't affect min distances
    pts1_clean = jnp.where(valid1[:, None], pts1, 1e6)
    pts2_clean = jnp.where(valid2[:, None], pts2, 1e6)

    # Pairwise squared distances: (N1, N2)
    # Use chunked computation to avoid OOM on large point clouds
    # dist[i,j] = ||pts1[i] - pts2[j]||^2
    diff = pts1_clean[:, None, :] - pts2_clean[None, :, :]
    sq_dists = jnp.sum(diff ** 2, axis=-1)

    # For each point in pc1, find nearest in pc2
    min_dist_1to2 = jnp.min(sq_dists, axis=1)
    # For each point in pc2, find nearest in pc1
    min_dist_2to1 = jnp.min(sq_dists, axis=0)

    # Average over valid points only
    sum_1to2 = jnp.where(valid1, min_dist_1to2, 0.0).sum()
    sum_2to1 = jnp.where(valid2, min_dist_2to1, 0.0).sum()
    count1 = jnp.maximum(valid1.sum(), 1)
    count2 = jnp.maximum(valid2.sum(), 1)

    chamfer = sum_1to2 / count1 + sum_2to1 / count2

    # Return negative distance (higher = better match)
    return -chamfer


@functools.partial(jax.jit)
def chamfer_distance_batch(
    observed_xyz: jnp.ndarray,
    rendered_xyz_batch: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched Chamfer distance: one observed cloud vs N rendered clouds.

    Args:
        observed_xyz: Observed point cloud, shape (H, W, 3)
        rendered_xyz_batch: Batch of rendered point clouds, shape (N, H, W, 3)

    Returns:
        Negative Chamfer distances, shape (N,)
    """
    def _single(rendered_xyz):
        return chamfer_distance(observed_xyz, rendered_xyz)

    return jax.vmap(_single)(rendered_xyz_batch)
