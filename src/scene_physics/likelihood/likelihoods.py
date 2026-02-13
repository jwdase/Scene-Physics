"""
Standalone point cloud likelihood functions.

Extracted from b3d to remove GenJax dependency.
These are pure JAX functions for computing 3DP3-style Gaussian mixture
likelihoods between observed and rendered point clouds.
"""

import functools

import jax
import jax.numpy as jnp

# The decorator allows the function to be deployed in parrallel
# on every single pixel
@functools.partial(
    jnp.vectorize,
    signature="(m)->()",
    excluded=(1, 2, 3, 4, 5, 6),
)
def _gaussian_mixture_vectorize(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variance: float,
    outlier_prob: float,
    outlier_volume: float,
    filter_size: int,
):
    """
    Compute Gaussian mixture likelihood for a single pixel.

    For each observed pixel, computes distances to nearby rendered pixels
    within a filter window and returns the max log probability as inlier score.
    """

    # Queries what is the distance between our observed point and 
    # the 49 points nearby defined by the filter. These are our distances
    # note this is an approximation WE ARE NOT LOOKING for closest point 
    # in 3D space. That would have a different result.
    distances = observed_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(
        rendered_xyz_padded,
        (ij[0], ij[1], 0),
        (2 * filter_size + 1, 2 * filter_size + 1, 3),
    )

    # For each distance, computes the log probability which is:  
    #       log P(dx) = -dx²/(2σ²) - log(√(2πσ²)) 
    # Then multiplies by weight - for each rendered component that is
    #       (1 / (H * W)) or - log(H * W)
    probabilities = jax.scipy.stats.norm.logpdf(
        distances, loc=0.0, scale=jnp.sqrt(variance)
    ).sum(-1) - jnp.log(observed_xyz.shape[0] * observed_xyz.shape[1])

    # Inlier score: "If this point is an inlier, how well does the best rendered
    # point explain it?
    inlier_score = probabilities.max() + jnp.log(1.0 - outlier_prob)

    # Outlier score: "If this point is noise, it could be anywehere in the volume
    outlier_score = jnp.log(outlier_prob) - jnp.log(outlier_volume)

    # Intuition on why inlier and outlier: "We could have a very low inlier score -1000
    # which when raised to e -> 0. Thus we need a way to balance it. To do that, we add
    # the noice prob which will dominate. Creating out probability score. 
    return {
        "pix_score": jnp.logaddexp(inlier_score, outlier_score),
        "inlier_score": inlier_score,
        "outlier_score": outlier_score,
    }


def threedp3_likelihood_per_pixel(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance: float = 0.001,
    outlier_prob: float = 0.001,
    outlier_volume: float = 1.0,
    filter_size: int = 3,
):
    """
    Compute 3DP3-style Gaussian mixture likelihood between two point clouds.

    For each pixel in the observed point cloud, computes a mixture likelihood:
    - Inlier component: Gaussian centered on nearby rendered pixels (within filter window)
    - Outlier component: Uniform distribution over outlier_volume

    Args:
        observed_xyz: Observed point cloud, shape (H, W, 3)
        rendered_xyz: Rendered point cloud, shape (H, W, 3)
        variance: Gaussian variance for inlier likelihood
        outlier_prob: Prior probability of a pixel being an outlier
        outlier_volume: Volume of the uniform outlier distribution
        filter_size: Half-size of the filter window (full window is 2*filter_size+1)

    Returns:
        Dictionary with:
        - "pix_score": Per-pixel log likelihood, shape (H, W)
        - "inlier_score": Per-pixel inlier log score, shape (H, W)
        - "outlier_score": Per-pixel outlier log score, shape (H, W)

    Point Cloud:
        - Shape: (H, W, 3)
        - The point cloud takes in index (x, y) and returns (X, Y, Z) which is 3D world
          coordinates
        - The indexing comes from observed pixel in depth camera -> point cloud conversion
          turns it into the new shape

    Example:
        >>> observed = jax.random.uniform(key, (64, 64, 3))
        >>> rendered = observed + 0.01 * jax.random.normal(key2, (64, 64, 3))
        >>> result = threedp3_likelihood_per_pixel(observed, rendered)
        >>> total_score = result["pix_score"].sum()
    """

    # Replace NaN in rendered point cloud with far-away sentinel so those pixels
    # get near-zero Gaussian probability instead of propagating NaN through the
    # filter window and corrupting valid observed pixel scores.
    rendered_xyz = jnp.where(jnp.isnan(rendered_xyz), -100.0, rendered_xyz)

    # Adds padding for comparison (H, W, 3)
    # +; - dimension for height and width, but not for 3rd dim
    rendered_xyz_padded = jax.lax.pad(
        rendered_xyz,
        -100.0,
        (
            (filter_size, filter_size, 0),
            (filter_size, filter_size, 0),
            (0, 0, 0),
        ),
    )

    # Creates all pixel coordinates so the vectorized function
    # can work in parrallel - querying each point in OBSERVED
    # and asking what the score is (H, W)
    jj, ii = jnp.meshgrid(
        jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0])
    )
    indices = jnp.stack([ii, jj], axis=-1)


    log_probabilities = _gaussian_mixture_vectorize(
        indices,
        observed_xyz,
        rendered_xyz_padded,
        variance,
        outlier_prob,
        outlier_volume,
        filter_size,
    )
    return log_probabilities


def compute_likelihood_score(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance: float = 0.001,
    outlier_prob: float = 0.001,
    outlier_volume: float = 1.0,
    filter_size: int = 3,
):
    """
    Convenience function to compute total log likelihood score.

    Filters out NaN values before summing (common with invalid depth pixels).

    Args:
        observed_xyz: Observed point cloud, shape (H, W, 3)
        rendered_xyz: Rendered point cloud, shape (H, W, 3)
        variance: Gaussian variance for inlier likelihood
        outlier_prob: Prior probability of a pixel being an outlier
        outlier_volume: Volume of the uniform outlier distribution
        filter_size: Half-size of the filter window

    Returns:
        Total log likelihood score (higher is better match)
    """
    result = threedp3_likelihood_per_pixel(
        observed_xyz, rendered_xyz, variance, outlier_prob, outlier_volume, filter_size
    )
    pix_score = result["pix_score"]
    # Filter NaN values (from invalid depth pixels)
    valid_scores = jnp.where(jnp.isnan(pix_score), 0.0, pix_score)
    return valid_scores.sum()


class Likelihood:
    def __init__(self, initial_point_cloud):
        self.correct_pointcloud = initial_point_cloud
        self.baseline_score = self._compute_baseline()

    def _compute_baseline(self):
        """Compute the self-comparison score used as normalization baseline."""
        return compute_likelihood_score(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=self.correct_pointcloud,
        )

    def new_proposal_likelihood(self, proposal):
        """Returns log-ratio of proposal likelihood to baseline."""
        new_score = compute_likelihood_score(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=proposal,
        )
        return new_score - self.baseline_score


