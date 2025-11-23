import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def threedp3_likelihood(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance: float,
    outlier_prob: float,
    outlier_volume: float,
    filter_size: int,
):
    """
    Compute total log-likelihood over all pixels.
    """
    log_probabilities_per_pixel = threedp3_likelihood_per_pixel(
        observed_xyz,
        rendered_xyz,
        variance,
        outlier_prob,
        outlier_volume,
        filter_size,
    )
    return log_probabilities_per_pixel.sum()


def threedp3_likelihood_per_pixel(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance: float,
    outlier_prob: float,
    outlier_volume: float,
    filter_size: int,
):
    """
    Compute per-pixel log-likelihoods.
    """
    # Pad rendered_xyz so we can take (2*filter_size+1) x (2*filter_size+1) patches
    rendered_xyz_padded = jax.lax.pad(
        rendered_xyz,
        padding_value=-100.0,
        padding_config=(
            (filter_size, filter_size, 0),  # height
            (filter_size, filter_size, 0),  # width
            (0, 0, 0),                      # channels
        ),
    )

    height, width = observed_xyz.shape[:2]

    # Build a list of (i, j) pixel coordinates, flattened
    ii, jj = jnp.meshgrid(
        jnp.arange(height),
        jnp.arange(width),
        indexing="ij",
    )
    indices = jnp.stack([ii.reshape(-1), jj.reshape(-1)], axis=-1)  # (N, 2)

    # Vectorize over all pixel indices
    log_probs_flat = jax.vmap(
        gaussian_mixture_logprob,
        in_axes=(0, None, None, None, None, None, None),
    )(
        indices,
        observed_xyz,
        rendered_xyz_padded,
        variance,
        outlier_prob,
        outlier_volume,
        filter_size,
    )

    # Reshape back to (H, W)
    return log_probs_flat.reshape(height, width)


def gaussian_mixture_logprob(
    ij: jnp.ndarray,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variance: float,
    outlier_prob: float,
    outlier_volume: float,
    filter_size: int,
):
    """
    Log-likelihood for a single pixel (i, j) under your mixture model.

    ij: shape (2,) with (i, j) coordinates.
    """
    i, j = ij[0], ij[1]

    # Observed 3D point at this pixel
    obs = observed_xyz[i, j, :3]

    # Extract the local (2*filter_size+1) x (2*filter_size+1) neighborhood in rendered_xyz
    patch = jax.lax.dynamic_slice(
        rendered_xyz_padded,
        (i, j, 0),
        (2 * filter_size + 1, 2 * filter_size + 1, 3),
    )

    # Distances between observed point and each candidate in the patch
    distances = obs - patch  # shape: (K, K, 3)

    # Gaussian log-probabilities for each candidate
    # (sum over XYZ, then normalize by number of pixels as in your original code)
    log_probs = (
        norm.logpdf(distances, loc=0.0, scale=jnp.sqrt(variance)).sum(-1)
        - jnp.log(observed_xyz.shape[0] * observed_xyz.shape[1])
    )

    # Mixture with outlier component: keep your original formula
    return jnp.logaddexp(
        log_probs.max() + jnp.log(1.0 - outlier_prob),
        jnp.log(outlier_prob) - jnp.log(outlier_volume),
    )
