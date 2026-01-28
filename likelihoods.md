# Likelihood Function

This module implements a **3DP3-style Gaussian mixture likelihood** for comparing observed and rendered point clouds.

## Core Concept

For each pixel in the observed point cloud, the likelihood is computed as a **mixture of two components**:

1. **Inlier component**: A Gaussian distribution centered on nearby rendered pixels
2. **Outlier component**: A uniform distribution over a specified volume

## The Algorithm (`threedp3_likelihood_per_pixel`)

**Step 1: Pad the rendered point cloud** (`likelihoods.py:88-96`)
- The rendered point cloud is padded with -100.0 values around the edges to allow the filter window to operate on boundary pixels

**Step 2: For each observed pixel** (`_gaussian_mixture_vectorize`, lines 20-50):

1. **Compute distances** to all rendered pixels within a local window of size `(2*filter_size+1) x (2*filter_size+1)` centered on the same pixel location

2. **Compute inlier probability** using a 3D Gaussian:
   ```
   log P(obs | inlier) = sum of log-normal PDFs over x,y,z
   ```
   Takes the **max** probability over the filter window (best match)

3. **Compute outlier probability**:
   ```
   log P(obs | outlier) = log(outlier_prob) - log(outlier_volume)
   ```

4. **Combine via log-sum-exp**:
   ```
   pix_score = logaddexp(inlier_score, outlier_score)
   ```

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `variance` | 0.001 | Controls how tight the Gaussian match must be |
| `outlier_prob` | 0.001 | Prior probability a pixel is an outlier (noise/occlusion) |
| `outlier_volume` | 1.0 | Volume of the uniform outlier distribution |
| `filter_size` | 3 | Half-width of the local search window (7x7 default) |

## Why This Approach?

- **Robustness**: The outlier component handles noise, occlusions, and misalignments gracefully
- **Local matching**: The filter window allows small spatial misalignments between observed and rendered
- **Probabilistic**: Returns a proper log-likelihood that can be used in MCMC or other inference algorithms

The total scene likelihood is simply the sum of per-pixel log scores (`compute_likelihood_score`).
