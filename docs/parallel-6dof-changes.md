# Multi-Body 6DOF Parallel Sampling — Changes Summary

Extends Scene-Physics from single-object (x, z) sequential sampling to multi-body 6DOF parallel proposal evaluation using Newton's multi-world GPU capability.

---

## Modified Files

### `src/scene_physics/properties/shapes.py`

Added two methods to the `Body` class:

- **`move_6dof_wp(state, position, quat)`** — Full 6DOF state update. Accepts `(x, y, z)` position and `(qx, qy, qz, qw)` quaternion. Same GPU round-trip pattern as existing `move_position_wp()` but without hardcoding `y=0`.

- **`move_position_in_array(body_q_np, position, quat)`** — Writes position/rotation directly into a numpy `body_q` array at this body's index. Used for parallel world updates where you modify multiple bodies in a single numpy array before one CPU→GPU transfer.

Existing `move_position_wp()` is unchanged for backward compatibility.

### `src/scene_physics/kernels/image_process.py`

Added **`render_point_clouds_batch()`** — Same render pipeline as `render_point_cloud()` but preserves the world dimension instead of squeezing it. Returns `(num_worlds, H, W, 3)` JAX array via DLPack zero-copy transfer.

Existing `render_point_cloud()` is unchanged.

### `src/scene_physics/likelihood/likelihoods.py`

- **`compute_likelihood_score_batch(observed_xyz, rendered_xyz_batch)`** — Computes 3DP3 likelihood for N rendered point clouds against one observed cloud using `jax.vmap`. Returns `(N,)` array of log-likelihoods.

- **`Likelihood.new_proposal_likelihood_batch(proposals_batch)`** — Batch version of `new_proposal_likelihood()`. Accepts `(N, H, W, 3)` and returns `(N,)` log-ratios relative to baseline.

Existing `compute_likelihood_score()` and `Likelihood` class are unchanged.

### `src/scene_physics/likelihood/likelihoods_physics.py`

Added **`Likelihood_Physics_Parallel`** class:

- Constructor takes `num_worlds` and sets up batch rendering buffers
- **`new_proposal_likelihood_batch(scene)`** — Runs forward physics (single `solver.step()` loop processes all worlds simultaneously), batch renders at eval points, batch computes likelihoods via `compute_likelihood_score_batch`. Returns `(num_worlds,)` scores.
- Physics states are pre-allocated and reused across iterations (same pattern as `Likelihood_Physics`)

Existing `Likelihood_Physics` class is unchanged.

---

## New Files

### `src/scene_physics/simulation/parallel_builder.py`

**`build_parallel_worlds(base_builder_fn, num_worlds)`**

Replicates a scene into N isolated parallel worlds:
- Ground plane is shared across all worlds (`current_world = -1`)
- Each world's objects are added via `add_builder(world_builder, world=i)`
- Returns `body_index_map`: a dict mapping `(world_idx, body_name) -> body_q_index` so proposals can target the correct entries in the combined state array

### `src/scene_physics/likelihood/chamfer.py`

Bidirectional Chamfer distance in JAX:
- **`chamfer_distance(pc1, pc2)`** — Returns negative Chamfer distance (higher = better match). Filters NaN points from invalid depth pixels.
- **`chamfer_distance_batch(observed_xyz, rendered_xyz_batch)`** — Batched via `jax.vmap`. Drop-in alternative to `compute_likelihood_score_batch`.

### `src/scene_physics/sampling/proposals.py`

**`SixDOFProposal`** class:
- Position proposals: Gaussian random walk on `(x, y, z)`
- Rotation proposals: Small axis-angle perturbation (via scipy `Rotation`) composed with current quaternion
- **`propose_batch(current_pos, current_quat, num_proposals)`** — Generates `num_worlds` proposals at once
- Variance scheduling via pluggable `schedule` callable. Includes `linear_decay` and `exponential_decay`

### `src/scene_physics/sampling/parallel_mh.py`

**`ParallelPhysicsMHSampler`** class:

Core sampling loop (sequential object placement):
1. For each body in `placement_order`:
   - Generate `num_worlds` 6DOF proposals
   - Write each proposal into its world's `body_q` (single numpy array, one CPU→GPU transfer)
   - Run forward physics (all worlds processed by one `solver.step()` loop)
   - Batch render all worlds
   - Batch compute likelihoods
   - Greedy best-of-N selection (pick highest scoring proposal)
   - Freeze accepted position, move to next body

### `src/scene_physics/simulation/run_parallel_sampling.py`

End-to-end experiment script:
- Builds Scene01 across 16 parallel worlds
- Configures 6DOF proposals with linear decay scheduling
- Runs sequential placement of bowl and coffee (table is static)
- Saves per-body score histories to `recordings/parallel_6dof/`

---

## Data Flow

```
Proposals (CPU, numpy)
    │
    ▼  [one CPU→GPU transfer per iteration]
state.body_q (GPU, Warp)
    │
    ▼  [solver.step() — all worlds in parallel]
Physics simulation (GPU, Warp)
    │
    ▼  [sensor.render() + depth_to_point_cloud kernel]
Point clouds (GPU, Warp)
    │
    ▼  [DLPack — zero-copy pointer sharing]
Point clouds (GPU, JAX)
    │
    ▼  [vmap'd likelihood — all worlds in parallel]
Scores (GPU, JAX) → numpy for best-of-N selection
```

The Warp simulation stays alive on GPU throughout. The only CPU round-trip per iteration is writing proposals into `body_q`.

---

## Configuration

Key parameters in `run_parallel_sampling.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `NUM_WORLDS` | 16 | Scale up on H100 (32, 64, ...) |
| `ITERATIONS_PER_OBJECT` | 100 | MH iterations per body |
| `POS_STD` | 0.05 | Position proposal std (meters) |
| `ROT_STD` | 0.1 | Rotation proposal std (radians) |
| Schedule | `linear_decay` | Anneals std from 1.0x to 0.1x |
