# Acceleration Proposals for MH Sampling Pipeline

**Target**: H100 80GB HBM3, CUDA 12.9, Scene-Physics MH sampling
**Current pipeline**: Propose position → 50 frames forward physics (saving snapshots every `eval_every` frames) → render point clouds on saved states → JAX likelihood on saved states → aggregate → accept/reject

---

## High Impact

### 1. Reduce XPBD Solver Iterations (100 → 4) -- DONE

**File**: `likelihoods_physics.py`
**Change**: `iterations=4` in `SolverXPBD`. Previously 100 — typical XPBD uses 2-10, Newton benchmarks use 2. Was running 5,000 constraint-solve iterations per proposal, now 200.

### 2. Pre-allocate Physics States -- DONE

**File**: `likelihoods_physics.py`
**Change**: `self._state_0` and `self._state_1` allocated once in `__init__`, reused via `.assign()` each proposal. Also pre-allocates `self._eval_states` for snapshot storage. Eliminates hundreds of GPU allocations per sampling run.

### 3. Separate Physics from Evaluation with `eval_every` -- DONE

**File**: `likelihoods_physics.py`
**Change**: Restructured `new_proposal_likelihood` into two phases:
- **Phase 1**: Run all 50 physics frames uninterrupted, saving state snapshots every `eval_every` frames (default 5) into pre-allocated `_eval_states`.
- **Phase 2**: Render + compute likelihood only on the 10 saved states.

This reduces render+likelihood calls from 50 → 10 per proposal (5x reduction on the most expensive per-frame operations) and keeps the physics loop tighter without JAX/rendering context switches. Also positions the physics loop for future CUDA graph capture.

### 4. Pre-compute JAX Meshgrid Indices -- DONE

**File**: `likelihoods.py`
**Change**: Added `_get_pixel_indices()` with a module-level cache keyed by `(H, W)`. The (480, 640, 2) index grid is computed once on first call and reused for all subsequent likelihood evaluations. Previously recreated on every call.

### 5. JIT-Compile the Likelihood Function -- DONE

**File**: `likelihoods.py`
**Change**: Added `@jax.jit` with `static_argnames` for scalar parameters to `compute_likelihood_score`. First call traces and compiles the XLA graph; all subsequent calls with same shapes/scalars replay the compiled kernel. Fuses NaN handling, padding, vectorized likelihood, and reduction into a single optimized GPU program.

### 6. GPU-Direct Point Cloud Transfer (Warp → JAX) -- DONE

**File**: `kernels/image_process.py`
**Change**: Replaced `points_gpu.numpy()` (GPU→CPU) + `jnp.array()` (CPU→GPU) with `jnp.from_dlpack(points_gpu)` for zero-copy GPU-to-GPU transfer via DLPack. Eliminates a round-trip per render call.

---

## Medium Impact

### 7. CUDA Graph Capture for Physics Loop

**File**: `likelihoods_physics.py`
**Current**: Phase 1 physics loop still launches individual kernels per frame (clear_forces → collide → solver.step). Newton supports `wp.ScopedCapture()` / `wp.capture_launch()` for CUDA graphs.
**Blocker**: `model.collide()` may allocate memory dynamically, which is incompatible with graph capture. Newton's own benchmarks pre-compute contacts outside the graph. Would need to either pre-compute contacts or verify `collide()` is graph-safe for this scene.
**Estimated speedup**: 5-10x on physics if achievable.

### 8. Replace `jnp.vectorize` with `jax.vmap`

**File**: `likelihoods.py`
**Current**: `_gaussian_mixture_vectorize` uses `jnp.vectorize(signature="(m)->()")` which traces implicitly over 307,200 pixels.
**Fix**: Rewrite using `jax.vmap` or a batched implementation for explicit vectorization.
**Estimated speedup**: 2-5x on the per-pixel computation.

### 9. Eliminate GPU↔CPU Round-trip in `move_position_wp`

**File**: `properties/shapes.py`
**Current**: `move_position_wp` does `state.body_q.numpy()` (GPU→CPU), modifies one element, then `wp.array(body_q, ...)` (CPU→GPU). Two transfers per proposal.
**Fix**: Use a Warp kernel to update the single body transform in-place on GPU, or compute on CPU and `wp.copy()` just that element.
**Estimated speedup**: Eliminates 200 round-trips over 100 samples.

### 10. Pre-allocate JAX Padding Buffer

**File**: `likelihoods.py`
**Current**: `jax.lax.pad` allocates a new (486, 646, 3) array per call.
**Fix**: With `@jax.jit` (item 5) now applied, XLA should handle buffer reuse automatically. Profile to confirm — if still allocating, pre-allocate manually.
**Status**: Likely already mitigated by JIT compilation.

---

## Low Impact (Polish)

### 11. Remove Redundant NaN Handling

**File**: `likelihoods.py`
**Current**: `jnp.where(jnp.isnan(rendered_xyz), -100.0, rendered_xyz)` checks all pixels per call. With JIT, this is fused into the compiled kernel, so overhead is minimal.
**Status**: Low priority — JIT fusion reduces the standalone cost.

### 12. Lower Render Resolution

**Current**: 640x480 = 307,200 pixels per render.
**Fix**: For sampling (not final visualization), use 320x240 (76,800 pixels). The likelihood is a spatial comparison — half-resolution retains most signal.
**Trade-off**: Faster rendering + smaller JAX arrays, but less spatial precision.
**Estimated speedup**: ~4x on render + likelihood per eval point.

### 13. Reduce Filter Size in Likelihood

**File**: `likelihoods.py` — `filter_size=3` (7x7 window)
**Fix**: Try `filter_size=1` (3x3 window). Fewer distance computations per pixel (9 vs 49 neighbors).
**Trade-off**: Less spatial tolerance for matching, but 5x fewer comparisons per pixel.

### 14. Batch Proposals via Multi-World Simulation

**Current**: Sequential MH — one proposal at a time.
**Fix**: Newton supports multi-world simulation. Run K proposals in parallel using batched environments, evaluate all K likelihoods simultaneously. Could use Multiple-Try Metropolis for theoretical soundness, or run K independent chains.
**Estimated speedup**: Up to Kx on GPU utilization (H100 is likely underutilized by a single 3-body scene).
**Complexity**: High — requires significant refactoring of sampler + likelihood + scene setup.
