# GPU Speedup Proposal

## Summary

The GPU is underutilized (~50-60%) despite GPU passes taking the longest. The root cause is
repeated CPU-GPU ping-pong: the GPU finishes quickly, then sits idle while the CPU transfers
state, generates proposals, and queues the next kernel. All fixes below target this pattern.

Estimated combined speedup: **1.5-1.8x** (subject to profiling — see caveat below).

---

## Profiling First

Before implementing anything, isolate the two dominant costs:

```python
import time
import warp as wp
import jax

# Time just the Newton XPBD solver
t0 = time.perf_counter()
for frame in range(self.frames):
    state_0.clear_forces()
    contacts = self.model.collide(state_0)
    self.solver.step(state_0, state_1, self.control, contacts, self.dt)
    state_0, state_1 = state_1, state_0
wp.synchronize()
print(f"Solver: {(time.perf_counter() - t0) * 1000:.1f}ms")

# Time just render + likelihood
t0 = time.perf_counter()
batch_clouds = self._render_batch(self._render_state)
scores = compute_likelihood_score_batch(
    observed_xyz=self.correct_pointcloud,
    rendered_xyz_batch=batch_clouds,
)
jax.block_until_ready(scores)
print(f"Render + likelihood: {(time.perf_counter() - t0) * 1000:.1f}ms")
```

If the solver dominates (e.g. 30ms vs 10ms for everything else), the ceiling on improvement
is ~1.3x. If render+likelihood dominates, fixes 3 and 4 below are the priority.

---

## Fix 1 — Vectorize rotation perturbation (Trivial, ~4ms/iter)

**File:** `sampling/proposals.py:96-97`

**Current:**
```python
for i in range(1, self.num):                                        # 49 serial iterations
    positions[i, 3:] = self._perturb_rotation(positions[i, 3:].squeeze(), rot_std)
```
GPU sits idle during 49 sequential scipy `Rotation` calls on CPU.

**Fix:** `Rotation.from_rotvec` and `Rotation.from_quat` both accept `(N, 3)` / `(N, 4)` batched
input — collapse the loop entirely:

```python
axis_angles = np.random.normal(0, rot_std, size=(self.num - 1, 3))
perturbations = Rotation.from_rotvec(axis_angles)
current_rots = Rotation.from_quat(positions[1:, 3:])
positions[1:, 3:] = (perturbations * current_rots).as_quat()
```

**Estimated saving:** ~4ms/iteration (removes CPU bottleneck between GPU likelihood and next proposal).

---

## Fix 2 — Warp scatter kernel for in-place state updates (Small, ~4ms/iter)

**Files:** `properties/shapes.py:212-214`, `sampling/parallel_mh.py:169-175`

**Current (`move_6dof_wp`):**
```python
bodies = scene.body_q.numpy()           # GPU→CPU: downloads ALL bodies, ALL worlds
bodies[self.allocs] = prop_pos          # Modifies only this object's rows on CPU
scene.body_q = wp.array(bodies, ...)    # CPU→GPU: uploads EVERYTHING back
```
Forces full GPU sync and round-trips the entire state (all worlds, all bodies) just to update
one object's rows. `_update_all_worlds` has the same pattern (lines 169-175).

**Fix:** A small Warp kernel scatters `prop_pos` directly into `body_q` on GPU with no CPU
involvement:

```python
@wp.kernel
def scatter_body_q(
    body_q: wp.array(dtype=wp.transformf),
    allocs: wp.array(dtype=wp.int32),
    proposals: wp.array(dtype=wp.transformf),
):
    i = wp.tid()
    body_q[allocs[i]] = proposals[i]
```

```python
# replaces move_6dof_wp body
wp.launch(scatter_body_q, dim=len(self.allocs),
          inputs=[scene.body_q, self.allocs_wp, prop_pos_wp])
```

Requires `self.allocs` to be kept as a persistent Warp int32 array (set once in
`give_finalized_world`). Same kernel covers `_update_all_worlds`.

**Estimated saving:** ~4ms/iteration (3 hard syncs removed — 1 in `move_6dof_wp`, 2 in
`_update_all_worlds`).

---

## Fix 3 — Batch all eval-point renders in one GPU call (Medium, ~5-8ms/iter)

**File:** `likelihood/likelihoods_physics.py:344-365`

**Current:** Phase 1 saves snapshots by forcing GPU→CPU sync every `eval_every` frames, then
Phase 2 iterates serially — each iteration uploads a snapshot, renders, and runs likelihood
as a separate GPU launch:

```python
# Phase 1 — 2 GPU→CPU syncs (frames=50, eval_every=20 → eval_idx=2)
self._eval_states[eval_idx] = state_1.body_q.numpy()   # GPU→CPU sync

# Phase 2 — 2 serial iterations, each with:
for i in range(eval_idx):
    self._render_state.body_q = wp.array(self._eval_states[i], ...)  # CPU→GPU upload
    batch_clouds = self._render_batch(self._render_state)             # GPU render
    scores = compute_likelihood_score_batch(...)                      # JAX JIT
    total_scores += np.asarray(scores)                                # GPU→CPU
```

**Fix:** Keep eval snapshots as GPU Warp arrays. After physics, render all eval states in
one batched call with effective batch size `(eval_idx × num_worlds)`:

```python
# Phase 1 — no CPU sync, keep on GPU
if (frame + 1) % self.eval_every == 0:
    self._eval_states[eval_idx].body_q.assign(state_1)   # GPU→GPU copy, no sync
    eval_idx += 1

# Phase 2 — single batched render across all eval points
# Stack body_q for all eval states, render with (eval_idx * num_worlds) effective batch
# then reshape scores to (eval_idx, num_worlds) and sum over eval_idx axis
```

Requires `_eval_states` to revert to a list of `model.state()` objects (GPU-resident), but
the Phase 2 loop collapses from `eval_idx` serial JAX calls to one.

**Estimated saving:** ~5-8ms/iteration (removes `eval_idx` serial launches and CPU round-trips).

---

## Fix 4 — Softmax and argmax on JAX/GPU (Trivial, ~0.5ms/iter)

**File:** `sampling/parallel_mh.py:83, 158, 162`

**Current:**
```python
probs = softmax(scores)       # scipy — CPU
top_index = np.argmax(scores) # numpy — CPU
```

Scores are already on CPU (from a prior `.numpy()` download), so this is only a minor saving.
However, if Fix 2 keeps scores GPU-resident longer in the future, this becomes important.

**Fix:**
```python
probs = jax.nn.softmax(jnp.asarray(scores))
top_index = int(jnp.argmax(probs))
```

**Estimated saving:** ~0.5ms/iteration.

---

## Estimated Per-Iteration Breakdown

| Phase | Current | After Fixes | Notes |
|-------|---------|-------------|-------|
| Proposal generation (CPU) | ~4ms | ~0.1ms | Fix 1 |
| `move_6dof_wp` sync | ~2ms | ~0ms | Fix 2 |
| Newton XPBD solver (GPU) | ~5-15ms | ~5-15ms | Unchanged |
| Eval snapshot syncs | ~2ms | ~0ms | Fix 3 |
| Render + likelihood (GPU) | ~15-25ms | ~10-18ms | Fix 3 |
| `_update_all_worlds` sync | ~2ms | ~0ms | Fix 2 |
| Softmax / argmax (CPU) | ~1ms | ~0.5ms | Fix 4 |
| **Total** | **~30-50ms** | **~20-35ms** | **~1.5-1.8x** |

---

## Important Caveat

If Newton's XPBD solver alone takes >30ms/iteration, it dominates and the ceiling on
improvement is closer to **1.2-1.3x** regardless of the above fixes. Profile first (see top
of document) before investing in implementation.

The fixes are ordered by effort:

| Fix | Effort | Impact |
|-----|--------|--------|
| 1 — Vectorize rotations | 15 min | Medium |
| 4 — JAX softmax/argmax | 15 min | Low |
| 2 — Warp scatter kernel | 2-3 hours | High |
| 3 — Batch eval renders | 3-4 hours | High |
