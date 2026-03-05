# Proposal: Switch to Collapsed Sequential Gibbs Sampling

## Problem

The current sampler (`sampling/parallel_mh.py`) places objects one at a time. When
sampling object _i_, all objects not yet placed are hidden at `OFF_POSITION = (0, -1000, 0)`.
This creates a systematic scoring bias:

```
score(bowl_proposal) = likelihood(target: bowl+coffee+table,
                                  render: bowl_proposal+table)
```

The target contains coffee's pixels; the render does not. Object _i_ is penalised for
failing to explain pixels that belong to objects it should never be responsible for. In
practice this pushes each object toward positions that partially explain _every_ other
object's visual contribution, not just its own.

---

## Proposed Algorithm: Collapsed Sequential Gibbs

Keep **all objects rendered at all times** — at their current best estimate. When
sampling object _i_, every other object _j_ is fixed at its current estimate across all
parallel worlds. Object _i_ is the only thing that varies.

```
initialise: all objects at prior samples (e.g. above the table, small Gaussian)

repeat until convergence:
    for each object i:
        write current_best[j] into sample_state for all j ≠ i  (same across all worlds)
        generate N proposals for object i                        (varies per world)
        render:  object_i @ proposal[k],  all j ≠ i @ current_best[j]
        score:   likelihood(full_target, render_k)   ← now correct
        update:  current_best[i] ← best scoring proposal
```

Because all other objects are rendered, their pixels are already explained. Object _i_'s
score measures only its **marginal contribution** — the gradient is correct.

---

## Required Code Changes

### 1. `simulation/complete_parrallel.py` — remove freeze after target capture

**Current:**
```python
for obj in all_objects:
    obj.move_to_target()
target = model.state()

for obj in (objects["observed"] + objects["unobserved"]):
    obj.freeze_finalized_body()          # ← hides objects; source of the bias
```

**New:**
```python
for obj in all_objects:
    obj.move_to_target()
target = model.state()

for obj in (objects["observed"] + objects["unobserved"]):
    obj.move_to_prior()                  # place at a reasonable prior, keep active
```

`move_to_prior` should place objects at a sensible starting location (e.g. slightly above
the table surface with small XZ noise), not at `OFF_POSITION`. The objects must remain
physically active (non-zero `inv_mass`) throughout sampling so the renderer sees them.

`freeze_finalized_body` and `unfreeze_finalized_body` are no longer needed in the Gibbs
pipeline and can be retired.

---

### 2. `properties/shapes.py` — add `move_to_prior` and remove `body_locked` enforcement

**Add** a `move_to_prior(self, scene, position)` method on `Parallel_Mesh` that writes a
given `(num_worlds, 7)` array into `scene.body_q` at `self.allocs`, analogous to
`move_6dof_wp` but intended for initialisation.

**Remove** the `body_locked` guard from `move_6dof_wp`. In Gibbs sampling every object
must be repositionable in every sweep; permanent locking is incompatible with
re-sampling.

```python
# remove this assert:
assert self.body_locked is False, "Body is already locked"
```

`place_final_position` can be simplified: it no longer replicates a position across all
worlds and sets a lock. Instead it just records the best position found in the last sweep.

---

### 3. `sampling/parallel_mh.py` — rewrite `run_sampling` and `run_single_body_sampling`

This is the largest change. The sampler needs to:

1. Hold a `current_best: dict[str, np.ndarray]` mapping object name → `(7,)` pose.
2. Before sampling object _i_, broadcast `current_best[j]` into all worlds of
   `sample_state` for every `j ≠ i`.
3. Run the proposal / score loop for object _i_ only.
4. After the loop, update `current_best[i]` with the best pose found.
5. Repeat this inner loop over all objects for multiple sweeps.
6. Stop when the maximum positional change across all objects between sweeps falls below
   a convergence threshold `ε`.

**Sketch of new `run_sampling`:**

```python
def run_sampling(self, num_sweeps=5, convergence_eps=0.01):
    all_dynamic = self.objects["observed"] + self.objects["unobserved"]

    # Initialise current_best from prior positions already set in sample_state
    current_best = {obj.name: self._read_world0_pose(obj) for obj in all_dynamic}

    for sweep in range(num_sweeps):
        prev_best = {k: v.copy() for k, v in current_best.items()}

        for obj in all_dynamic:
            # Fix all other objects at current_best across all worlds
            for other in all_dynamic:
                if other.name != obj.name:
                    tiled = np.tile(current_best[other.name], (obj.num_worlds, 1))
                    other.move_6dof_wp(tiled, self.sample_state)

            # Sample object i
            best_pose = self.run_single_body_sampling(
                obj,
                self.iter_per_obj,
                physics=(obj in self.objects["unobserved"]),
            )
            current_best[obj.name] = best_pose

        # Convergence check
        max_shift = max(
            np.linalg.norm(current_best[k][:3] - prev_best[k][:3])
            for k in current_best
        )
        print(f"Sweep {sweep+1}: max position shift = {max_shift:.4f}")
        if max_shift < convergence_eps:
            print("Converged.")
            break
```

`run_single_body_sampling` should return the best `(7,)` pose found across **all
iterations** (not just the last batch), requiring a running `global_best_pos` and
`global_best_score` inside the loop.

---

### 4. `sampling/proposals.py` — warm-start from current estimate

`initial_positions` currently draws from `Normal(0, 0.05)` regardless of where the
object currently is. In Gibbs sampling the object starts each sweep at `current_best[i]`,
so proposals should be centred there.

**Add** an `initial_positions_from(self, centre_pose: np.ndarray)` method:

```python
def initial_positions_from(self, centre_pose: np.ndarray):
    """
    Initialise proposals centred on a given (7,) pose.
    Args:
        centre_pose: (7,) [x, y, z, qx, qy, qz, qw]
    Returns:
        (num_worlds, 7) array
    """
    positions = np.tile(centre_pose, (self.num, 1))
    positions[:, :3] += np.random.normal(0, self._init_std, size=(self.num, 3))
    for i in range(self.num):
        positions[i, 3:] = self._perturb_rotation(positions[i, 3:], self._init_std)
    return positions
```

`run_single_body_sampling` should call this instead of `initial_positions()`, passing
`current_best[obj.name]` as the centre.

---

### 5. Global best tracking inside `run_single_body_sampling`

Currently `place_final_position` uses only `prev_positions` / `prev_scores` from the
**last** iteration. If the optimum was reached at iteration 50 and noise subsequently
perturbed away from it, that best is lost.

Replace with a running global best:

```python
global_best_score = -np.inf
global_best_pos   = None

for iteration in range(total_iter):
    new_positions = proposor.propose_batch(...)
    obj.move_6dof_wp(new_positions, self.sample_state)
    scores = self.likelihood.new_proposal_likelihood_still_batch(self.sample_state)

    top_idx   = int(np.argmax(scores))
    top_score = float(scores[top_idx])
    if top_score > global_best_score:
        global_best_score = top_score
        global_best_pos   = new_positions[top_idx].copy()

    prev_positions = new_positions
    prev_scores    = scores

return global_best_pos   # caller updates current_best
```

---

## Summary of File Changes

| File | Change |
|------|--------|
| `simulation/complete_parrallel.py` | Replace `freeze_finalized_body` with `move_to_prior`; remove freeze loop |
| `properties/shapes.py` | Add `move_to_prior`; remove `body_locked` guard from `move_6dof_wp`; simplify `place_final_position` |
| `sampling/parallel_mh.py` | Rewrite `run_sampling` with multi-sweep Gibbs loop + convergence check; rewrite `run_single_body_sampling` to return best pose + global best tracking |
| `sampling/proposals.py` | Add `initial_positions_from(centre_pose)` for warm-starting |
| `simulation/build_worlds` (helper) | `move_to_prior` replaces `freeze_finalized_body` in the world-building pipeline |

---

## What Does Not Change

- The parallel-worlds infrastructure (`parallel_builder.py`, Newton model/state) is unchanged.
- The likelihood functions (`likelihoods_physics.py`) are unchanged; the scoring is
  correct once all objects are rendered.
- The rendering pipeline (`kernels/image_process.py`, `visualization/camera.py`) is unchanged.
- `SixDOFProposal.propose_batch` and `_perturb_rotation` are unchanged.

---

## Optional Extension: MH Accept/Reject

Once Gibbs sweeps are working, a Metropolis step can be added inside
`run_single_body_sampling` for each proposal batch. Accept the best proposal with
probability `min(1, exp(best_score - current_score))`. This turns the beam search into
an ergodic chain that provably converges to the conditional posterior
`p(x_i | x_{-i}, target)` instead of just its mode.
