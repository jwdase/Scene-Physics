# Newton State Copying Reference

## State Overview

Newton's `State` class (`src/newton/newton/_src/sim/state.py`) holds all time-varying simulation data as Warp GPU arrays:

| Field | Type | Description |
|-------|------|-------------|
| `body_q` | `wp.array(wp.transformf)` | Body transforms (7-DOF: pos + quat) |
| `body_qd` | `wp.array(wp.spatial_vectorf)` | Body velocities (6-DOF) |
| `body_f` | `wp.array(wp.spatial_vectorf)` | Body forces |
| `particle_q` | `wp.array(wp.vec3)` | Particle positions |
| `particle_qd` | `wp.array(wp.vec3)` | Particle velocities |
| `particle_f` | `wp.array(wp.vec3)` | Particle forces |
| `joint_q` | `wp.array(float)` | Joint position coordinates |
| `joint_qd` | `wp.array(float)` | Joint velocity coordinates |

## Copy Methods (Cheapest to Most Expensive)

### 1. Reference Swap — Free

No data movement. Just swaps Python references.

```python
state_a, state_b = state_b, state_a
```

### 2. `State.assign()` — Fast In-Place GPU Copy

Copies all array contents from one state into another. Both states must already be allocated with matching shapes. No new memory allocation — pure GPU memcpy.

```python
state_dest.assign(state_src)
```

### 3. `wp.clone()` — Allocate + Copy

Creates a brand new Warp array with copied data. Use when you need an independent copy of a single array.

```python
saved_positions = wp.clone(state.body_q)
```

### 4. `wp.copy()` — Low-Level Array Copy

Copies data between two existing arrays (no allocation).

```python
wp.copy(dest=backup_array, src=state.body_q)
```

## Pattern for MCMC Sampling

Pre-allocate two states up front, then use `assign()` to checkpoint and restore without repeated allocation:

```python
# Setup — allocate once
state_current = model.state()
state_proposed = model.state()

# Initialize current state (e.g., from observed data)
# ...

for step in range(num_samples):
    # Save current state as checkpoint
    state_proposed.assign(state_current)

    # Perturb the proposed state (e.g., modify body_q for one object)
    # ...

    # Simulate forward to get proposed outcome
    for substep in range(sim_substeps):
        solver.step(model, state_proposed, dt)

    # Compute acceptance ratio (e.g., B3D likelihood)
    log_alpha = log_likelihood_proposed - log_likelihood_current

    if np.log(np.random.uniform()) < log_alpha:
        # Accept — proposed becomes current
        state_current.assign(state_proposed)
    else:
        # Reject — restore proposed from current for next iteration
        state_proposed.assign(state_current)
```

### If You Only Need to Save/Restore Positions

For lighter-weight checkpointing when you only modify `body_q`:

```python
saved_q = wp.clone(state.body_q)  # save

# ... perturb and simulate ...

if rejected:
    wp.copy(dest=state.body_q, src=saved_q)  # restore just positions
```
