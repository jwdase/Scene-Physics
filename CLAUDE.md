# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scene-Physics investigates whether probabilistic models can reconstruct 3D scene structure while incorporating physical properties. The core idea: use GPU-parallel physics simulation to evaluate many placement proposals simultaneously, then pick the best via importance sampling.

## Running Scripts

All scripts are run from `src/scene_physics/` using `uv run`:

```bash
cd src/scene_physics
uv run simulation/complete_parrallel.py
```

Recordings are written to `src/scene_physics/recordings/<experiment_name>/` relative to CWD. Object mesh files are resolved from `src/scene_physics/objects/` via `__file__`-relative paths in each script, so CWD does not affect mesh loading.

Dev tooling (run from project root):
```bash
uv run black .
uv run isort .
uv run pytest
```

## Canonical Script Structure

`simulation/complete_parrallel.py` is the canonical example. Every script follows this 3-phase pattern:

**Phase 1 — Build the world**
```python
worlds = allocate_worlds(NUM_WORLDS)       # N isolated physics worlds + shared ground
obj = make_scene01_world()                 # returns {"observed": [...], "static": [...], "unobserved": [...]}
model, target_state = build_worlds(worlds, obj)
```

**Phase 2 — Build the likelihood**
```python
likelihood = Likelihood_Physics_Parallel(target_state=target_state, model=model, num_worlds=NUM_WORLDS, ...)
```

**Phase 3 — Build and run the sampler**
```python
sampler = ImportanceSampling(model, likelihood, obj, ...)
sampler.run_sampling_linear_print()   # or run_sampling_gibbs()
sampler.print_results()
```

## Object Taxonomy

Every scene function returns a dict with three keys:

| Key | Type | Role |
|-----|------|------|
| `"observed"` | `list[Parallel_Mesh]` | Dynamic objects whose positions are sampled (visible in GT render) |
| `"unobserved"` | `list[Parallel_Mesh]` | Dynamic objects sampled using physics (not directly visible) |
| `"static"` | `list[Parallel_Static_Mesh]` | Kinematic objects shared across all worlds (table, floor accessories) |

## World Building Pipeline (`build_worlds`)

Order matters for `allocs` tracking:
1. Insert static objects once (`world=-1`, shared globally)
2. For each world index: insert all observed dynamic objects, then all unobserved
3. `worlds.finalize()` → call `give_finalized_world(model)` on all objects (sets `obj.allocs` as numpy array and `obj.num_worlds`)
4. Call `move_to_target()` on all objects (sets world-0 to ground-truth position)
5. Capture `target = model.state()` — this is the ground-truth observation
6. Call `freeze_finalized_body()` on dynamic objects (zeros inv_mass/inv_inertia, moves to `OFF_POSITION=(0,-1000,0)`)

`build_worlds` returns `(model, target_state)` — two values only.

## Priors (`properties/priors.py`)

Each `Parallel_Mesh` carries a `Priors` dataclass (defaults shown):

```python
@dataclass
class Priors:
    init_mean: float = 0.0    # initial position mean
    init_std:  float = 0.01   # initial position std
    pos_std:   float = 0.1    # per-step position noise
    rot_std:   float = 0.1    # per-step rotation noise (radians)
    total_iter: int  = 40     # expected iterations (used by scheduler)
    x_min: float = -1.0       # XZ bounding box for proposals
    x_max: float =  1.0
    z_min: float = -1.0
    z_max: float =  1.0
```

Pass per-object priors at construction: `Parallel_Mesh(..., priors=Priors(pos_std=0.05))`.
`obj.set_proposal()` returns `(priors, num_worlds)` — used internally by `ImportanceSampling._gen_proposals`.

## Key Conventions

- **Quaternions**: XYZW ordering throughout (`[qx, qy, qz, qw]`). `body_q` shape is `[N_bodies, 7]` = `[x, y, z, qx, qy, qz, qw]`.
- **Static bodies**: `density=0.0` (`Still_Material`) makes Newton treat a body as kinematic.
- **GPU transfers**: always use `jnp.from_dlpack(warp_array)` — never `.numpy()` then back to JAX.
- **Pixel indices**: `_get_pixel_indices` in `likelihoods.py` must return **numpy arrays** (not jnp) to avoid JAX tracer leaks across JIT compilations.
- **Recordings**: written relative to CWD — always run from `src/scene_physics/`.

## Likelihood Functions

`Likelihood_Physics_Parallel` in `likelihood/likelihoods_physics.py`:

- `new_proposal_likelihood_still_batch(scene)` — batch render current positions, no physics. Returns `(num_worlds,)` numpy array of scores relative to baseline.
- `new_proposal_likelihood_physics_batch(scene)` — run XPBD forward sim for `frames` steps, save CPU snapshots every `eval_every` frames, batch render + average. Slower but accounts for physical plausibility.

Both subtract `self.baseline_score` (self-comparison of the target point cloud). The physics eval stores body_q snapshots as numpy arrays and replays them via a single reusable `_render_state`.

Camera setup: `setup_depth_camera(model, eye, target, width, height, num_worlds)` — note `num_worlds` is required. It builds a `(1, num_worlds)` `camera_transforms` array so every parallel world uses the same viewpoint correctly.

## Sampler (`sampling/parallel_mh.py`)

The sampler class is `ImportanceSampling`. Two run modes:

**`run_sampling_linear_print(debug=False)`** — sequential placement:
1. For each observed object: `run_single_body_sampling(obj, iter_per_obj, physics=False)`
2. For each unobserved object: `run_single_body_sampling(obj, iter_per_obj, physics=True)`
3. Calls `_give_final_positions()` at the end

**`run_sampling_gibbs(iters, debug, burn_in)`** — Gibbs sampling:
1. Burn-in: `run_single_body_sampling` for each object (no physics, `count=False`)
2. Loop: randomly choose an object, call `run_single_sample` (physics, `count=True`)
3. Calls `_give_final_positions()` at the end

### `run_single_body_sampling` inner loop

```
initial_positions()  →  (num_worlds, 7) near init_mean ± init_std, Y ≥ 0
move_6dof_wp → score
for each iteration:
    _generate_positions(positions, scores)  →  softmax resample, top world pinned at [0]
    propose_general(positions, epoch, count)  →  add Gaussian noise, clip to XZ bounds
    move_6dof_wp → score
    _update_all_worlds(scores)              →  particle-filter resample across all objects
_give_final_positions()                     →  one final physics pass, pick top world
```

### Key internal methods

- **`_generate_positions(position, scores)`**: computes `softmax(scores)`, pins the top world at index 0, resamples the rest with replacement — so the best candidate is always preserved.
- **`_update_all_worlds(scores)`**: after each iteration, resamples all objects' body_q rows together using the same softmax-weighted indices, keeping the worlds coherent across objects.
- **`_give_final_positions()`**: runs `new_proposal_likelihood_physics_batch` once to get final scores, finds `top_world = argmax(scores)`, calls `obj.place_final_position(top_world, sample_state)` for every object.

## Proposal (`sampling/proposals.py`)

`SixDOFProposal(priors, num_worlds, seed, schedule="no_decay")`:

- `initial_positions()` → `(num_worlds, 7)` from `Normal(init_mean, init_std)`, `Y = abs(Y)`, identity quaternion.
- `propose_general(positions, epoch_num, count)` → adds noise to rows `[1:]` (row 0 is the pinned best), clips X and Z to bounds, perturbs rotation via axis-angle. If `count=True`, records `pos_std/rot_std` for plotting.
- `get_std()` → applies the schedule to `priors.pos_std` / `priors.rot_std` using `self.cur_iters` and `priors.total_iter`.

Available schedules (passed as string): `"no_decay"`, `"linear"`, `"exp"`.

## Architecture Diagram

```
allocate_worlds(N)
    └── ModelBuilder with N isolated collision worlds + shared ground
         │
         ├── Parallel_Static_Mesh.insert_object_static()  (world=-1, once)
         └── Parallel_Mesh.insert_object(world=i)         (once per world)
              │
              └── worlds.finalize() → Newton Model
                   │
                   ├── give_finalized_world(model)  → obj.allocs, obj.num_worlds
                   ├── move_to_target()             → world-0 = ground truth
                   ├── model.state()                → target_state
                   └── freeze_finalized_body()      → dynamic objects hidden

Likelihood_Physics_Parallel(target_state, model, num_worlds)
    └── setup_depth_camera(..., num_worlds)  → camera_transforms (1, num_worlds)
    └── renders target_state world-0 → correct_pointcloud (H,W,3)
    └── baseline_score = compute_likelihood_score(correct, correct)

ImportanceSampling.run_sampling_linear_print()
    └── for each observed obj:   run_single_body_sampling(physics=False)
    └── for each unobserved obj: run_single_body_sampling(physics=True)
         └── initial_positions() → (num_worlds, 7)
         └── _generate_positions + propose_general → new (num_worlds, 7)
         └── obj.move_6dof_wp(proposals, sample_state)
         └── likelihood_batch(sample_state) → (num_worlds,) scores
         └── _update_all_worlds(scores) → resample body_q across all objects
    └── _give_final_positions() → obj.place_final_position(top_world, sample_state)
```

---

## Known Issues / Next Steps

**Still open**
- `init_positions` path in `run_single_body_sampling` raises `NotImplementedError` — needed for warm-starting.
- No convergence criterion — always runs for exactly `total_iter` / `iters` iterations.
- `Parallel_Mesh.unfreeze_finalized_body` places bodies at random normal positions — may overlap other objects. Should respect `Priors` bounds.
- No tests despite `pytest` being a dev dependency. At minimum: likelihood shape, proposal shape, `build_worlds` smoke test.
- `simulation/mh.py` (`XZ_MH_Sampler`, `XZ_Physics_MH_Sampler`) is legacy single-object 2D sampling — consider archiving.
- Scene design: all target positions currently at `(0,0,0)` — objects overlap, weakening the likelihood gradient. Should use distinct non-overlapping target positions.

**Recently fixed**
- `camera_transforms` was `(1,1)` — caused garbage point clouds for all worlds except world 0. Fixed: now `(1, num_worlds)`.
- `np.clip` missing `a_max` in `propose_general` — fixed to `np.clip(..., 0.0, None)`.
