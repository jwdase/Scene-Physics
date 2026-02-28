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

`simulation/complete_parrallel.py` is the canonical example of how experiments are organized. Every script follows this 3-phase pattern:

**Phase 1 — Build the world**
```python
worlds = allocate_worlds(NUM_WORLDS)          # N isolated physics worlds + shared ground
obj = make_scene01_world()                     # returns {"observed": [...], "static": [...], "unobserved": [...]}
model, target_state, _ = build_worlds(worlds, obj["static"], obj["observed"], obj["unobserved"])
```

**Phase 2 — Build the likelihood**
```python
likelihood = Likelihood_Physics_Parallel(target_state=target_state, model=model, ...)
```

**Phase 3 — Build and run the sampler**
```python
sampler = ParallelPhysicsMHSampler(model, likelihood, obj)
sampler.run_sampling()
sampler.print_results()
```

## Object Taxonomy

Every scene function returns a dict with three keys:

| Key | Type | Role |
|-----|------|------|
| `"observed"` | `list[Parallel_Mesh]` | Dynamic objects whose positions are sampled (visible in GT render) |
| `"unobserved"` | `list[Parallel_Mesh]` | Dynamic objects sampled using physics (not directly visible) |
| `"static"` | `list[Parallel_Static_Mesh]` | Kinematic objects shared across all worlds (table, floor accessories) |

`ParallelPhysicsMHSampler.run_sampling()` uses this distinction: observed objects use `new_proposal_likelihood_still_batch` (no physics, fast), unobserved use `new_proposal_likelihood_physics_batch` (forward sim, slower).

## World Building Pipeline (`build_worlds`)

Order matters for `allocs` tracking:
1. Insert static objects once (`world=-1`, shared globally)
2. For each world index: insert all observed dynamic objects, then all unobserved
3. `worlds.finalize()` → call `give_finalized_world(model)` on all objects (sets `obj.allocs` as numpy array and `obj.num_worlds`)
4. Call `move_to_target()` on all objects (sets world-0 to ground-truth position)
5. Capture `target = model.state()` — this is the ground-truth observation
6. Call `freeze_finalized_body()` on dynamic objects (zeros inv_mass/inv_inertia, moves to `OFF_POSITION=(0,-1000,0)`)

## Key Conventions

- **Quaternions**: XYZW ordering throughout (`[qx, qy, qz, qw]`). `body_q` shape is `[N_bodies, 7]` = `[x, y, z, qx, qy, qz, qw]`.
- **Static bodies**: `density=0.0` (`Still_Material`) makes Newton treat a body as kinematic.
- **GPU transfers**: always use `jnp.from_dlpack(warp_array)` — never `.numpy()` then back to JAX.
- **Pixel indices**: `_get_pixel_indices` in `likelihoods.py` must return **numpy arrays** (not jnp) to avoid JAX tracer leaks across JIT compilations.
- **Recordings**: written relative to CWD — always run from `src/scene_physics/`.

## Likelihood Functions

Two likelihood modes live in `likelihood/likelihoods_physics.py` on `Likelihood_Physics_Parallel`:

- `new_proposal_likelihood_still_batch(scene)` — batch render current positions, no physics. Returns `(num_worlds,)` numpy array of scores relative to baseline.
- `new_proposal_likelihood_physics_batch(scene)` — run XPBD forward sim for `frames` steps, render at `eval_every` checkpoints, average. Slower but accounts for physical plausibility.

Both subtract `self.baseline_score` (self-comparison of the target point cloud).

## Sampler (`sampling/parallel_mh.py`)

`ParallelPhysicsMHSampler` takes `objects` dict (same structure as scene function return). `run_single_body_sampling(obj, total_iter, physics=False)`:
1. `SixDOFProposal(obj)` — uses `obj.num_worlds` to size proposals
2. `initial_positions()` → `(num_worlds, 7)` array, identity quaternion, small Gaussian position
3. Loop: `propose_batch` picks top-5 by score, repeats+perturbs → new `(num_worlds, 7)` proposals
4. After loop: `place_final_position` takes best score, replicates across all worlds, locks body

## Proposal (`sampling/proposals.py`)

`SixDOFProposal.propose_batch(pos, scores, cur_it, total_it)`:
- Selects top `n=5` positions by score via `np.argpartition`
- Repeats+truncates to `num_proposals` rows
- Adds Gaussian noise to XYZ; perturbs rotation via axis-angle composition

`n=5` is hardcoded — will fail if `num_worlds < 5`.

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

Likelihood_Physics_Parallel(target_state, model)
    └── renders target_state → correct_pointcloud (H,W,3)
    └── baseline_score = compute_likelihood_score(correct, correct)

ParallelPhysicsMHSampler.run_sampling()
    └── for each observed obj:  run_single_body_sampling(physics=False)
    └── for each unobserved obj: run_single_body_sampling(physics=True)
         └── SixDOFProposal → proposals (num_worlds, 7)
         └── obj.move_6dof_wp(proposals, sample_state)
         └── likelihood_batch(sample_state) → (num_worlds,) scores
         └── obj.place_final_position(best) → lock body
```

---

## Next Steps / Improvements

**Bugs / Fragile Code**
- `SixDOFProposal.propose_batch` hardcodes `n=5` — crashes if `NUM_WORLDS < 5`. Should be `n = min(5, num_proposals)`.
- `ParallelPhysicsMHSampler.print_results` iterates all keys including `"static"`, but static objects never have `final_position` set — will print `None` or error if that changes.
- `build_worlds` inserts dynamic objects with outer loop over worlds, inner loop over objects. This means `allocs` for each object is `[world_0_idx, world_1_idx, ...]` — correct, but the insertion pattern is fragile if object order changes between world iterations.
- `Parallel_Mesh.unfreeze_finalized_body` places bodies at a random normal position, which may overlap with other objects. Should sample from a reasonable prior (e.g., above the table).

**Missing Features**
- `init_positions` path in `run_single_body_sampling` raises `NotImplementedError` — needed for warm-starting from a previous run or prior.
- No MH accept/reject step in `run_single_body_sampling` — currently pure importance sampling (greedy best-of-N). True MH would allow escaping local optima.
- `SixDOFProposal._init_mean` and `_init_std` are hardcoded (`0.0`, `0.05`) — should be configurable per-object or inferred from scene bounds.
- No convergence criterion — sampling always runs for exactly `total_iter` iterations.
- `VideoVisualizer` / `PyVistaVisualizer` in `visualization/scene.py` are not wired into the parallel pipeline — no way to watch the sampling progress.

**Quality / Infrastructure**
- No tests exist despite `pytest` being a dev dependency. At minimum: likelihood shape tests, proposal shape tests, `build_worlds` smoke test.
- `simulation/mh.py` (`XZ_MH_Sampler`, `XZ_Physics_MH_Sampler`) is legacy single-object 2D sampling — consider removing or archiving to reduce confusion.
- `simulation/run_parallel_sampling.py`, `accelerated_depth.py`, `accelerated_sampling.py` appear to be older experiments — unclear if still valid.
- `build_worlds` could become a standalone utility in `parallel_builder.py` rather than living in each experiment script.
