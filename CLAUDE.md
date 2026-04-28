# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scene-Physics investigates whether probabilistic models can reconstruct 3D scene structure while incorporating physical properties. The core idea: use GPU-parallel physics simulation to evaluate many placement proposals simultaneously, then pick the best via importance sampling.

## Running Scripts

The active workflow is driven by `simulation/sim_sampling.py`. Run from `src/scene_physics/` using `uv run`:

```bash
cd src/scene_physics
uv run simulation/sim_sampling.py
```

Scenes are loaded from `simulation/scene01/data/*.usdc`. Priors are loaded from `simulation/scene01/data/scene01_priors.json`. Outputs (rendered point clouds, etc.) are written under `simulation/scene01/results/` — that directory must exist before running.

Dev tooling (run from project root):
```bash
uv run black .
uv run isort .
uv run pytest
```

## Canonical Script Structure

`simulation/sim_sampling.py::run_importance_sampling` is the canonical entry point. It runs five phases:

**Step 1 — Render the ground-truth point cloud**
```python
point_cloud = gen_save_point_cloud(scene_usd, intrinsics, f"{save_dir}/point_cloud.ply")
```
Loads the USD scene into a single-world `ModelBuilder`, runs the camera once, saves the cloud as `.ply`.

**Step 2 — Build the parallel worlds**
```python
model, objects = build_worlds(scene_usd, scene_makeup)
```
Loads the USD into a "blueprint" `ModelBuilder`, then `replicate`s it `NUM_WORLDS` times into the real builder (with a shared ground plane, Z-up). Returns the finalized `model` and an `Object_Collection` whose `Body` entries track per-world body indices via `allocs`.

**Step 3 — Build the likelihood**
```python
multi_camera = MultiWorldCamera(intrinsics, model)
likelihood   = ParallelPhysicsLikelihood(multi_camera, point_cloud, model)
```
The camera owns the `SensorTiledCamera`, ray buffer, and per-world `camera_transforms`. The likelihood owns the XPBD solver + reusable state buffers and a baseline score.

**Step 4 — Insert priors on the dynamic objects**
```python
objects.assign_priors(prior_json, NoDecayProposal, base_rng)
```
Reads per-object priors from JSON and constructs a `Proposer` (e.g. `NoDecayProposal`) on each `Dynamic` body. Static bodies are skipped.

**Step 5 — Run importance sampling**
Currently empty; the proposer/objects/model/likelihood are wired up but the loop is not yet ported into the new workflow.

## Object Taxonomy (`properties/shapes.py`)

`Scene_Makeup` partitions named objects into three roles:

| Class | Role |
|-------|------|
| `Static`   | Kinematic; no proposer assigned |
| `Observed` | Dynamic; visible in the GT render |
| `Hidden`   | Dynamic; not directly visible, sampled via physics |

`Object_Collection` is a dict-like container of `Body` instances. Each `Body` carries `name`, `allocs` (numpy array of body indices across replicated worlds), and — for `Dynamic` bodies — a `prior` and `proposer` after `assign_priors` runs.

`object_collection(model, scene_makeup)` walks `model.body_key`, takes `name = body_name.split('/')[-1]`, and asserts the name is in `scene_makeup`. **Caveat:** `add_ground_plane()` in `build_worlds` may add a ground body whose key is not in `scene_makeup` and will trip the assert. If you hit this, either skip non-makeup names in the loop or include the ground key in `Scene_Makeup.static`.

## Priors (`scene01/data/scene01_priors.json`)

Per-object priors are loaded from JSON. Each entry has the shape:

```json
{
  "f10_apple_iphone_4": {
    "position": [x, y, z, qx, qy, qz, qw],
    "pos_std":  0.1,
    "rot_std":  0.1,
    "x_max":  0.5, "x_min": -0.5,
    "y_max":  0.5, "y_min": -0.5
  }
}
```

`sampling/proposals.py::Prior.__init__` parses this dict. Only objects listed in the JSON receive a proposer; unlisted ones (e.g. statics) are skipped.

## Proposer (`sampling/proposals.py`)

- `Proposer(rand_seed, prior)` — base class. Holds `self.rng`, `self._cur_pos_std`, `self._cur_rot_std`. Subclasses implement `_update_pos_std` / `_update_rot_std` for scheduling.
- `NoDecayProposal(Proposer)` — fixed std (no decay).
- `initial_proposal(num_worlds)` → `(num_worlds, 7)` body_q rows. Row 0 is the prior's mean transform; rows `[1:]` add Gaussian position noise and an axis-angle rotation perturbation. Bounds clipped via `_apply_bounds`.
- `propose(positions, likelihood)` → `(n, 7)` particle-filter resample. Row 0 keeps the argmax; rows `[1:]` are sampled from `positions` weighted by `likelihood`, then perturbed.

`_apply_bounds` clips the X and Y columns to `prior.x_min/max`, `prior.y_min/max`. This is the table-top plane (Z is height in the Z-up convention).

## Likelihood (`likelihood/likelihoods.py`)

`ParallelPhysicsLikelihood(camera, target_point_cloud, model, ...)` exposes two scoring modes:

- `still(scene)` — render `scene` once, return `(num_worlds,)` scores − `baseline_score`.
- `physics(scene)` — forward-sim `frames` steps of XPBD with `substeps` per frame; snapshot `body_q` every `eval_every` frames; replay each snapshot through the camera; return `(num_worlds,)` averaged scores − `baseline_score`.

`baseline_score` is `compute_likelihood_score(target, target)` (self-comparison), so the returned scores are *deltas* from the perfect match.

## Camera (`simulation/sim_sampling.py`)

`SingleWorldCamera` and `MultiWorldCamera` both extend a base `Camera`. Both wrap `SensorTiledCamera` and build a `(1, num_worlds)` `camera_transforms` so every parallel world uses the same viewpoint.

`CameraIntrinsics(width, height, fov_degree, max_depth, eye, target, up)` is a dataclass; defaults are Z-up with `eye=(0, -1.5, 1.5)`, `target=(0,0,0)`, `up=(0,0,1)`.

## Key Conventions

- **Up axis**: Z-up. `eye=(0, -1.5, 1.5)` looks from −Y at the origin. The table top is the XY plane; objects sit at `z ≈ 0.75`.
- **Quaternions**: XYZW ordering. `body_q` rows are `[x, y, z, qx, qy, qz, qw]`.
- **Static bodies**: `density=0.0` (`Still_Material`) makes Newton treat the body as kinematic.
- **GPU transfers**: use `jnp.from_dlpack(warp_array)` — never `.numpy()` then back to JAX.
- **Working directory**: always run from `src/scene_physics/`. Relative paths in `__main__` (e.g. `scene01/data/...`) resolve from there.

## Architecture Diagram

```
gen_point_cloud(scene_usd)            # single-world ground-truth render
    └── ModelBuilder + add_ground_plane + add_usd
    └── SingleWorldCamera.render → (H,W,3) point cloud → save .ply

build_worlds(scene_usd, scene_makeup)
    ├── blueprint = ModelBuilder().add_usd(...)
    └── builder.replicate(blueprint, NUM_WORLDS, spacing=0)
         └── model = builder.finalize()
         └── object_collection(model, scene_makeup)
              └── walks model.body_key, dispatches Static/Observed/Hidden

ParallelPhysicsLikelihood(MultiWorldCamera, target_pc, model)
    ├── SolverXPBD + state buffers + render_state
    └── baseline_score = score(target, target)

Object_Collection.assign_priors(prior_json, NoDecayProposal, rng)
    └── for each name in JSON:
         └── Dynamic.set_proposer(child_rng, prior_dict, NoDecayProposal)
              ├── self.prior    = Prior(prior_dict)
              └── self.proposer = NoDecayProposal(child_rng, self.prior)
```

---

## Legacy Code

The following files are **legacy** under the new workflow and are not on the active path. They reference symbols that no longer exist (`Parallel_Mesh`, `Parallel_Static_Mesh`, `SixDOFProposal`, `allocate_worlds`, `build_worlds` from `utils/setup.py`) and will not import as-is. Either rewrite against the new API or archive:

- `sampling/parallel_mh.py` — old `ImportanceSampling` class with `run_sampling_linear_print` / `run_sampling_gibbs`. References undefined `SixDOFProposal`. The new sampling loop has not yet been ported here.
- `simulation/sampling.py` — uses old `allocate_worlds` / `build_worlds`.
- `simulation/simulation.py` — older single-script entry point.
- `utils/setup.py` — `build_worlds(worlds, objects)` over the old object taxonomy.
- `utils/parallel_builder.py` — `allocate_worlds`.
- `visualization/scene.py` — pyvista-based viewer using `Parallel_Mesh` types.

Don't import these from new code. The `scene_physics.sampling` package's `__init__.py` only re-exports `ImportanceSampling`; instantiating it currently fails because of the missing `SixDOFProposal`.

---

## Known Issues / Next Steps

**Still open**
- **Step 5 unimplemented** — `run_importance_sampling` ends after `assign_priors`; the actual sampling loop (call `proposer.initial_proposal`, score with `likelihood.still` / `likelihood.physics`, resample with `proposer.propose`) is not yet wired up.
- **Ground-plane assertion** — `object_collection` may assert on a ground body name that isn't in `Scene_Makeup`. Confirm by printing `model.body_key` after `build_worlds`.
- **Coordinate convention in `_apply_bounds`** — clips X and Y, which is correct for Z-up table-top. Confirm priors `y_max/min` match the actual Y extent of the table.
- **No convergence criterion** — the planned loop will need an iteration cap or a stop rule.
- **No tests** despite `pytest` being a dev dependency. Minimum target: prior JSON parse, `Object_Collection.assign_priors` smoke test, likelihood shape, proposer shape.
- **Output directory** — `gen_save_point_cloud` writes to `{save_dir}/point_cloud.ply` and will raise `FileNotFoundError` if the parent doesn't exist. Either create it in the script or document it in the run instructions.

**Recently fixed**
- `sampling/__init__.py` — removed broken `from .proposals import SixDOFProposal` re-export.
- `properties/shapes.py` — typed `rng` as `np.random.Generator` (was `np.Generator`).
- `proposals.py::Prior` — `pos_y/z` and `quat_x/y/z/w` now indexed correctly.
- `proposals.py::Proposer.initial_proposal` — uses `np.tile` (was `np.repeat`, which flattened).
- `proposals.py::Proposer.propose` — `self.rng.choice` resamples row indices then gathers, instead of flattening the 2-D array.
- Unused imports cleaned across `material.py`, `kernels/image_process.py`, `visualization/scene.py`, `proposals.py`, `shapes.py`.
