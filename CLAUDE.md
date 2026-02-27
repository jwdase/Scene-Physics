# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scene-Physics investigates whether probabilistic models can reconstruct 3D scene structure while incorporating physical properties and physics simulations. The research question: Can we build probabilistic models that take physical properties and physics into account when reconstructing the 3D structure of a scene from visual input?

## Running Code

The `PYTHONPATH` must be set to the project root for imports to work:

```bash
PYTHONPATH=. python src/scene_physics/simulation/scenes/scene01.py
PYTHONPATH=. python src/scene_physics/simulation/run_parallel_sampling.py
```

Scene entry points are in `src/scene_physics/simulation/scenes/` (single-world MH) and `src/scene_physics/simulation/` (parallel multi-world experiments).

## Development Commands

```bash
# Format code
black .

# Sort imports
isort .

# Run tests
pytest

# Run a single test file
pytest src/path/to/test_file.py
```

## Dependencies

**Python packages:** `warp`, `newton`, `pyvista`, `jax`, `scipy`, `numpy`

**External submodules:**
- **Newton** (`src/newton/`): NVIDIA GPU-accelerated physics engine using Warp. Provides `ModelBuilder`, `SolverXPBD`, `SensorTiledCamera`.
- **B3D/Bayes3D** (`src/b3d/`): Probabilistic inverse graphics library (JAX-based). Point cloud likelihoods have been extracted into `src/scene_physics/likelihood/likelihoods.py` to avoid the GenJax dependency for core sampling.

## Architecture

The main project package is `src/scene_physics/`. All imports use the `scene_physics.*` namespace.

### Package Layout

```
src/scene_physics/
  properties/       # Object representation
  simulation/       # Physics simulation and scene setup
  likelihood/       # Point cloud likelihood functions
  sampling/         # MH samplers and proposals
  kernels/          # Warp GPU kernels
  visualization/    # PyVista rendering
  utils/            # I/O helpers, scene builders, plotting
  objects/          # .obj mesh files for scenes
  recordings/       # Output: videos, images, point clouds
```

### Key Classes

**`properties/shapes.py`**
- `Parallel_Mesh`: dynamic body inserted once per parallel world; tracks `allocs` (body_q indices per world) for reading/writing positions
- `Parallel_Static_Mesh`: shared body inserted globally (world=-1) — floors, tables
- `MeshBody`: single-world simplification of `Parallel_Mesh`

**`properties/material.py` / `basic_materials.py`**
- `Material`: wraps friction (`mu`), `restitution`, `density`, `contact_ke/kd` into Newton's `ShapeConfig`
- `Dynamic_Material`: density=1000, mu=0.8, restitution=0.3
- `Still_Material`: density=0.0 (kinematic/static body)

**`simulation/parallel_builder.py`**
- `allocate_worlds(n)`: creates a Newton `ModelBuilder` with a shared ground plane replicated across N isolated collision worlds
- `build_parallel_worlds(base_builder_fn, num_worlds)`: calls a scene-factory function once per world and merges into a combined builder with a `body_index_map` from `(world_idx, body_name)` → `body_q` index

**`likelihood/likelihoods.py`**
- `compute_likelihood_score(observed_xyz, rendered_xyz)`: JIT-compiled JAX function; 3DP3-style Gaussian mixture log-likelihood over pixel-structured `(H, W, 3)` point clouds
- `compute_likelihood_score_batch(observed_xyz, rendered_xyz_batch)`: vmapped version for `(N, H, W, 3)` batch

**`likelihood/likelihoods_physics.py`**
- `Likelihood_Physics`: single-world version; runs forward physics N frames, evaluates likelihood at `eval_every` snapshots, averages
- `Likelihood_Physics_Parallel`: N-world version; one solver loop runs all worlds in parallel, then batch-renders and batch-evaluates

**`kernels/image_process.py`**
- `depth_to_point_cloud`: Warp GPU kernel converting depth images to world-space `(H, W, 3)` point clouds
- `render_point_cloud` / `render_point_clouds_batch`: orchestrate Newton `SensorTiledCamera` → Warp kernel → JAX via DLPack (zero CPU round-trip)

**`sampling/proposals.py`**
- `SixDOFProposal`: batched 6DOF Gaussian random-walk proposals with optional linear/exponential annealing schedule; rotations use axis-angle perturbation via scipy `Rotation`

**`sampling/parallel_mh.py`**
- `ParallelPhysicsMHSampler`: sequential placement MH — converges one object at a time (greedy best of N parallel proposals), then freezes it and moves to the next

**`visualization/camera.py`**
- `setup_depth_camera`: configures Newton `SensorTiledCamera` with a look-at transform; camera convention is -Z forward, +Y up

**`visualization/scene.py`**
- `PyVistaVisualizer` / `VideoVisualizer`: off-screen PyVista rendering to PNG or MP4 from simulation history arrays

### Key Conventions

- **Coordinate system**: Y-axis up, gravity -9.81 on Y
- **Quaternion format**: XYZW ordering throughout (Warp/Scipy convention)
- **Body state**: `body_q` arrays have shape `[total_bodies, 7]` — `[x, y, z, qx, qy, qz, qw]`
- **Point clouds**: structured `(H, W, 3)` arrays (pixel-indexed, not unordered); invalid/background pixels are `NaN`
- **Warp ↔ JAX**: always use DLPack (`jnp.from_dlpack`) to transfer GPU tensors without CPU round-trips
- **Static bodies**: `density=0.0` makes Newton treat a body as kinematic (infinite mass)
- **Parallel worlds**: Newton isolates collision between worlds by `builder.current_world` assignment; world `-1` is shared global
- **Recordings output**: scripts write to `recordings/<name>/` relative to the working directory (must run from project root)

### End-to-End Sampling Flow

1. `allocate_worlds(N)` → combined `ModelBuilder` with N isolated worlds + shared ground
2. Insert `Parallel_Static_Mesh` objects (world=-1) and `Parallel_Mesh` objects (once per world)
3. `model = builder.finalize()` → call `obj.give_finalized_world(model)` on all objects
4. Construct `Likelihood_Physics_Parallel` with a ground-truth target state
5. `ParallelPhysicsMHSampler.run_sampling()` → for each object: generate N proposals, write to `state.body_q`, run physics, batch render, batch likelihood, keep best
