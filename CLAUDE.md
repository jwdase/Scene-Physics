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

To **generate a benchmark dataset of scenes** (a separate workflow that produces the USD + priors/truth that a sampling run consumes), see the **Scene Dataset Generation** section below:
```bash
cd src/scene_physics
uv run python -m scene_physics.data_gen.scene_gen
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

## Scene Dataset Generation (`data_gen/`)

Generates a benchmark dataset of `~N` scenes — each a self-contained copy of the `scene01/data/` layout — from the raw OBJ library in `resources/objects/objects/`. Every scene is a physically settled tabletop: a static table, one **hidden** occluded target (a flat-laid wood block), a generator-drawn **occluder** in front of it, and several **observed** surrounders — one of which is dropped to **rest on the hidden target** as a physics-interaction probe (a *visible* object whose settled pose couples, via physics, to the *hidden* block's). The output is drop-in for `sim_sampling.py`.

**Files**
- `data_gen/object_library.py` — load `.obj` → Z-up `newton.Mesh` (+ AABB / `center` / `height`), `lru_cache`d; `available_objects()`, `sample_objects(...)`.
- `data_gen/usd_export.py` — `write_layout_usd(...)` authors a `scene01`-style physics USD; `safe_usd_name(...)`.
- `data_gen/scene_gen.py` — the Drop-&-Settle generator: `SceneSpec`, `generate_scene`, `generate_dataset` (sequential), `generate_dataset_parallel` (batched single-GPU).
- `data_gen/__init__.py` — re-exports the public API.

**Run** (from `src/scene_physics/`): `uv run python -m scene_physics.data_gen.scene_gen` runs the `__main__` block, which calls **`generate_dataset_parallel`** (batched single-GPU generation, see below) and writes to `src/scene_physics/generated_scenes/` — that path is anchored to the script, so it no longer depends on the cwd. Edit `__main__` to configure, or call programmatically:
```python
from scene_physics.data_gen import (
    SceneSpec, generate_dataset, generate_dataset_parallel, available_objects,
)
spec = SceneSpec(
    target="square_wood_block",                  # flat-laid -> stable ~7.7x7.7 cm, 3 cm-tall platform
    pool={
        "mid_height": ["jug04", "bee", "glass1", "int_kitchen_accessories_le_creuset_bowl_30cm",
                       "b05_coffee_grinder", "b04_candle_holder_metal", "vase_05"],
        "small":      ["b03_loafbread", "bung", "orange", "pepper", "coffeemug", "coffeecup004_fix",
                       "shark", "heart", "banana_fix2", "star_wood_block",
                       "round_coaster_stone", "f10_apple_iphone_4", "b03_cocacola_can_cage"],
    },
    n_mid=2, n_small=3)                          # -> 6 placed objects/scene (target + 2 mid [1 occluder] + 3 small)
# Sequential oversample-and-filter (one world per attempt):
generate_dataset(spec, n_scenes=100, out_root="generated_scenes", seed=0)
# OR batched single-GPU: settle `num_worlds` candidate scenes at once, keep passers (one
# process, one GPU, no sharding). Size num_worlds to the card (~64 on an 80 GB H100):
generate_dataset_parallel(spec, n_scenes=100, out_root="generated_scenes", num_worlds=64, seed=0)
```
`SceneSpec(target, pool, n_mid=2, n_small=3, table="dining_room_table")`: you name the target; `pool` is a role-split dict (`{"mid_height": [...], "small": [...]}`). Each scene samples exactly `n_mid` mid-height objects and `n_small` small ones (each accepts an `int` or a `(lo, hi)` range). The **occluder is a height-weighted draw from the sampled `mid_height` objects and counts toward `n_mid`** (so `n_mid` must be ≥ 1); the remaining mid + all small objects are the surrounders. **Curate each list to tabletop scale**: `mid_height` are the occluder candidates (a tall occluder buries the object resting on the low block — see *Physics-interaction probe* below), `small` are the surrounders (several squat enough to be stacked on the block). `available_objects()` lists the ~193 names.

**Per-scene output** under `<out_root>/sceneNNN/`:
- `data/sceneNNN_physics.usdc` — layout USD, re-readable by `add_usd` (identical structure to `scene01_physics.usdc`).
- `data/sceneNNN_truth.json` — `{name: [x,y,z,qx,qy,qz,qw]}` for the table + every dynamic object (settled poses).
- `data/sceneNNN_priors.json` — per dynamic object, in the `scene01` prior schema.
- `data/sceneNNN_makeup.json` — `{static, observed, hidden, occluder, target}`, ready to build `Scene_Makeup`.
- `results/` — empty, pre-created (so the GT render won't `FileNotFoundError`).
- At the dataset root: `<out_root>/scene_stats.txt` — a tab-separated log written incrementally, one row per saved scene (`occluded_fraction`, `occluder`, `candidate`, `surrounders`), a header (`target`, `occlusion_thresh`, `seed`, and `num_worlds` for the batched path) and a final yield footer.

**Pipeline** (one candidate scene; both drivers below are *oversample-and-filter* — a rejected candidate is **discarded, never retried**):
1. `_sample_placements` — the **occluder is a height-weighted random draw** (`_draw_occluder`, `OCCLUDER_HEIGHT_BIAS` — not always the tallest, so a shorter occluder is sometimes chosen and the resting object can show above it) from the `mid_height` pool; surrounders are the remaining mid + all `small` picks. XY placed by **rejection sampling** so footprints don't overlap (the main cause of XPBD blow-ups); target near table center, laid flat via `TARGET_BASE_QUAT`; occluder just in front on the **−Y (camera) side**. With probability `P_STACK` one **small, squat** surrounder (`STACK_MAX_FOOT`/`STACK_MAX_HEIGHT`) is dropped onto the target's top to rest on it. Initial orientation = random yaw about Z, dropped `DROP_LIFT` above the table.
2. `_build_model` (single world) / `_add_table_and_dynamics` (shared helper) — Z-up `ModelBuilder`; table = static collider via **`add_shape_box`** spanning the table AABB (top face at `aabb_max[2]`); dynamics via `add_shape_convex_hull`. The box replaced an `add_shape_mesh` table collider: the table `.obj` is a thin shell that dropped objects tunnelled through under the stacking impact, so **settling uses a solid box** while rendering and the exported USD keep the full visual mesh.
3. `_settle` — XPBD (`SUBSTEPS`×`SOLVER_ITERS`) until `max|body_qd| < REST_THRESH` or `MAX_SETTLE_FRAMES`.
4. **Validate** — `at_rest`, `_on_table` (each object's world AABB-center over the table), and `_occluded_fraction(target) ≥ OCCLUSION_THRESH`. For a stack scene also: the stacked object is in **true contact** with the target (`_contact_pair_exists` after re-colliding at the tight `STACK_CONTACT_MARGIN`), is **resting on its top face** (`_rests_on_target`, within `STACK_REST_Z_TOL`), and stays **visible** (`_visible_fraction ≥ STACK_VISIBLE_THRESH`). Any failure → the candidate is discarded.
5. `_write_artifacts` — emit the four files + `results/`.

Heavy GPU objects are localized per candidate/batch and `gc.collect()`'d → steady-state memory (runs on an 8 GB GPU). There is **no per-scene retry**: each driver draws independent candidates and keeps the passers until `n_scenes` are saved or the candidate budget (`n_scenes * CANDIDATE_BUDGET`, 25) is hit. At the strict `OCCLUSION_THRESH=0.9` gate the yield is low (~7% in a small check), so expect many candidates per saved scene.

**Single-GPU batched generation (`generate_dataset_parallel`).** Instead of one scene at a time, allocate `num_worlds` candidate scenes as `num_worlds` collision-isolated Newton **worlds** in one model and run the physics on all of them at once (no job sharding — one process, one GPU):
- `_build_batched_model` samples `num_worlds` **independent** scenes (each its own `_sample_placements` draw — *not* copies of one scene; contrast `build_worlds`/`replicate` in `sim_sampling.py`, which replicates one scene) and adds each as a separate world via `ModelBuilder.add_world` (all stacked at the origin; isolation is by world index, not spacing). With fixed `n_mid`/`n_small` every world has the same 6 dynamic bodies; per-world `{name: global_body_index}` dicts come from the body-count offset captured before each `add_world`.
- `_settle_batched` settles every world in one XPBD loop; per-world rest is read by slicing `body_qd` with each world's indices.
- Validation reuses the single-world gates per world; occlusion is **three batched depth renders** (full scene / target-only / stacked-only) via a `num_worlds`-sized `Camera`, reading per-world depth from `depth_image[w, 0]`.
- Passers are written and batches repeat until `n_scenes` good (or the budget). `DEFAULT_BATCH_WORLDS` sizes the batch to GPU memory (~64 on an 80 GB H100; the 3-pass full-mesh render is the memory peak, so drop it on small cards). This is the path `__main__` runs. The `candidate` column is bumped per batch, so it is a cumulative count of scenes drawn, not a unique per-scene index.

**Prior semantics**
- **Hidden target**: mean = the occluder's center — `(x, y)` of the occluder's settled AABB-centroid, `z` = table top, identity rotation. (Encodes "the hidden object is probably where the thing blocking it sits.")
- **Observed** (occluder + surrounders): mean = their own settled truth.
- Bounds `x/y_min/max` for every object = the table's XY AABB.

**Occlusion / visibility check** (`_occluded_fraction(name)`, generic over any object): render full settled-scene depth vs the named object rendered alone (default `CameraIntrinsics`, full meshes); a silhouette pixel counts as occluded when something is nearer in the full scene by `DEPTH_MARGIN`. The hidden target must be ≥ `OCCLUSION_THRESH` (0.9) occluded; `_visible_fraction = 1 − _occluded_fraction` gates the stacked object's visibility. (The batched path computes the same fraction from per-world depth slices via `_occ_frac`.)

**Physics-interaction probe (resting contact).** Each scene drops one small surrounder onto the hidden block so a *visible* object's settled pose couples (via physics) to the *hidden* target's pose. A stack is accepted only as a genuine resting contact:
- **True contact** — `model.collide` emits a contact for any pair within `shape_contact_margin`, whose Newton default is **0.1 m** (≈10 cm) — far too loose to mean "touching" (a two-box rig confirms boxes 15 cm apart register as "contact"). `_attempt_scene` therefore sets `shape_contact_margin = STACK_CONTACT_MARGIN` (0.01 m) and re-collides before `_contact_pair_exists` (touching ⇒ within ~0.5 cm). The `dot(normal, p1−p0)` gap is *not* reliable (reads 0–5 cm regardless of true gap), hence the tight-margin re-collide.
- **On the top face** — `_rests_on_target` requires the stacked object's lowest world-z within `STACK_REST_Z_TOL` of the target's top (resting on the block, not standing beside it).
- **Stays visible** — `_visible_fraction(stacked) ≥ STACK_VISIBLE_THRESH`. The resting object is the cue, so it must not be buried by the occluder. **This is why the occluder is a height-weighted draw (not always the tallest) and why the tall objects are kept out of the pool**: a 30–40 cm occluder that hides the 3 cm block also buries a short object on it. Removing the tall objects (occluders then 5–18 cm) raised the visible-interaction rate from ~50% to 100% in a 16-scene check, at the cost of more retries.
- Only small/squat surrounders (`STACK_MAX_FOOT` 0.06 m, `STACK_MAX_HEIGHT` 0.16 m) are eligible to be stacked, so they settle on the narrow ~7.7 cm block top instead of toppling.

**Conventions & gotchas (data_gen-specific)**
- **OBJ are Y-up**: library assets export from Blender Y-up; `load_object` rotates to Z-up via `(x,y,z)→(x,−z,y)` (pure rotation; reproduces scene01 geometry). Sizes stay at **native `.obj` scale** — no normalization.
- **Curate the pool to tabletop scale.** Some assets are huge at native scale (`b04_orange_00` ≈ 1.2 m) and topple/eject → every attempt fails. Use small objects.
- **Physics = convex hull, rendering = full mesh.** The settle and the exported USD use convex hulls; the ray-trace sensor *cannot render hulls* (`Unsupported shape geom type: 10`), so the occlusion check and the downstream GT point cloud use `add_shape_mesh` (full visual mesh). Never render a convex-hull-only model.
- **Table collider = box, settle-time only.** Settling uses a solid `add_shape_box` for the table (Pipeline step 2); the exported USD and all rendering keep the full table mesh. ⚠️ The box is **not** baked into the USD, so re-simulating the *exported* USD can tunnel the block through the table again — e.g. the `simulation/view_all_scenes.sh` viewer runs `simulation.py`, which loads the thin mesh with `skip_mesh_approximation=True` and steps physics. To inspect a generated scene, render its static settled poses rather than re-simulating; a re-sim consumer that needs stability must add its own box collider (or stop skipping mesh approximation and author a `boundingCube`/`convexHull` collider on the table).
- **Digit-leading names** can't be USD prim names; `safe_usd_name` prefixes `_` (`9v_battery` → `_9v_battery`) consistently across the prim path **and** all JSON keys, so the reloaded `body_key` leaf matches the keys. (Such objects appear under the sanitized name in truth/priors/makeup.)
- **Quaternions** are canonicalized to the `qw ≥ 0` hemisphere so `truth.json` matches the `add_usd` readback; near-180° rotations stay sign-ambiguous (double cover) — compare rotations, not raw components.
- **Mass**: dynamics settle with `density=DENSITY`; Newton's computed per-body mass is written into the USD (`PhysicsMassAPI`), so in-memory truth and the reloaded USD agree.
- **Rolling friction** is set on dynamics so cans/batteries stop rather than rolling forever (else they never reach `at_rest`).

Tunables are constants at the top of `scene_gen.py`: physics (gravity, density, friction, `SOLVER_ITERS`=32/`SUBSTEPS`, settle frames, `REST_THRESH`); placement (`PLACE_MARGIN`, `OCCLUDER_GAP`, `OCCLUDER_HEIGHT_BIAS`, `P_NEAR`, `P_STACK`, `DROP_LIFT`=0.02); target orientation (`TARGET_BASE_QUAT`); stack acceptance (`STACK_CONTACT_MARGIN`, `STACK_REST_Z_TOL`, `STACK_MAX_FOOT`, `STACK_MAX_HEIGHT`, `STACK_VISIBLE_THRESH`); occlusion (`OCCLUSION_THRESH`=0.9, `DEPTH_MARGIN`); and the oversample/parallel knobs `CANDIDATE_BUDGET` (25, candidate budget = `n_scenes * CANDIDATE_BUDGET`) and `DEFAULT_BATCH_WORLDS` (worlds per batch for `generate_dataset_parallel`).

**Consuming a generated scene** (drop-in for `sim_sampling.py`):
```python
import json
mk = json.load(open("generated_scenes/scene001/data/scene001_makeup.json"))
scene_makeup = Scene_Makeup(static=set(mk["static"]), observed=set(mk["observed"]), hidden=set(mk["hidden"]))
run_importance_sampling(
    "generated_scenes/scene001/data/scene001_physics.usdc",
    "generated_scenes/scene001/data/scene001_priors.json",
    default_camera, scene_makeup, "generated_scenes/scene001/results")
```
The table is a static collider, so it is **not** in `model.body_key` (only dynamics are); every body name is in the makeup's observed/hidden, so `object_collection` won't trip the ground/table assert.

**Verified this build**: USD round-trips (positions exact, rotation ≈0°); `object_collection` consumes generated scenes unmodified. The **box table collider** fixes the stacking-impact penetration — settled targets rest exactly on the table top (`gap ≈ 0`) in both the sequential and batched paths. **Batched `generate_dataset_parallel`** validated end-to-end (3/3 scenes from 42 candidates at `OCCLUSION_THRESH=0.9`, occlusion ≥0.93, targets on-table, artifacts structurally identical to the sequential path); the core multi-world allocate+settle was checked at 4 worlds × 6 bodies with every world's target resting on the table.

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

---

## Design Note: Composite Likelihood + Volume-Based Penetration Term (planned, not yet implemented)

**Goal.** Today `ParallelPhysicsLikelihood` returns a single purely-visual `(num_worlds,)` score (3DP3 point-cloud match − baseline). The plan is to make it a **composite likelihood** — a weighted product of "expert" factors, summed in log space:

> log L_c(world) = β_v · log L_visual(world) + β_p · log L_phys(world)

Each factor is its own `(num_worlds,)` vector, so the composite is just a weighted sum of vectors. Refactor so components are a list (visual, physics, …) rather than hard-wired — this also makes visual-only vs composite a trivial ablation. The physics factor `log L_phys` is a **physical-plausibility expert**: a config with objects interpenetrating at t=0 is impossible and should be down-weighted.

**Why a *volume* metric (not a contact count / gap).** The dataset is built around **resting contact** — a visible object is *supposed* to touch the hidden target — so correct configs sit at ~0 gap. A binary "any contact ⇒ reject" gate therefore penalizes the *correct* answer; we need a penetration **magnitude with a tolerance band** that reads ~0 for resting and grows with real overlap. `model.collide`'s gap field is unreliable in this rig (see data_gen note: `dot(normal, p1−p0)` reads 0–5 cm regardless of true gap) and its contact *count* over-reports at the default 0.1 m margin, so neither gives a trustworthy magnitude. An *exact* mesh/polytope intersection volume is the right quantity but is too slow / robustness-fragile to run per-world per-iteration.

**The method: sampled (Monte-Carlo) interpenetration volume.** Approximate the overlap volume between each object pair by point-sampling instead of computing it exactly. GPU-parallel, smooth, magnitude-bearing, and cheap (rigid transforms + a max-reduce); sub-ms–low-ms at `num_worlds × ~15 pairs × K~1e3` on an H100.

General algorithm (per importance-sampling evaluation):
1. **Precompute once per object** (independent of pose): K sample points filling the object's volume, expressed in the object's body frame. Convex hulls (the settle/USD geometry) make this easy: sample the hull AABB, keep points inside the hull. Store `points_body[obj]` `(K, 3)` and `vol[obj]`.
2. **Per world, per ordered pair (A, B)** of dynamic bodies (skip non-colliding pairs; optionally include body-vs-table):
   - Transform A's points to world by A's `body_q`, then into B's body frame by `inv(body_q[B])`.
   - **Inside-B test** (convex hull): `inside = all_i(nᵢ · x − dᵢ ≤ 0)` over B's face halfplanes — a matmul + max-reduce. (Non-convex objects: use a precomputed SDF / occupancy grid lookup instead.)
   - `pen_vol(A,B) ≈ mean(inside) · vol(A)`. Symmetrize or just sum over ordered pairs.
3. **Per-world penetration** `V(world) = Σ_pairs pen_vol`. Vectorize the whole thing over `(worlds, pairs, points)` — no Python loop over worlds.
4. **Tolerance band + factor.** Subtract a small allowance for legitimate resting contact, then turn into a Gibbs/Boltzmann factor:
   > log L_phys(world) = −β_p · max(0, V(world) − V_tol)
   `V_tol` absorbs the thin resting overlap; `β_p` sets how hard penetration is penalized. (This is a product-of-experts: penetration treated as an energy, `exp(−βE)`.)

**Caveats to resolve when implementing.**
- **Rank resampler flattens magnitudes.** `proposals.py::propose` weights by *rank*, not `exp(score)`, so a smooth volume penalty only bites when it changes ordering. To use the magnitude as intended, also switch `propose()` to softmax/`exp(score)` weights; otherwise the volume term degenerates toward a gate.
- **Convex-hull sampling is an over-estimate** vs the true (possibly concave) mesh, but it matches the geometry physics already uses, and the penalty only needs to be monotone in real overlap.
- **Per-world bucketing is *not* needed for the sampled-volume path** (you index bodies directly via `Body.allocs[w]`), unlike the `model.collide` route, which pools all worlds into one flat contact buffer with a single global count.
- **`V_tol` calibration**: set it from the typical resting-contact overlap of an accepted dataset scene (the stack probe) so true rests score ~0 and only excess interpenetration is penalized.
- **Alternative considered**: reuse the XPBD solver — step once and measure the correction ‖Δq‖ (overlap → large correction, rest → tiny). Free, but inherits the solver non-determinism logged in the max-likelihood-wander note, so the sampled-volume metric is preferred as the deterministic signal.
