# Agent task — maximize photorealism of the Scene-Physics renders

You are a Claude Code agent running on an **H100 box**. Your job is to **greatly
increase the photorealism** of the Blender renders of the generated scenes, then
render the full dataset. These images are **perceptual stimuli shown to human
study participants**, so the bar is: *a naive viewer should read them as
photographs of a real tabletop, not CG.* That overrides everything else below.

This is an **iterative, vision-driven** task. You have image vision: render,
**open the PNGs with the Read tool, critique them like a CG lighting/lookdev
artist, adjust, re-render, repeat.** Do not declare success without looking.

---

## 0. Orient yourself first (read before changing anything)

- Working dir for all commands: `src/scene_physics/`.
- Read these to load context:
  - `CLAUDE.md` (project overview + Scene Dataset Generation section).
  - `docs/SensorTiledCamera_Guide.md` (camera internals).
  - The render pipeline you'll be improving:
    - uv side: `src/scene_physics/visualization/{render_pipeline,blender_runner,segmentation,newton_projection}.py`
    - Blender side (standalone bpy scripts): `src/scene_physics/visualization/blender/{render_scene,_materials,material_specs,_lighting,_seg,_camera,_scene}.py`
  - Tests that encode the invariants you must NOT break:
    `tests/test_camera_match.py`, `tests/test_render_pipeline.py`.
- Scenes live in `resources/generated_scenes/scene*/`. Each has
  `data/<scene>_physics.usdc`, `_truth.json`, `_makeup.json`, and a `results/`.
- There are ~100 scenes drawing from **19 distinct objects** (incl. the table).
  The full object list + current per-object material guesses are in
  `material_specs.py`.

### How the pipeline works (so you change the right thing)
`render_pipeline.render_scene()` writes a job JSON and calls Blender
`render_scene.py`, which: imports the scene USD (poses bake in exactly), assigns a
**procedural** material per object by name, sets a studio HDRI world + the matched
camera, renders a **Cycles beauty pass** (`render.png`) and a flat-color **ID
pass** (`seg_raw.png`); then `segmentation.py` (uv side) decodes the ID pass into
`segmentation.png` + overlay. **Improving fidelity = editing the Blender-side
lookdev (`_materials.py`, `material_specs.py`, `render_scene.py`, the world/HDRI,
Cycles settings), not the camera or the segmentation logic.**

---

## 1. Environment setup (H100)

- The renders need a **USD-capable official Blender** (distro/apt builds are often
  compiled WITHOUT USD and without GPU Cycles). Check
  `blender --background --python-expr "import bpy;print(bpy.app.build_options.usd)"`.
  If it prints `False` or USD import fails, download an official build
  (`https://download.blender.org/release/`, 4.2 LTS or newer), extract it, and
  point `$BLENDER` at it. Verify it sees the H100 for Cycles (OPTIX/CUDA).
- Always export `BLENDER=/path/to/official/blender` before running the pipeline.
- You have 80 GB of VRAM — render on **GPU (OPTIX)**, use high sample counts, and
  batch freely. (`render_scene.py::configure_cycles` already prefers OPTIX/CUDA;
  the `device="GPU"` default applies.)
- Run a single scene:
  `BLENDER=... uv run python -m scene_physics.visualization.render_pipeline --scenes scene001 --samples 512`
  All scenes: add `--all`. Outputs land in each scene's `results/`.

---

## 2. Hard invariants — keep these GREEN the whole time

Run after any change that could affect geometry/camera/segmentation:
`BLENDER=... uv run pytest tests/test_camera_match.py tests/test_render_pipeline.py -q`

- **Camera match**: do NOT edit `_camera.py` or `newton_projection.py` math. The
  Blender camera is verified identical to Newton's `SensorTiledCamera` to <1px.
  The stimulus must stay aligned with the point cloud / segmentation.
- **Segmentation stays exact and aligned**: the ID pass must keep using the same
  camera + poses as the beauty pass. If you restructure rendering, re-verify the
  overlay registers (a test checks this). Each of the 19 objects must keep a
  distinct, decodable label.
- **Poses**: keep importing the scene USD (poses are baked, verified 0 mm). Don't
  move/rescale objects.
- **No UVs → procedural only**: the source meshes have **zero UV coordinates**, so
  you cannot apply image textures/decals. Use Principled BSDF + procedural detail
  on *Generated*/*Object* texture coordinates. (Consequence: branded/figurative
  items — iPhone screen, etc. — cannot be made recognizable. That's an accepted
  limitation; do NOT spend effort UV-unwrapping unless explicitly told to.)
- **Per-object materials are keyed by name**; every one of the 19 dataset objects
  must have a tuned spec (a test enforces coverage).
- Don't commit or push unless asked.

---

## 3. Fidelity levers — push these hard

Realism here comes mostly from **lighting + material micro-detail + clean
sampling**, not geometry. Concretely:

**Lighting & environment** — *a baseline already exists in `_lighting.py`*: a
studio HDRI world (reflections + fill) + a key/fill/rim area-light rig
(`setup_studio_lighting`, key-dominant) + a `add_ground_plane` at the scene's
min-z. It produces a clean high-key studio look. **Refine it, don't rebuild it:**
- Try a higher-res / different studio HDRI (≥4k) and tune `world_strength` +
  per-light energies for more convincing contrast and softer, well-defined
  shadows. Keep lighting identical across all scenes (controlled stimulus).
- The floor currently blends into the HDRI background (seamless white sweep).
  Decide if you want that vs a more visible neutral surface or a curved cyclorama;
  adjust the ground material/world accordingly. Keep it grounded with contact
  shadows either way.
- Tune exposure / view transform (AgX default; compare Filmic). Avoid blown
  highlights and crushed blacks.
- NOTE: the ground plane + lights are hidden during the segmentation ID pass
  (`_lighting.hide_for_id_pass`). If you add new non-object geometry/lights, hide
  them there too so they don't pollute the labels.

**Sampling & denoising (you have the VRAM)**
- Raise samples substantially (e.g. 512–4096; find the knee). Enable adaptive
  sampling. Use OpenImageDenoiser **with albedo + normal passes (prefiltered)**.
  Clamp indirect to kill fireflies. Final renders must be visibly noise-free.
- Set enough light bounces for glass/transmission and glossy interreflection.

**Materials (per category — make them physically plausible)**
- *Wood (table, blocks)*: the current `wood` Wave bump reads too stripey — replace
  with layered noise + subtle anisotropy + grain-driven roughness; vary base
  color. Make it look like finished/oiled wood.
- *Ceramics (mug, jug, vase, le creuset)*: glossy dielectric, slight roughness
  variation, subtle edge sheen; colored glaze where appropriate.
- *Metal (candle holder, grinder)*: proper metallic, anisotropic brushing or
  roughness noise, slight grime in crevices.
- *Glass (glass1)*: real transmission + roughness, correct IOR, maybe dispersion;
  ensure caustics/bounces make it read as glass, not plastic.
- *Organic (bread, banana, pepper)*: subsurface scattering tuned to scale; porous/
  bumpy skin via voronoi/noise; color variation (banana bruising, bread crust).
- *Stone (coaster)*: noise color + roughness variation, micro-displacement.
- Add **universal realism cues**: edge wear via geometry *Pointiness*, faint dust/
  smudge via procedural masks, and small base-color/roughness variation to break
  up flatness. These sell photorealism more than anything.

**Geometry shading**
- Shade-smooth with an auto-smooth angle (already applied) — verify no faceting on
  curved objects; consider a subdivision/displacement pass for surfaces that
  benefit (bread, fruit, stone). Check normals aren't inverted.

**Camera realism (optional, judgement call)**
- Subtle depth of field can add realism but may be undesirable for stimuli where
  all objects must be sharp. If you add it, keep it subtle and document it. Do not
  change FOV/position/sensor_fit (camera-match invariant).

---

## 4. Iteration loop

1. Pick a **representative review set** (~5–6 scenes) covering the hard cases:
   glass, metal, the wood table+blocks, fruit/bread (SSS), and a
   hidden-target/occluder scene. (Inspect `*_makeup.json` to find variety.)
2. Render them at a working quality (e.g. 256 samples) on GPU.
3. **Open each `render.png` with Read.** Critique against this checklist:
   - Does it read as a photo or as CG? What specifically gives it away?
   - Lighting believable? Soft shadows? Objects grounded with contact shadows?
   - Each material plausible for its object? Right glossiness/color/scale?
   - Reflections/refractions correct (glass, metal)?
   - Any noise, fireflies, banding, blown/crushed regions?
   - Backdrop present and natural?
4. Make targeted lookdev edits. Re-render the same set. Compare before/after
   (keep prior PNGs to diff visually).
5. Repeat until the review set meets the acceptance bar (below). Re-run the
   invariant tests whenever you touch render structure.
6. **Lock the look**, then render the **full dataset** (`--all`) at final quality.
   Spot-check a random sample of the outputs (beauty + segmentation overlay).

---

## 5. Acceptance criteria (when to stop)

- The review-set renders look like **photographs**; you can't trivially tell
  they're CG.
- **No visible render noise/fireflies** at final samples.
- Every object's material is plausible; nothing reads as flat untextured gray.
- Objects are **grounded** (contact shadows), on a real surface, not floating.
- `tests/test_camera_match.py` and `tests/test_render_pipeline.py` are **green**.
- Segmentation overlays still register exactly to the beauty renders.

---

## 6. Report back

- Summary of lookdev changes (lighting, per-material, render settings) and the
  final sample count / settings.
- Before/after sample renders for 2–3 scenes.
- Confirmation the full dataset rendered, with output locations and any scenes
  that failed/looked off.
- Any object whose material is a guess you couldn't validate (e.g. ambiguous
  `pepper`, `shark`, `bee`, `heart`) — flag for human confirmation.
- Note the wall-clock + per-scene render time at final settings (for planning
  re-renders).

> Reminder: render → **look at the image** → critique → adjust. The whole point of
> this task is visual quality, and you can see the images. Use that every loop.
