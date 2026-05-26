"""
Drop-&-Settle scene generator.

Given a named occluded `target`, a role-split `pool` of candidate object names
(`mid_height` occluder candidates and `small` surrounders) and per-role counts,
each scene is produced by:

  1. Building a Z-up world: a static table, the target, a generator-picked
     occluder placed on the camera (-Y) side of the target, and sampled
     surrounders clustered for object-object contact.
  2. Forward-simulating XPBD until the bodies come to rest. The settled poses
     are the ground truth (rest is stable by construction).
  3. Validating: everything still on the table, at rest, and the target is
     actually occluded from the camera (full-scene vs target-only depth).
     Each candidate is an independent draw — failures are discarded, not retried
     (oversample-and-filter), so the generation loop is embarrassingly parallel.
  4. Emitting `<scene>_truth.json`, `<scene>_priors.json`,
     `<scene>_makeup.json`, and `<scene>_physics.usdc`, plus a `results/` dir.

The hidden target's prior mean is the occluder's center (x, y of the occluder
centroid, z at the table top); observed objects are centered on their own
settled truth. Bounds come from the table's XY extent.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverXPBD
from scipy.spatial.transform import Rotation

from scene_physics.data_gen.object_library import (
    LoadedObject,
    load_object,
    sample_objects,
)
from scene_physics.data_gen.usd_export import UsdBody, write_layout_usd, safe_usd_name
from scene_physics.simulation.sim_sampling import Camera, SingleWorldCamera, default_camera
from scene_physics.kernels.image_process import render_point_clouds_batch

# Physics / settling
GRAVITY = -9.81
DENSITY = 1000.0
MU = 0.8
RESTITUTION = 0.0
ROLL_FRICTION = 0.1     # damps endless rolling of cylindrical objects so they settle
TORSION_FRICTION = 0.1
SOLVER_ITERS = 32
SUBSTEPS = 8
DT = 1.0 / 60.0
SUB_DT = DT / SUBSTEPS
MIN_SETTLE_FRAMES = 40
MAX_SETTLE_FRAMES = 220
REST_CHECK_EVERY = 10
REST_THRESH = 0.08  # max |body spatial velocity| considered "at rest"

# Placement
PLACE_MARGIN = 0.10        # inset from table edge
TARGET_HALF_SPAN = 0.20    # target sampled within +/- this of table center
DROP_LIFT = 0.02           # initial clearance above the table
OCCLUDER_GAP = (0.0, 0.05)  # extra clearance beyond contact when spawning the occluder
# Occluder is drawn (not forced) from each scene's objects, weighted toward taller ones
# by height rank, so a shorter object is sometimes chosen and the object resting on the
# low target can show above it. 0 -> uniform (ignore height); large -> always the tallest.
OCCLUDER_HEIGHT_BIAS = 1.5
NEAR_RADIUS = 0.20         # surrounders placed "near" land within this of the target
P_NEAR = 0.5               # probability a surrounder is placed near the target
P_STACK = 1.0              # probability a surrounder is rested on the target (1.0 -> almost every scene gets an interaction)

# Resting orientation applied to the target before its random yaw. The wood-block
# target is authored standing on a thin edge; +90 deg about X lays it on its broad
# face (~7.7x7.7 cm, 3 cm tall) so a dropped object has a stable platform to rest on.
# Use identity ([0,0,0,1]) for a target that already sits flat (e.g. banana_fix2).
TARGET_BASE_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])  # XYZW

# A stack is accepted only as a genuine RESTING contact: the stacked object must be in
# true contact with the target AND sitting on its top face. The contact is queried with
# a tight margin because Newton's default contact margin (0.1 m) emits a "contact" for
# any pair within ~10 cm, which is far too loose to mean "touching".
STACK_CONTACT_MARGIN = 0.01   # metres; contact-query margin used as the "touching" threshold
STACK_REST_Z_TOL = 0.02       # metres; stacked base may sit at most this far below the target top
# Only small, squat surrounders are eligible to be stacked, so they settle on the
# narrow block top instead of toppling off.
STACK_MAX_FOOT = 0.06         # max footprint radius of a stackable surrounder
STACK_MAX_HEIGHT = 0.16       # max height of a stackable surrounder

# Occlusion
OCCLUSION_THRESH = 0.9     # fraction of the target silhouette that must be hidden
# Set high because the scenes are used to infer WHERE the occluded target can be: a
# barely-occluded target is mostly observed and carries little hidden-pose ambiguity.
DEPTH_MARGIN = 0.02        # metres; closer-than-this counts as an occluder in front
# The stacked object is the visible cue that couples (via physics) to the hidden
# target's pose, so it is useless if the occluder hides it too: require at least this
# fraction of its silhouette to remain visible in the full scene.
STACK_VISIBLE_THRESH = 0.4

# Default candidate budget per requested scene: generate_dataset draws up to
# n_scenes * CANDIDATE_BUDGET candidates so a low validation yield can't spin forever.
CANDIDATE_BUDGET = 25
# Worlds settled simultaneously per batch by generate_dataset_parallel (one GPU, no sharding).
# Size to GPU memory: each world holds the hull settle bodies plus, transiently, full-mesh
# render copies. ~64 is comfortable on an 80 GB H100; drop it on smaller cards.
DEFAULT_BATCH_WORLDS = 8
TABLE = "dining_room_table"


@dataclass
class SceneSpec:
    target: str
    pool: dict[str, list[str]]            # {"mid_height": [...], "small": [...]}
    n_mid: int | tuple[int, int] = 2      # mid-height objects per scene (incl. the occluder)
    n_small: int | tuple[int, int] = 3    # small surrounders per scene
    table: str = TABLE


@dataclass
class _Placement:
    name: str
    obj: LoadedObject
    pose: np.ndarray  # initial [x,y,z,qx,qy,qz,qw]


@dataclass
class SceneResult:
    name: str
    target: str
    occluder: str
    surrounders: list[str]
    occluded_fraction: float
    attempts: int
    paths: dict[str, str] = field(default_factory=dict)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def _xform(pose: np.ndarray) -> wp.transform:
    p = pose[:3]
    q = pose[3:]
    return wp.transform(
        (float(p[0]), float(p[1]), float(p[2])),
        wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
    )


def _world_center(pose: np.ndarray, local_center: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat(pose[3:])  # scipy expects [x,y,z,w]
    return rot.apply(local_center) + pose[:3]


def _canonical_pose(pose: np.ndarray) -> np.ndarray:
    """Flip the quaternion to the qw >= 0 hemisphere (newton's import convention),
    so truth.json matches the pose read back through add_usd bit-for-bit."""
    pose = np.asarray(pose, dtype=float).copy()
    if pose[6] < 0.0:
        pose[3:] = -pose[3:]
    return pose


def _initial_z(obj: LoadedObject, table_top: float, extra: float = 0.0) -> float:
    """Body-origin z so the (upright) mesh bottom clears the table by DROP_LIFT."""
    return table_top + DROP_LIFT + extra - obj.min_z


def _resolve_count(n: int | tuple[int, int], rng: np.random.Generator) -> int:
    if isinstance(n, tuple):
        return int(rng.integers(n[0], n[1] + 1))
    return int(n)


def _footprint_radius(obj: LoadedObject) -> float:
    return 0.5 * float(max(obj.extents[0], obj.extents[1]))


def _draw_occluder(picks, objs, rng) -> str:
    """Draw the occluder from `picks`, weighted toward taller objects by height rank
    (rank 0 = tallest), with OCCLUDER_HEIGHT_BIAS setting the strength (0 = uniform,
    large -> always the tallest). Drawing rather than forcing the tallest lets a shorter
    object occlude sometimes, so the object resting on the low target isn't always buried."""
    order = sorted(picks, key=lambda n: objs[n].height, reverse=True)  # tallest first
    k = len(order)
    scores = np.array([k - i for i in range(k)], dtype=float) ** OCCLUDER_HEIGHT_BIAS
    return str(rng.choice(order, p=scores / scores.sum()))


def _sample_placements(
    spec: SceneSpec, rng: np.random.Generator, table: LoadedObject
) -> tuple[list[_Placement], str, list[str]]:
    """Pick objects and non-overlapping initial poses for one attempt.

    Footprints are kept apart by rejection sampling so nothing starts
    interpenetrating (the main cause of XPBD blow-ups); the only intentional
    overlap is the optional stacked surrounder, which is dropped from above.
    """
    table_top = float(table.aabb_max[2])
    x_lo, y_lo = table.aabb_min[:2] + PLACE_MARGIN
    x_hi, y_hi = table.aabb_max[:2] - PLACE_MARGIN

    def clip(xy):
        return np.array([np.clip(xy[0], x_lo, x_hi), np.clip(xy[1], y_lo, y_hi)])

    n_mid = _resolve_count(spec.n_mid, rng)
    n_small = _resolve_count(spec.n_small, rng)
    if n_mid < 1:
        raise ValueError("n_mid must be >= 1: the occluder is drawn from the mid_height pool")
    mid_picks = sample_objects(n_mid, rng, pool=spec.pool["mid_height"], exclude=(spec.target,))
    small_picks = sample_objects(n_small, rng, pool=spec.pool["small"], exclude=(spec.target,))
    objs = {name: load_object(name) for name in (*mid_picks, *small_picks)}
    # Occluder is a height-weighted draw (see OCCLUDER_HEIGHT_BIAS) from the mid_height
    # picks only — it counts toward n_mid; the remaining mid + all small objects become
    # surrounders. Drawing rather than forcing the tallest lets a shorter mid object
    # occlude sometimes, so the object resting on the low target can show above it.
    occluder = _draw_occluder(mid_picks, objs, rng)
    surrounders = [p for p in (*mid_picks, *small_picks) if p != occluder]
    target_obj = load_object(spec.target)

    placed: list[tuple[np.ndarray, float]] = []  # (xy, footprint radius)

    def free(xy, r):
        return all(np.hypot(*(xy - pxy)) >= (r + pr) for pxy, pr in placed)

    def find_xy(sampler, r, tries=40):
        xy = clip(sampler())
        for _ in range(tries):
            if free(xy, r):
                break
            xy = clip(sampler())
        return xy

    placements: list[_Placement] = []

    def add(name, obj, xy, extra=0.0, reserve=True, base_quat=None):
        # base_quat is the object's resting orientation before the random yaw;
        # yaw is about Z, so it changes neither the drop height nor the footprint.
        if base_quat is None:
            z = _initial_z(obj, table_top, extra)
            quat = _yaw_quat(rng.uniform(0, 2 * np.pi))
            foot_r = _footprint_radius(obj)
        else:
            rest_v = Rotation.from_quat(base_quat).apply(obj.vertices)
            z = table_top + DROP_LIFT + extra - float(rest_v[:, 2].min())
            yaw = Rotation.from_rotvec([0.0, 0.0, rng.uniform(0, 2 * np.pi)])
            quat = (yaw * Rotation.from_quat(base_quat)).as_quat()
            foot_r = 0.5 * float(max(np.ptp(rest_v[:, 0]), np.ptp(rest_v[:, 1])))
        pose = np.concatenate([xy, [z], quat])
        placements.append(_Placement(name, obj, pose))
        if reserve:
            placed.append((np.asarray(xy, dtype=float), foot_r))

    # Target near the table centre, laid flat (TARGET_BASE_QUAT) for a broad top;
    # take its resting footprint and top height from the rotated mesh.
    target_rest_v = Rotation.from_quat(TARGET_BASE_QUAT).apply(target_obj.vertices)
    target_foot_r = 0.5 * float(max(np.ptp(target_rest_v[:, 0]), np.ptp(target_rest_v[:, 1])))
    target_top_h = float(np.ptp(target_rest_v[:, 2]))

    t_xy = find_xy(lambda: rng.uniform(-TARGET_HALF_SPAN, TARGET_HALF_SPAN, size=2),
                   target_foot_r)
    add(spec.target, target_obj, t_xy, base_quat=TARGET_BASE_QUAT)

    # Occluder just in front (-Y) of the target, near contact distance.
    r_o = _footprint_radius(objs[occluder])

    def occ_sampler():
        gap = target_foot_r + r_o + rng.uniform(*OCCLUDER_GAP)
        return t_xy + np.array([rng.uniform(-0.05, 0.05), -gap])

    add(occluder, objs[occluder], find_xy(occ_sampler, r_o))

    # Pick which surrounder (if any) to rest on the target: only small, squat objects
    # are eligible so they settle stably on the narrow block top rather than toppling.
    stackable = [n for n in surrounders
                 if _footprint_radius(objs[n]) <= STACK_MAX_FOOT and objs[n].height <= STACK_MAX_HEIGHT]
    stacked_name = str(rng.choice(stackable)) if (stackable and rng.random() < P_STACK) else None
    for name in surrounders:
        obj, r = objs[name], _footprint_radius(objs[name])
        if name == stacked_name:
            # Rest a visible object on the hidden target: coupling the stacked object
            # to the target is the probe for whether physics interactions are
            # informative about the hidden object's pose.
            add(name, obj, t_xy, extra=target_top_h + 0.03, reserve=False)
        elif rng.random() < P_NEAR:
            add(name, obj, find_xy(lambda: t_xy + rng.uniform(-NEAR_RADIUS, NEAR_RADIUS, size=2), r))
        else:
            add(name, obj, find_xy(lambda: np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)]), r))

    return placements, occluder, surrounders, stacked_name


def _add_table_and_dynamics(builder, table: LoadedObject, placements: list[_Placement]) -> dict:
    """Add the box table (static) + convex-hull dynamics to `builder`; return name -> body index.

    No ground plane is added here, so this works both for a single-world model (caller adds
    its own ground) and for a per-world sub-builder fed to `add_world` (the main builder owns
    one shared ground). Body indices are local to `builder` at call time.
    """
    static_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU, restitution=RESTITUTION)
    # Settle against a solid box spanning the table's AABB rather than the visual mesh.
    # The table .obj is a thin shell, so under the stacking impact a dropped object can
    # overshoot the surface and tunnel through it; a solid box gives an un-tunnelable
    # contact volume. Its top face is at aabb_max[2] (the mesh's top), so rest heights
    # match the full-mesh table that rendering and the exported USD still use.
    half = 0.5 * (table.aabb_max - table.aabb_min)
    center = 0.5 * (table.aabb_max + table.aabb_min)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(
            (float(center[0]), float(center[1]), float(center[2])), wp.quat_identity()
        ),
        hx=float(half[0]), hy=float(half[1]), hz=float(half[2]),
        cfg=static_cfg,
    )

    dyn_cfg = newton.ModelBuilder.ShapeConfig(
        density=DENSITY, mu=MU, restitution=RESTITUTION,
        rolling_friction=ROLL_FRICTION, torsional_friction=TORSION_FRICTION,
    )
    index = {}
    for pl in placements:
        bi = builder.add_body(xform=_xform(pl.pose), key=pl.name)
        builder.add_shape_convex_hull(body=bi, mesh=pl.obj.mesh, cfg=dyn_cfg)
        index[pl.name] = bi
    return index


def _build_model(table: LoadedObject, placements: list[_Placement]):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
    builder.add_ground_plane()
    index = _add_table_and_dynamics(builder, table, placements)
    return builder.finalize(), index


def _build_batched_model(table: LoadedObject, batch: list[list[_Placement]]):
    """Allocate one Newton model holding `len(batch)` independent scenes as separate worlds.

    Each scene becomes a collision-isolated world via `add_world` (all stacked at the origin —
    isolation is by world index, not spacing, which keeps the solver numerically stable). One
    shared global ground plane sits under every world. Returns the finalized model and a list
    of per-world `{name: global_body_index}` dicts (the global index is the per-world body
    offset, captured before each `add_world`, plus the body's local index).
    """
    main = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
    main.add_ground_plane()

    indices = []
    for placements in batch:
        sub = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
        local = _add_table_and_dynamics(sub, table, placements)
        base = main.body_count
        main.add_world(sub)
        indices.append({name: base + bi for name, bi in local.items()})

    return main.finalize(), indices


def _settle_loop(model):
    """Forward-simulate XPBD until rest (or MAX_SETTLE_FRAMES); return the final State.

    The SUBSTEPS-long substep loop is captured once as a CUDA graph and replayed once per
    frame, so the ~MAX_SETTLE_FRAMES*SUBSTEPS individual collide/solve kernel launches collapse
    into one graph launch per frame. On a fast GPU the settle was launch-overhead bound (each
    kernel is tiny but every launch pays host->device latency), so the replay is the main win.
    Rest is polled on the host only every REST_CHECK_EVERY frames, between launches, so the hot
    path carries no GPU->CPU sync.

    Works for a single-world or a batched (add_world) model alike: when every body's spatial
    velocity is below REST_THRESH the global max trips the early-out, so a single slow world
    just extends the sim for the whole batch. Off CUDA it falls back to an eager loop.
    """
    s0, s1 = model.state(), model.state()
    solver = SolverXPBD(model, iterations=SOLVER_ITERS)
    control = model.control()

    def substep_frame():
        nonlocal s0, s1
        for _ in range(SUBSTEPS):
            s0.clear_forces()
            contacts = model.collide(s0)
            solver.step(s0, s1, control, contacts, SUB_DT)
            s0, s1 = s1, s0

    use_graph = wp.get_device().is_cuda
    graph = None
    for frame in range(MAX_SETTLE_FRAMES):
        if not use_graph:
            substep_frame()
        elif graph is None:
            # First frame: run eagerly to compile kernels and allocate the contact buffers,
            # then capture the same frame for replay. SUBSTEPS is even, so the state buffers
            # alias back to (s0, s1) after each frame and the captured launch sequence stays
            # valid on every replay. Capture records without executing, so it adds no frame.
            substep_frame()
            with wp.ScopedCapture() as capture:
                substep_frame()
            graph = capture.graph
        else:
            wp.capture_launch(graph)
        if frame >= MIN_SETTLE_FRAMES and (frame + 1) % REST_CHECK_EVERY == 0:
            if np.abs(s0.body_qd.numpy()).max() < REST_THRESH:
                break
    return s0


def _settle(model):
    s0 = _settle_loop(model)
    at_rest = float(np.abs(s0.body_qd.numpy()).max()) < REST_THRESH
    return s0, s0.body_q.numpy(), at_rest


def _settle_batched(model):
    """Settle every world in a batched model with one shared (graph-captured) XPBD loop.

    Per-world rest is decided later by slicing body_qd with each world's body indices.
    Returns the final state, the flat settled body_q, and the flat settled body_qd.
    """
    s0 = _settle_loop(model)
    return s0, s0.body_q.numpy(), s0.body_qd.numpy()


def _contact_pair_exists(model, contacts, body_a: int, body_b: int) -> bool:
    """True if `contacts` (from model.collide) include a rigid contact between the two
    bodies. A contact is emitted for any pair within model.shape_contact_margin, so
    tighten that margin before colliding if you need 'actually touching' rather than
    'nearby' (Newton's 0.1 m default reports a contact up to ~10 cm apart)."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return False
    shape_body = model.shape_body.numpy()
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    want = {body_a, body_b}
    return any(
        a >= 0 and b >= 0 and {int(shape_body[a]), int(shape_body[b])} == want
        for a, b in zip(s0, s1)
    )


def _world_z_bounds(pose: np.ndarray, obj: LoadedObject) -> tuple[float, float]:
    """(min, max) world z of the object's vertices at `pose`."""
    vw = Rotation.from_quat(pose[3:]).apply(obj.vertices) + pose[:3]
    return float(vw[:, 2].min()), float(vw[:, 2].max())


def _rests_on_target(stacked_pose, stacked_obj, target_pose, target_obj) -> bool:
    """True if the stacked object sits on the target's top face: its lowest point is no
    more than STACK_REST_Z_TOL below the target's highest point, so it is resting on top
    rather than standing on the table beside the target."""
    _, target_top = _world_z_bounds(target_pose, target_obj)
    stacked_bottom, _ = _world_z_bounds(stacked_pose, stacked_obj)
    return stacked_bottom >= target_top - STACK_REST_Z_TOL


def _render_depth_of(items, intrinsics, table=None) -> np.ndarray:
    """Render a depth image of `items` (obj, pose) using full visual meshes.

    Physics settles on convex hulls, but the ray-trace sensor only renders
    triangle meshes (and the downstream point cloud renders the visual mesh
    too), so depth is taken from a mesh-built copy of the scene at the settled
    poses.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
    if table is not None:
        builder.add_shape_mesh(
            body=-1, xform=wp.transform((0, 0, 0), wp.quat_identity()),
            mesh=table.mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU, restitution=RESTITUTION),
        )
    cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY, mu=MU, restitution=RESTITUTION)
    for obj, pose in items:
        bi = builder.add_body(xform=_xform(pose))
        builder.add_shape_mesh(body=bi, mesh=obj.mesh, cfg=cfg)

    model = builder.finalize()
    camera = SingleWorldCamera(intrinsics, model)
    camera.render(model.state())
    h, w = intrinsics.height, intrinsics.width
    return camera.depth_image.numpy()[0, 0].reshape(h, w)


def _occluded_fraction(placements, settled, index, name, table, intrinsics):
    """Fraction of `name`'s camera silhouette hidden by something nearer in the full
    scene (full-scene depth vs the object rendered alone)."""
    full_items = [(pl.obj, settled[index[pl.name]]) for pl in placements]
    depth_full = _render_depth_of(full_items, intrinsics, table=table)

    obj_pl = next(pl for pl in placements if pl.name == name)
    depth_obj = _render_depth_of([(obj_pl.obj, settled[index[name]])], intrinsics)

    max_depth = intrinsics.max_depth
    obj_mask = (depth_obj > 0) & (depth_obj < max_depth)
    n_obj = int(obj_mask.sum())
    if n_obj == 0:
        return 0.0
    occluded = obj_mask & (depth_full > 0) & (depth_full < depth_obj - DEPTH_MARGIN)
    return float(occluded.sum()) / n_obj


def _visible_fraction(placements, settled, index, name, table, intrinsics):
    """Fraction of `name`'s silhouette that stays visible (not occluded) in the full scene."""
    return 1.0 - _occluded_fraction(placements, settled, index, name, table, intrinsics)


def _occ_frac(depth_full: np.ndarray, depth_obj: np.ndarray, max_depth: float) -> float:
    """Fraction of the object's silhouette (depth_obj) hidden by something nearer in the
    full-scene depth — the per-world building block of the occlusion / visibility gates."""
    mask = (depth_obj > 0) & (depth_obj < max_depth)
    n = int(mask.sum())
    if n == 0:
        return 0.0
    occluded = mask & (depth_full > 0) & (depth_full < depth_obj - DEPTH_MARGIN)
    return float(occluded.sum()) / n


def _render_depth_batch(table, batch, settled, indices, intrinsics, *, include_table, select):
    """Render one depth image per world for a batch, using full visual meshes.

    Each world is built as its own `add_world` sub-model containing (optionally) the table
    mesh plus the (obj, pose) items returned by `select(world_idx, placements_w, index_w)`.
    Returns a list of (H, W) depth arrays, one per world. Settling happens on convex hulls
    elsewhere; the ray-trace sensor needs triangle meshes, so depth is taken from this
    mesh-built copy posed at the settled `body_q`.
    """
    static_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU, restitution=RESTITUTION)
    dyn_cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY, mu=MU, restitution=RESTITUTION)

    main = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
    for w, placements_w in enumerate(batch):
        sub = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
        if include_table:
            sub.add_shape_mesh(
                body=-1, xform=wp.transform((0, 0, 0), wp.quat_identity()),
                mesh=table.mesh, cfg=static_cfg,
            )
        for obj, pose in select(w, placements_w, indices[w]):
            bi = sub.add_body(xform=_xform(pose))
            sub.add_shape_mesh(body=bi, mesh=obj.mesh, cfg=dyn_cfg)
        main.add_world(sub)

    model = main.finalize()
    n = len(batch)
    cam = Camera(intrinsics, model, num_worlds=n)
    h, w = intrinsics.height, intrinsics.width
    render_point_clouds_batch(
        cam.sensor, model.state(), cam.camera_transforms, cam.camera_rays,
        cam.depth_image, cam.points_gpu, h, w, intrinsics.max_depth, n,
    )
    depth = cam.depth_image.numpy()
    return [depth[i, 0].reshape(h, w) for i in range(n)]


def _on_table(pose, obj, table) -> bool:
    c = _world_center(pose, obj.center)
    top = float(table.aabb_max[2])
    return (
        table.aabb_min[0] <= c[0] <= table.aabb_max[0]
        and table.aabb_min[1] <= c[1] <= table.aabb_max[1]
        and (top - 0.05) <= c[2] <= (top + 0.6)
    )


def _write_artifacts(scene_dir, name, table, placements, settled, index,
                     masses, occluder, surrounders, target):
    data_dir = scene_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "results").mkdir(parents=True, exist_ok=True)

    table_top = float(table.aabb_max[2])
    pose_of = {pl.name: _canonical_pose(settled[index[pl.name]]) for pl in placements}
    S = safe_usd_name  # object names must be valid USD prim identifiers

    # truth.json — table first (fixed at origin), then every dynamic object.
    truth = {S(table.name): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
    for pl in placements:
        truth[S(pl.name)] = pose_of[pl.name].tolist()

    # priors.json
    occ_center = _world_center(pose_of[occluder], load_object(occluder).center)
    bounds = {
        "x_max": float(table.aabb_max[0]), "x_min": float(table.aabb_min[0]),
        "y_max": float(table.aabb_max[1]), "y_min": float(table.aabb_min[1]),
    }
    priors = {}
    for pl in placements:
        if pl.name == target:
            mean = [float(occ_center[0]), float(occ_center[1]), table_top, 0.0, 0.0, 0.0, 1.0]
        else:
            mean = pose_of[pl.name].tolist()
        priors[S(pl.name)] = {"position": mean, "pos_std": 0.1, "rot_std": 0.1, **bounds}

    makeup = {
        "static": [S(table.name)], "observed": [S(occluder)] + [S(s) for s in surrounders],
        "hidden": [S(target)], "occluder": S(occluder), "target": S(target),
    }

    bodies = [UsdBody(S(table.name), table.vertices, table.indices,
                      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), is_static=True)]
    for pl in placements:
        bodies.append(UsdBody(S(pl.name), pl.obj.vertices, pl.obj.indices,
                              pose_of[pl.name], is_static=False,
                              mass=float(masses[index[pl.name]])))

    paths = {
        "truth": str(data_dir / f"{name}_truth.json"),
        "priors": str(data_dir / f"{name}_priors.json"),
        "makeup": str(data_dir / f"{name}_makeup.json"),
        "usd": str(data_dir / f"{name}_physics.usdc"),
    }
    for key, payload in (("truth", truth), ("priors", priors), ("makeup", makeup)):
        with open(paths[key], "w") as f:
            json.dump(payload, f, indent=4)
    write_layout_usd(paths["usd"], bodies, gravity=abs(GRAVITY),
                     friction=MU, restitution=RESTITUTION,
                     rolling_friction=ROLL_FRICTION, torsional_friction=TORSION_FRICTION)
    return paths


def _attempt_scene(spec, rng, table, scene_dir, name, attempt, intrinsics):
    """One Drop-&-Settle attempt; returns a SceneResult on success else None.

    All heavy GPU objects (models, solver, cameras) stay local to this frame so
    they are released between attempts — the caller forces a gc afterwards.
    """
    placements, occluder, surrounders, stacked_name = _sample_placements(spec, rng, table)
    model, index = _build_model(table, placements)
    state, settled, at_rest = _settle(model)

    if not at_rest:
        return None
    if not all(_on_table(settled[index[pl.name]], pl.obj, table) for pl in placements):
        return None

    # In a stack scene, keep only a genuine resting interaction: with a tightened contact
    # margin (Newton's 0.1 m default would pass objects ~10 cm apart) the stacked object
    # must be in true contact with the target AND sitting on its top face.
    if stacked_name is not None:
        pl_by_name = {pl.name: pl for pl in placements}
        model.shape_contact_margin.fill_(STACK_CONTACT_MARGIN)
        contacts = model.collide(state)
        if not _contact_pair_exists(model, contacts, index[spec.target], index[stacked_name]):
            return None
        if not _rests_on_target(settled[index[stacked_name]], pl_by_name[stacked_name].obj,
                                settled[index[spec.target]], pl_by_name[spec.target].obj):
            return None
        # The stacked object must remain visible to be a usable cue to the hidden pose.
        if _visible_fraction(placements, settled, index, stacked_name, table, intrinsics) < STACK_VISIBLE_THRESH:
            return None

    frac = _occluded_fraction(placements, settled, index, spec.target, table, intrinsics)
    if frac < OCCLUSION_THRESH:
        return None

    masses = model.body_mass.numpy()
    paths = _write_artifacts(Path(scene_dir), name, table, placements, settled,
                             index, masses, occluder, surrounders, spec.target)
    return SceneResult(name, spec.target, occluder, surrounders, frac, attempt, paths)


def generate_scene(spec, rng, scene_dir, name, intrinsics=default_camera, candidate=1):
    """Generate ONE candidate scene (a single Drop-&-Settle attempt).

    Returns a SceneResult if it passes validation, else None. There is no retry: a
    rejected candidate is simply discarded, so each call is independent and parallel-safe.
    Artifacts are written (under scene_dir) only on success.
    """
    table = load_object(spec.table)
    result = _attempt_scene(spec, rng, table, scene_dir, name, candidate, intrinsics)
    gc.collect()
    return result


def generate_dataset(spec, n_scenes, out_root, seed=0, intrinsics=default_camera,
                     start_index=1, max_candidates=None):
    """Oversample-and-filter generation (no per-scene retry).

    Draw independent candidate scenes and save only the ones that pass validation, until
    `n_scenes` good scenes are written or `max_candidates` candidates have been tried.
    Every candidate is an independent draw, so this is embarrassingly parallel: launch
    many cluster jobs with distinct `seed`s (each writing to its own `out_root`) and merge
    the outputs. `start_index` offsets the saved scene numbers if you instead want several
    runs to share one directory without colliding.
    """
    rng = np.random.default_rng(seed)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if max_candidates is None:
        max_candidates = n_scenes * CANDIDATE_BUDGET

    table = load_object(spec.table)
    stats_path = out_root / "scene_stats.txt"
    results = []
    candidates = 0
    with open(stats_path, "w") as stats:
        stats.write(f"# target={spec.target}\tocclusion_thresh={OCCLUSION_THRESH:.2f}\tseed={seed}\n")
        stats.write("# scene\toccluded_fraction\toccluder\tcandidate\tsurrounders\n")
        stats.flush()
        while len(results) < n_scenes and candidates < max_candidates:
            candidates += 1
            name = f"scene{start_index + len(results):03d}"
            res = generate_scene(spec, rng, out_root / name, name, intrinsics, candidate=candidates)
            if res is None:
                continue
            results.append(res)
            print(f"[{name}] occluder={res.occluder} surrounders={res.surrounders} "
                  f"occluded={res.occluded_fraction:.2f} candidate={candidates}")
            stats.write(f"{name}\t{res.occluded_fraction:.4f}\t{res.occluder}\t"
                        f"{candidates}\t{','.join(res.surrounders)}\n")
            stats.flush()
        yield_pct = len(results) / candidates if candidates else 0.0
        stats.write(f"# saved {len(results)}/{n_scenes} good scenes from {candidates} "
                    f"candidates (yield {yield_pct:.1%})\n")

    print(f"\nSaved {len(results)}/{n_scenes} good scenes from {candidates} candidates in {out_root}")
    print(f"Stats written to {stats_path}")
    return results


def _pl_by_name(placements, name):
    return next(pl for pl in placements if pl.name == name)


def _validate_world(w, target, batch, metas, settled, qd, indices, table, model, contacts,
                    depth_full, depth_tgt, depth_stk, max_depth):
    """Return the target's occluded fraction if world `w` passes every gate, else None.

    Same gates as the single-world path (at-rest, on-table, resting-stack contact + top-face
    + visibility, and target occlusion), evaluated against per-world slices of the batched
    state and the batched depth renders.
    """
    idx, pls = indices[w], batch[w]
    stacked = metas[w][2]

    if np.abs(np.array([qd[idx[pl.name]] for pl in pls])).max() >= REST_THRESH:
        return None
    if not all(_on_table(settled[idx[pl.name]], pl.obj, table) for pl in pls):
        return None

    if stacked is not None:
        pbn = {pl.name: pl for pl in pls}
        if not _contact_pair_exists(model, contacts, idx[target], idx[stacked]):
            return None
        if not _rests_on_target(settled[idx[stacked]], pbn[stacked].obj,
                                settled[idx[target]], pbn[target].obj):
            return None
        if (1.0 - _occ_frac(depth_full[w], depth_stk[w], max_depth)) < STACK_VISIBLE_THRESH:
            return None

    frac = _occ_frac(depth_full[w], depth_tgt[w], max_depth)
    if frac < OCCLUSION_THRESH:
        return None
    return frac


def generate_dataset_parallel(spec, n_scenes, out_root, num_worlds=DEFAULT_BATCH_WORLDS,
                              seed=0, intrinsics=default_camera, start_index=1,
                              max_candidates=None):
    """Single-GPU batched oversample-and-filter generation.

    Allocates `num_worlds` Newton worlds at once (one independent candidate scene per world),
    settles them all in a single XPBD pass, validates each world, and writes the passers —
    repeating batches until `n_scenes` good scenes are saved (or the candidate budget is hit).
    No per-scene retry and no job sharding: one process drives many worlds on one GPU. Heavy
    GPU objects are localized per batch and gc'd, so memory is bounded by `num_worlds`.
    """
    rng = np.random.default_rng(seed)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if max_candidates is None:
        max_candidates = n_scenes * CANDIDATE_BUDGET
    table = load_object(spec.table)
    target = spec.target
    max_depth = intrinsics.max_depth

    print(f"[generate_dataset_parallel] allocating {num_worlds} worlds/batch on "
          f"{wp.get_device()} -> {n_scenes} scenes (candidate budget {max_candidates})")

    stats_path = out_root / "scene_stats.txt"
    results = []
    candidates = 0
    round_idx = 0
    with open(stats_path, "w") as stats:
        stats.write(f"# target={spec.target}\tocclusion_thresh={OCCLUSION_THRESH:.2f}"
                    f"\tseed={seed}\tnum_worlds={num_worlds}\n")
        stats.write("# scene\toccluded_fraction\toccluder\tcandidate\tsurrounders\n")
        stats.flush()

        while len(results) < n_scenes and candidates < max_candidates:
            round_idx += 1
            good_before = len(results)
            bs = min(num_worlds, max_candidates - candidates)

            # 1. Sample bs independent candidate scenes.
            batch, metas = [], []
            for _ in range(bs):
                placements, occluder, surrounders, stacked = _sample_placements(spec, rng, table)
                batch.append(placements)
                metas.append((occluder, surrounders, stacked))
            candidates += bs

            # 2. Allocate every world and settle them together.
            model, indices = _build_batched_model(table, batch)
            state, settled, qd = _settle_batched(model)
            masses = model.body_mass.numpy()

            # 3. Resting-contact query: tighten the margin and re-collide once for the batch.
            model.shape_contact_margin.fill_(STACK_CONTACT_MARGIN)
            contacts = model.collide(state)

            # 4. Batched depth renders: full scene, target-only, stacked-only (filler=target
            #    for no-stack worlds so no world is empty; their stack gate is skipped).
            def sel_full(w, pls, idx):
                return [(pl.obj, settled[idx[pl.name]]) for pl in pls]

            def sel_target(w, pls, idx):
                return [(_pl_by_name(pls, target).obj, settled[idx[target]])]

            def sel_stacked(w, pls, idx):
                name = metas[w][2] or target
                return [(_pl_by_name(pls, name).obj, settled[idx[name]])]

            depth_full = _render_depth_batch(table, batch, settled, indices, intrinsics,
                                             include_table=True, select=sel_full)
            depth_tgt = _render_depth_batch(table, batch, settled, indices, intrinsics,
                                            include_table=False, select=sel_target)
            depth_stk = _render_depth_batch(table, batch, settled, indices, intrinsics,
                                            include_table=False, select=sel_stacked)

            # 5. Per-world gates; write each passer as the next sceneNNN.
            for w in range(bs):
                if len(results) >= n_scenes:
                    break
                frac = _validate_world(w, target, batch, metas, settled, qd, indices, table,
                                       model, contacts, depth_full, depth_tgt, depth_stk, max_depth)
                if frac is None:
                    continue
                occluder, surrounders, _ = metas[w]
                name = f"scene{start_index + len(results):03d}"
                paths = _write_artifacts(out_root / name, name, table, batch[w], settled,
                                         indices[w], masses, occluder, surrounders, spec.target)
                results.append(SceneResult(name, spec.target, occluder, surrounders, frac,
                                           candidates, paths))
                print(f"[{name}] occluder={occluder} surrounders={surrounders} "
                      f"occluded={frac:.2f} candidate={candidates}")
                stats.write(f"{name}\t{frac:.4f}\t{occluder}\t{candidates}\t{','.join(surrounders)}\n")
                stats.flush()

            # 6. Release this batch's GPU objects before the next one.
            del model, state, contacts, depth_full, depth_tgt, depth_stk
            gc.collect()

            print(f"[round {round_idx}] settled {bs} worlds, kept {len(results) - good_before} "
                  f"-> {len(results)}/{n_scenes} scenes ({candidates} candidates tried)")

        yield_pct = len(results) / candidates if candidates else 0.0
        stats.write(f"# saved {len(results)}/{n_scenes} good scenes from {candidates} "
                    f"candidates (yield {yield_pct:.1%})\n")

    print(f"\nSaved {len(results)}/{n_scenes} good scenes from {candidates} candidates in {out_root}")
    print(f"Stats written to {stats_path}")
    return results


if __name__ == "__main__":
    # Hidden target is square_wood_block, laid flat by TARGET_BASE_QUAT into a ~7.7x7.7 cm,
    # 3 cm-tall platform; almost every scene rests a small surrounder on it (P_STACK=1.0)
    # as the physics-interaction probe.
    #
    # The pool is split by role: "mid_height" objects (~13-27 cm) are the occluder
    # candidates for the low block; "small" objects are the surrounders, several squat
    # enough to be stacked on the block. n_mid=2, n_small=3 -> 6 placed objects/scene
    # (target + 2 mid [one is the occluder] + 3 small). The occluder is a height-weighted
    # draw from the sampled mid_height objects (see OCCLUDER_HEIGHT_BIAS), so a shorter one
    # sometimes occludes and the stacked object can show above it instead of being buried.
    pool = {
        "mid_height": [
            "jug04", "bee", "glass1", "int_kitchen_accessories_le_creuset_bowl_30cm",
            "b05_coffee_grinder",        # ~18 cm (rescaled from native 1.17 m)
            "b04_candle_holder_metal",   # ~22 cm
            "vase_05",                   # ~27 cm
        ],
        "small": [
            "b03_loafbread", "bung", "pepper", "coffeemug",
            "shark", "heart", "banana_fix2", "star_wood_block",
            "round_coaster_stone", "f10_apple_iphone_4",
        ],
    }
    spec = SceneSpec(target="square_wood_block", pool=pool, n_mid=2, n_small=3)
    # Anchor the output to src/scene_physics/generated_scenes regardless of the directory
    # the script is launched from (a relative path lands wherever cwd happens to be, which
    # silently scatters datasets — and the viewer reads this canonical location).
    out_root = Path(__file__).resolve().parents[1] / "generated_scenes"
    # Single-GPU batched generation: settle num_worlds candidate scenes at once and keep the
    # passers (no per-scene retry, no job sharding). Size num_worlds to the GPU (DEFAULT_BATCH_WORLDS
    # ~64 suits an 80 GB H100; lower it on smaller cards).
    generate_dataset_parallel(spec, n_scenes=100, out_root=out_root, num_worlds=DEFAULT_BATCH_WORLDS, seed=0)
