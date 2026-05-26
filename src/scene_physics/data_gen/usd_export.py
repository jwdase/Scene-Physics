"""
Author a physics-layout `.usdc` that `newton.ModelBuilder.add_usd` can re-read.

The structure mirrors the hand-authored `scene01_physics.usdc`:

    /root                                Xform
    /root/PhysicsScene                   PhysicsScene (gravity -Z)
    /root/<name>                         Xform  (translate + orient hold the pose)
    /root/<name>/<name>                  Mesh   (geometry + physics schemas)

Dynamic bodies carry RigidBodyAPI + CollisionAPI + MeshCollisionAPI(convexHull)
+ MassAPI; static bodies (the table) carry CollisionAPI + MeshCollisionAPI(none)
only, which makes Newton treat them as fixed colliders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt

DEFAULT_MASS = 0.2


def safe_usd_name(name: str) -> str:
    """Coerce an object name into a valid USD prim identifier.

    USD prim names must match `[A-Za-z_][A-Za-z0-9_]*`, so names that start with
    a digit (e.g. `9v_battery`) get an underscore prefix. The mapping is
    idempotent, so it can be applied to already-safe names. The same name must
    be used as the JSON key for the object so the reloaded `body_key` matches.
    """
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "_" + s
    return s


@dataclass
class UsdBody:
    """One object to author into the layout USD."""

    name: str
    vertices: np.ndarray  # (N, 3) Z-up mesh points in the body's local frame
    indices: np.ndarray   # (3 * T,) triangle vertex indices
    pose: np.ndarray      # [x, y, z, qx, qy, qz, qw]
    is_static: bool
    mass: float = DEFAULT_MASS


def _add_mesh_geometry(stage: Usd.Stage, prim_path: str, body: UsdBody) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    verts = np.ascontiguousarray(body.vertices, dtype=np.float32)
    indices = np.ascontiguousarray(body.indices, dtype=np.int32)
    counts = np.full(indices.size // 3, 3, dtype=np.int32)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
    return mesh


def _apply_physics(prim: Usd.Prim, body: UsdBody) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)

    if body.is_static:
        # Full triangle mesh, no rigid body -> a fixed collider.
        mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.none)
        return

    mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(body.mass))


def _define_physics_material(
    stage: Usd.Stage,
    path: str,
    *,
    friction: float,
    restitution: float,
    rolling_friction: float | None = None,
    torsional_friction: float | None = None,
) -> UsdShade.Material:
    """Define a UsdPhysics rigid-body material at `path` and return it.

    Newton's `add_usd` reads `dynamicFriction` / `staticFriction` / `restitution` from the
    standard UsdPhysics material, plus rolling/torsional friction from the Newton-specific
    `newton:rollingFriction` / `newton:torsionalFriction` custom attributes (its default
    SchemaResolverNewton). Leaving those two unset makes the importer fall back to its
    defaults (0.0005 / 0.25) — exactly what scene_gen's *static* ShapeConfig leaves them at —
    so a static collider can omit them and still round-trip.
    """
    material = UsdShade.Material.Define(stage, path)
    prim = material.GetPrim()
    UsdPhysics.MaterialAPI.Apply(prim)
    api = UsdPhysics.MaterialAPI(prim)
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(float(restitution))
    if rolling_friction is not None:
        prim.CreateAttribute("newton:rollingFriction", Sdf.ValueTypeNames.Float).Set(
            float(rolling_friction)
        )
    if torsional_friction is not None:
        prim.CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(
            float(torsional_friction)
        )
    return material


def _bind_physics_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    """Bind `material` to `prim` for the physics purpose (the `material:binding:physics`
    relationship Newton's importer reads)."""
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material, materialPurpose="physics")


def write_layout_usd(
    path: str,
    bodies: list[UsdBody],
    *,
    gravity: float = 9.81,
    friction: float = 0.8,
    restitution: float = 0.0,
    rolling_friction: float = 0.1,
    torsional_friction: float = 0.1,
) -> str:
    """Write `bodies` to a Z-up physics USD at `path`; returns the path.

    The friction / restitution arguments author a UsdPhysics material so the shapes'
    contact properties round-trip through `add_usd` (otherwise reloaded shapes fall back to
    Newton's defaults). Defaults mirror the dataset generator's dynamic ShapeConfig; the
    static (table) material reuses `friction`/`restitution` and omits rolling/torsional so the
    importer's defaults apply, matching the generator's static ShapeConfig.
    """
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/root")

    scene = UsdPhysics.Scene.Define(stage, "/root/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(float(gravity))

    dyn_material = _define_physics_material(
        stage, "/root/PhysicsMaterialDynamic",
        friction=friction, restitution=restitution,
        rolling_friction=rolling_friction, torsional_friction=torsional_friction,
    )
    static_material = _define_physics_material(
        stage, "/root/PhysicsMaterialStatic",
        friction=friction, restitution=restitution,
    )

    for body in bodies:
        prim = safe_usd_name(body.name)
        xform = UsdGeom.Xform.Define(stage, f"/root/{prim}")
        x, y, z, qx, qy, qz, qw = (float(v) for v in body.pose)
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(qw, qx, qy, qz))

        mesh = _add_mesh_geometry(stage, f"/root/{prim}/{prim}", body)
        _apply_physics(mesh.GetPrim(), body)
        _bind_physics_material(
            mesh.GetPrim(), static_material if body.is_static else dyn_material
        )

    stage.GetRootLayer().Save()
    return str(path)
