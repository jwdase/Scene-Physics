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
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt

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


def write_layout_usd(path: str, bodies: list[UsdBody], *, gravity: float = 9.81) -> str:
    """Write `bodies` to a Z-up physics USD at `path`; returns the path."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/root")

    scene = UsdPhysics.Scene.Define(stage, "/root/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(float(gravity))

    for body in bodies:
        prim = safe_usd_name(body.name)
        xform = UsdGeom.Xform.Define(stage, f"/root/{prim}")
        x, y, z, qx, qy, qz, qw = (float(v) for v in body.pose)
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(qw, qx, qy, qz))

        mesh = _add_mesh_geometry(stage, f"/root/{prim}/{prim}", body)
        _apply_physics(mesh.GetPrim(), body)

    stage.GetRootLayer().Save()
    return str(path)
