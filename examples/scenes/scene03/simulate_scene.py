"""
Run a live physics simulation of scene03 in an interactive Newton viewer.

The USD has no physics APIs, so we walk the stage ourselves: the dining
table becomes a kinematic body (body=-1) and the iPhone, coffee, and soap
dispenser become dynamic bodies that fall under gravity and collide with
the table and ground.

After the simulation duration ends, the window stays open so you can orbit
around the resting configuration until you close it.
"""

import os

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, Gf

import newton
import newton.solvers
import newton.viewer


SCENE_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "objects", "scene03", "scene01.usdc",
)

STATIC_PRIM_NAMES = {"dining_room_table"}

DURATION_S = 5.0
FPS = 60
SUBSTEPS = 4


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tris = []
    off = 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[off], indices[off + k], indices[off + k + 1]))
        off += c
    return np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def _decompose(mat: Gf.Matrix4d) -> tuple[wp.transform, tuple[float, float, float]]:
    xf = Gf.Transform(mat)
    t = xf.GetTranslation()
    q = xf.GetRotation().GetQuat()
    i = q.GetImaginary()
    s = xf.GetScale()
    return (
        wp.transform((t[0], t[1], t[2]), (i[0], i[1], i[2], q.GetReal())),
        (s[0], s[1], s[2]),
    )


def _build() -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()

    stage = Usd.Stage.Open(SCENE_USD)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        m = UsdGeom.Mesh(prim)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float32)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if pts.size == 0 or counts.size == 0:
            continue

        mesh = newton.Mesh(pts, _triangulate(counts, idx).reshape(-1))
        world_xform, scale = _decompose(xcache.GetLocalToWorldTransform(prim))
        name = prim.GetParent().GetName()

        if name in STATIC_PRIM_NAMES:
            builder.add_shape_mesh(
                body=-1, xform=world_xform, mesh=mesh, scale=scale, key=str(prim.GetPath())
            )
        else:
            body = builder.add_body(xform=world_xform, key=name)
            builder.add_shape_mesh(
                body=body,
                xform=wp.transform(),
                mesh=mesh,
                scale=scale,
                key=str(prim.GetPath()),
            )

    return builder.finalize()


def main() -> None:
    model = _build()
    print(f"bodies={model.body_count}  shapes={model.shape_count}")

    solver = newton.solvers.SolverXPBD(model)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.collide(state_0)

    viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=False)
    viewer.set_model(model)

    dt = 1.0 / FPS
    sub_dt = dt / SUBSTEPS
    n_steps = int(FPS * DURATION_S)

    t = 0.0
    try:
        for i in range(n_steps):
            if not viewer.is_running():
                break
            for _ in range(SUBSTEPS):
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, None, contacts, sub_dt)
                state_0, state_1 = state_1, state_0

            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.end_frame()
            t += dt

        print("simulation finished — window stays open until closed")
        while viewer.is_running():
            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.end_frame()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
