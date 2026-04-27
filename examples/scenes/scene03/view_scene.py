"""
Open an interactive Newton viewer on scene03 with no physics stepping.

The scene USD is visual-only (no UsdPhysics APIs), so `ModelBuilder.add_usd`
ignores its meshes. We walk the stage ourselves and add each mesh as a
kinematic shape (body=-1) at its USD world transform.
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
import newton.viewer


SCENE_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "objects", "scene03", "scene01.usdc",
)


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tris = []
    off = 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[off], indices[off + k], indices[off + k + 1]))
        off += c
    return np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def _xform_to_transform(mat: Gf.Matrix4d) -> wp.transform:
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    i = q.GetImaginary()
    return wp.transform((t[0], t[1], t[2]), (i[0], i[1], i[2], q.GetReal()))


def _add_usd_meshes(builder: newton.ModelBuilder, usd_path: str) -> int:
    stage = Usd.Stage.Open(usd_path)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    n = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        m = UsdGeom.Mesh(prim)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float32)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if pts.size == 0 or counts.size == 0:
            continue
        tris = _triangulate(counts, idx)
        mesh = newton.Mesh(pts, tris.reshape(-1))
        world = xcache.GetLocalToWorldTransform(prim)
        builder.add_shape_mesh(
            body=-1,
            xform=_xform_to_transform(world),
            mesh=mesh,
            key=str(prim.GetPath()),
        )
        n += 1
    return n


def main() -> None:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    added = _add_usd_meshes(builder, SCENE_USD)
    print(f"added {added} meshes from USD; total shapes = {builder.shape_count}")

    model = builder.finalize()
    state = model.state()

    viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=False)
    viewer.set_model(model)

    t = 0.0
    dt = 1.0 / 60.0
    try:
        while viewer.is_running():
            viewer.begin_frame(t)
            viewer.log_state(state)
            viewer.end_frame()
            t += dt
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
