"""
Re-pose objects in an existing layout `.usdc` and save a copy.

A layout USD (authored by `usd_export.write_layout_usd`) stores each object's
pose entirely in the `translate` + `orient` ops on its `/root/<name>` Xform; the
geometry and physics schemas live on the child mesh and are pose-independent.
That means a new placement can be written by overwriting just those two ops — no
need to re-run physics or rebuild meshes.

Typical use: after a sampling run, write the recovered poses into a copy of the
scene's original USD for rendering / inspection.

    from scene_physics.data_gen.usd_repose import repose_usd

    repose_usd(
        "generated_scenes/scene001/data/scene001_physics.usdc",
        {"square_wood_block": [x, y, z, qx, qy, qz, qw]},
        out_path="generated_scenes/scene001/results/scene001_sampled.usdc",
    )

Pose convention matches the rest of the codebase: `[x, y, z, qx, qy, qz, qw]`
(XYZW). The USD orient op is authored w-first as `Gf.Quatf(qw, qx, qy, qz)`,
exactly as `write_layout_usd` does, so the copy round-trips through `add_usd`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from pxr import Usd, UsdGeom, Gf

from scene_physics.data_gen.usd_export import safe_usd_name


def _set_pose(prim: Usd.Prim, pose: Sequence[float]) -> None:
    """Overwrite the translate + orient ops on `prim` with `pose`.

    `pose` is `[x, y, z, qx, qy, qz, qw]`. Reuses the existing ops when present
    (preserving op order), otherwise authors them in the same order
    `write_layout_usd` uses.
    """
    x, y, z, qx, qy, qz, qw = (float(v) for v in pose)
    xform = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}

    if "xformOp:translate" in ops:
        ops["xformOp:translate"].Set(Gf.Vec3d(x, y, z))
    else:
        UsdGeom.Xform(prim).AddTranslateOp().Set(Gf.Vec3d(x, y, z))

    if "xformOp:orient" in ops:
        ops["xformOp:orient"].Set(Gf.Quatf(qw, qx, qy, qz))
    else:
        UsdGeom.Xform(prim).AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Quatf(qw, qx, qy, qz)
        )


def repose_usd(
    src_path: str,
    new_poses: Mapping[str, Sequence[float]],
    out_path: str | None = None,
    *,
    strict: bool = True,
) -> str:
    """Write a copy of `src_path` with the named objects re-posed.

    Args:
        src_path:   the original layout `.usdc` to read.
        new_poses:  `{object_name: [x, y, z, qx, qy, qz, qw]}`. Names are matched
                    through `safe_usd_name`, so the raw object names (the truth /
                    makeup keys) work directly. Objects not listed keep their
                    original pose.
        out_path:   where to write the copy. If `None`, the source is overwritten
                    in place.
        strict:     if True (default), raise when a requested name is missing from
                    the stage; if False, silently skip it.

    Returns:
        The path the result was written to.
    """
    stage = Usd.Stage.Open(str(src_path))
    if stage is None:
        raise FileNotFoundError(f"Could not open USD stage: {src_path}")

    for name, pose in new_poses.items():
        prim = stage.GetPrimAtPath(f"/root/{safe_usd_name(name)}")
        if not prim or not prim.IsValid():
            if strict:
                raise KeyError(f"{name!r} (/root/{safe_usd_name(name)}) not in {src_path}")
            continue
        _set_pose(prim, np.asarray(pose, dtype=float).reshape(7))

    if out_path is None:
        stage.GetRootLayer().Save()
        return str(src_path)

    stage.GetRootLayer().Export(str(out_path))
    return str(out_path)
