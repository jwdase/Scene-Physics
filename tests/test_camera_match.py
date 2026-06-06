"""Newton pinhole camera vs Blender camera -- are the two views identical?

The renders that go to study participants are produced by Blender, but the rest
of the pipeline (point cloud, occlusion gates, likelihood) sees the scene through
Newton's `SensorTiledCamera`. If the two cameras disagree, the stimulus and the
geometry it is paired with are misaligned. These tests pin the equivalence down
in layers:

  (a) numpy Newton projection  ==  Blender `world_to_camera_view`   [needs Blender]
  (b) numpy Newton projection  ==  the REAL Newton sensor           [gpu]
  (c) Newton sensor render     ==  Blender render  (silhouette IoU)  [gpu + Blender]

(a)+(b) chain to "real sensor == Blender camera" analytically; (c) confirms it
end-to-end through an actual render.

  uv run pytest tests/test_camera_match.py -v -s            # (a)
  uv run pytest tests/test_camera_match.py -v -s -m gpu     # (b),(c)
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from scene_physics.configs.camera import default_camera
from scene_physics.visualization import newton_projection as npj

HAVE_BLENDER = bool(shutil.which("blender"))
blender_only = pytest.mark.skipif(not HAVE_BLENDER, reason="Blender not installed")

W, H = default_camera.width, default_camera.height


def _tabletop_grid() -> np.ndarray:
    """A grid of points spanning the tabletop working volume (Z-up)."""
    xs = np.linspace(-0.45, 0.45, 7)
    ys = np.linspace(-0.45, 0.45, 7)
    zs = np.linspace(0.40, 1.05, 4)
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def _in_frame(px: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(px[:, 0])
        & np.isfinite(px[:, 1])
        & (px[:, 0] >= 0)
        & (px[:, 0] <= W - 1)
        & (px[:, 1] >= 0)
        & (px[:, 1] <= H - 1)
    )


@blender_only
def test_numpy_model_matches_blender():
    """The numpy Newton projection and Blender's camera land points on the same
    pixels (sub-pixel) for every in-frame tabletop point."""
    from scene_physics.visualization import blender_runner

    pts = _tabletop_grid()
    p_newton = npj.project_points(pts, default_camera)
    p_blender = blender_runner.project_points(default_camera, pts)

    mask = _in_frame(p_newton) & np.isfinite(p_blender).all(axis=1)
    assert mask.sum() >= 20, f"too few in-frame test points: {mask.sum()}"

    diff = np.abs(p_newton[mask] - p_blender[mask])
    print(
        f"\n[a] in-frame pts={mask.sum()}  max|Δpx|={diff.max():.4f}  "
        f"mean|Δpx|={diff.mean():.4f}"
    )
    assert diff.max() < 0.5, f"camera projection mismatch up to {diff.max():.3f}px"


@pytest.mark.gpu
def test_numpy_model_matches_sensor():
    """The numpy Newton projection matches the REAL SensorTiledCamera: a small
    sphere at a known world point renders centered on the projected pixel."""
    import warp as wp

    import newton
    from scene_physics.visualization.camera import SingleWorldCamera

    pts = np.array(
        [
            [0.0, 0.0, 0.75],
            [0.20, 0.15, 0.78],
            [-0.25, -0.10, 0.70],
            [0.15, -0.20, 0.90],
            [-0.18, 0.22, 0.60],
        ],
        dtype=float,
    )

    expected = npj.project_points(pts, default_camera)
    radius = 0.02
    errs = []
    for pt, exp in zip(pts, expected):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(
                (float(pt[0]), float(pt[1]), float(pt[2])), wp.quat_identity()
            ),
            radius=radius,
        )
        model = builder.finalize()
        cam = SingleWorldCamera(default_camera, model)
        cam.render(model.state())

        depth = cam.depth_image.numpy()[0, 0].reshape(H, W)
        mask = (depth > 0.0) & (depth < default_camera.max_depth)
        assert mask.sum() > 0, "sphere did not render"
        ys, xs = np.nonzero(mask)
        centroid = np.array([xs.mean(), ys.mean()])
        err = np.linalg.norm(centroid - exp)
        errs.append(err)
        print(
            f"\n[b] world={pt}  sensor_centroid={centroid.round(2)}  "
            f"projected={exp.round(2)}  err={err:.3f}px"
        )

    assert max(errs) < 1.5, f"sensor vs model centroid error up to {max(errs):.3f}px"


@pytest.mark.gpu
@blender_only
def test_render_silhouette_iou():
    """End-to-end: an identical primitive scene rendered by the Newton sensor and
    by Blender produces the same silhouette (high IoU + matched centroid)."""
    import warp as wp

    import newton
    from scene_physics.visualization import blender_runner
    from scene_physics.visualization.camera import SingleWorldCamera

    # A few boxes at known world poses (Z-up), sizes in meters.
    boxes = [
        {"pos": [0.0, 0.0, 0.75], "half": [0.06, 0.06, 0.06]},
        {"pos": [0.22, 0.10, 0.72], "half": [0.04, 0.08, 0.05]},
        {"pos": [-0.20, -0.12, 0.70], "half": [0.05, 0.05, 0.09]},
    ]

    # --- Newton sensor mask ---
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for b in boxes:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(tuple(map(float, b["pos"])), wp.quat_identity()),
            hx=b["half"][0],
            hy=b["half"][1],
            hz=b["half"][2],
        )
    model = builder.finalize()
    cam = SingleWorldCamera(default_camera, model)
    cam.render(model.state())
    depth = cam.depth_image.numpy()[0, 0].reshape(H, W)
    mask_newton = (depth > 0.0) & (depth < default_camera.max_depth)

    # --- Blender mask ---
    mask_blender = blender_runner.render_boxes_mask(default_camera, boxes)

    inter = np.logical_and(mask_newton, mask_blender).sum()
    union = np.logical_or(mask_newton, mask_blender).sum()
    iou = inter / max(union, 1)

    cyx_n = np.array(np.nonzero(mask_newton)).mean(axis=1)[::-1]
    cyx_b = np.array(np.nonzero(mask_blender)).mean(axis=1)[::-1]
    cdiff = np.linalg.norm(cyx_n - cyx_b)
    print(
        f"\n[c] IoU={iou:.4f}  centroidΔ={cdiff:.3f}px  "
        f"newton_px={mask_newton.sum()}  blender_px={mask_blender.sum()}"
    )

    assert iou > 0.95, f"silhouette IoU too low: {iou:.3f}"
    assert cdiff < 2.0, f"silhouette centroid mismatch: {cdiff:.3f}px"
