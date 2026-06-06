"""Tests for the Blender render + segmentation pipeline.

  * material coverage      -- every object in the dataset has a tuned material   [pure]
  * USD pose fidelity      -- Blender places objects at the settled truth poses   [blender]
  * render + segmentation  -- a scene renders and the ID map is correct/aligned   [blender,gpu]

Blender must be USD-capable; point $BLENDER at an official build (the distro one
is often compiled without USD). GPU is optional -- the render test falls back to
CPU, just slower.

  BLENDER=/path/to/blender uv run pytest tests/test_render_pipeline.py -v -s
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scene_physics.configs.camera import default_camera
from scene_physics.visualization.blender.material_specs import MATERIAL_SPECS

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "resources" / "generated_scenes"
SCENE = SCENES / "scene001"

HAVE_BLENDER = bool(os.environ.get("BLENDER") or shutil.which("blender"))
blender_only = pytest.mark.skipif(not HAVE_BLENDER, reason="Blender not installed")

W, H = default_camera.width, default_camera.height


def _distinct_dataset_objects() -> set[str]:
    names: set[str] = set()
    for truth in SCENES.glob("scene*/data/*_truth.json"):
        names.update(json.loads(truth.read_text()).keys())
    return names


def test_materials_cover_dataset():
    """Every object that appears in any scene has a hand-tuned material spec
    (otherwise it would silently fall back to flat gray in the stimulus)."""
    objects = _distinct_dataset_objects()
    assert objects, "no truth.json files found"
    missing = objects - set(MATERIAL_SPECS)
    assert not missing, f"objects with no material spec: {sorted(missing)}"


def _quat_angle_deg(a, b) -> float:
    a = np.asarray(a, float)
    a /= np.linalg.norm(a)
    b = np.asarray(b, float)
    b /= np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1.0, abs(float(a @ b)))))


@blender_only
def test_usd_import_poses():
    """Blender's USD import reproduces the settled truth.json poses."""
    from scene_physics.visualization import blender_runner

    truth = json.loads(next(SCENE.glob("data/*_truth.json")).read_text())
    usd = str(next(SCENE.glob("data/*_physics.usdc")))
    got = blender_runner.dump_usd_poses(usd, list(truth))

    assert set(got) == set(truth), "imported objects differ from truth"
    for name, pose in truth.items():
        dpos = math.dist(pose[:3], got[name]["pos"])
        dang = _quat_angle_deg(pose[3:], got[name]["quat_xyzw"])
        print(f"\n  {name:42s} Δpos={dpos*1000:.4f}mm  Δrot={dang:.3f}°")
        assert dpos < 1e-3, f"{name} position off by {dpos*1000:.3f} mm"
        assert dang < 3.0, f"{name} rotation off by {dang:.2f}°"


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Render scene001 once (low samples) into a temp dir; decode segmentation."""
    from scene_physics.visualization import render_pipeline, segmentation

    # Default to CPU so the test is robust regardless of GPU state: the Newton
    # gpu tests hold warp/JAX memory, which collides with Blender Cycles on small
    # cards (8 GB here). Set RENDER_DEVICE=GPU on a roomy card to speed it up.
    out = tmp_path_factory.mktemp("render")
    render_pipeline.render_scene(
        SCENE,
        samples=8,
        device=os.environ.get("RENDER_DEVICE", "CPU"),
        out_dir=out,
    )
    segmentation.write_outputs(str(out))
    return out


@blender_only
def test_render_outputs_exist(rendered):
    for fn in (
        "render.png",
        "seg_raw.png",
        "segmentation.png",
        "segmentation_overlay.png",
        "segmentation_labels.json",
    ):
        assert (rendered / fn).exists(), f"missing output {fn}"

    render = np.asarray(Image.open(rendered / "render.png"))
    assert render.shape[:2] == (H, W)
    # Not a blank frame: the studio backdrop + table should give real variance.
    assert render[..., :3].std() > 5.0, "render looks blank"


@blender_only
def test_segmentation_labels_valid(rendered):
    """Every labeled pixel maps to a real object; background dominates; the table
    (always visible) is present; the map is camera-aligned."""
    seg = np.asarray(Image.open(rendered / "segmentation.png"))
    labels = json.loads((rendered / "segmentation_labels.json").read_text())
    id_to_name = {v: k for k, v in labels.items()}

    assert seg.shape == (H, W)
    present = set(np.unique(seg).tolist())
    assert 0 in present, "no background pixels"

    # Every present id is a known label.
    for i in present:
        assert int(i) in id_to_name, f"unknown segmentation id {i}"

    counts = {int(i): int((seg == i).sum()) for i in present}
    # Background is the largest region (the table sits in a studio void).
    assert counts[0] == max(counts.values()), "background is not the largest region"
    # The table is always visible and sizeable.
    table_id = labels["dining_room_table"]
    assert counts.get(table_id, 0) > 0.05 * seg.size, "table under-segmented"

    # A meaningful number of the scene's objects are visible & labeled.
    truth = json.loads(next(SCENE.glob("data/*_truth.json")).read_text())
    visible = [n for n in truth if counts.get(labels.get(n, -1), 0) > 0]
    print(
        f"\n  visible/labeled objects: {len(visible)}/{len(truth)} -> {sorted(visible)}"
    )
    assert len(visible) >= max(3, len(truth) // 2)


@blender_only
def test_segmentation_registered_to_render(rendered):
    """Foreground in the segmentation must coincide with non-backdrop pixels in the
    beauty render (camera + poses identical across the two passes)."""
    seg = np.asarray(Image.open(rendered / "segmentation.png"))
    render = np.asarray(Image.open(rendered / "render.png").convert("RGB")).astype(
        np.float32
    )

    fg = seg > 0
    # The studio backdrop is a fairly uniform light gray; object/table pixels deviate
    # from the per-image background mode. Use seg-background pixels to estimate it.
    bg_color = render[seg == 0].mean(axis=0)
    deviation = np.linalg.norm(render - bg_color, axis=2)
    # Where segmentation says "object", the render should differ from the backdrop.
    frac = (deviation[fg] > 12.0).mean()
    print(f"\n  fg pixels deviating from backdrop: {frac*100:.1f}%")
    assert frac > 0.7, "segmentation foreground does not line up with the render"
