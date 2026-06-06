"""Render generated scenes to photoreal PNGs + segmentation maps (uv-env side).

Orchestrates the Blender render (blender/render_scene.py) and the uv-side decode
(segmentation.py) for one scene or a whole dataset. Blender does the geometry +
lighting; numpy/PIL here turn the flat ID render into the integer label map.

CLI (from src/scene_physics/):
    BLENDER=/path/to/blender uv run python -m scene_physics.visualization.render_pipeline \
        --scenes scene001 scene002          # or --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scene_physics.configs.camera import (CameraIntrinsics, default_camera,
                                          load_camera_config)
from scene_physics.visualization import blender_runner, segmentation
from scene_physics.visualization.blender_runner import intrinsics_to_dict

ROOT = Path(__file__).resolve().parents[3]
SCENES_ROOT = ROOT / "resources" / "generated_scenes"
DEFAULT_HDRI = ROOT / "resources" / "hdri" / "lythwood_room_4k.hdr"


def scene_names(scene_dir: Path) -> list[str]:
    """Object names to render: keys of <scene>_truth.json (table + dynamics)."""
    truth = next(scene_dir.glob("data/*_truth.json"))
    return list(json.loads(truth.read_text()).keys())


def render_scene(
    scene_dir: str | Path,
    intr=default_camera,
    hdri: str | Path = DEFAULT_HDRI,
    samples: int = 128,
    device: str = "GPU",
    world_strength: float = 1.0,
    hdri_rotation: float = 0.0,
    view_transform: str = "AgX",
    out_dir: str | Path | None = None,
) -> Path:
    """Render one scene; writes results/{render,seg_raw,segmentation,...}. Returns results dir."""
    scene_dir = Path(scene_dir)
    usd = next(scene_dir.glob("data/*_physics.usdc"))
    out_dir = Path(out_dir) if out_dir is not None else scene_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "usd": str(usd),
        "names": scene_names(scene_dir),
        "intrinsics": intrinsics_to_dict(intr),
        "hdri": str(hdri),
        "out_dir": str(out_dir),
        "samples": samples,
        "device": device,
        "world_strength": world_strength,
        "hdri_rotation": hdri_rotation,
        "view_transform": view_transform,
    }
    blender_runner.run_render_scene(job)
    segmentation.write_outputs(str(out_dir))
    return out_dir


# (azimuth°, elevation°, distance m, fov°, label) orbiting the tabletop centre.
# azimuth 0 = camera on -Y (the default front); + rotates toward +X.
_CAMERA_VIEWS = [
    (0, 12, 1.70, 50, "eye-level front"),
    (0, 28, 1.55, 50, "raised front"),
    (-35, 25, 1.70, 50, "3/4 front-left"),
    (35, 25, 1.70, 50, "3/4 front-right"),
    (-30, 14, 1.60, 52, "low 3/4 left"),
    (25, 45, 1.55, 50, "high angle"),
    (10, 68, 1.75, 48, "near top-down"),
    (0, 18, 1.25, 58, "close low front"),
    (-22, 33, 2.05, 42, "wide far-left"),
    (-72, 20, 1.70, 50, "side profile"),
]
_VIEW_TARGET = np.array([0.0, 0.0, 0.80])  # tabletop centre (objects sit ~z=0.78)


def _orbit_eye(target, az_deg, el_deg, dist):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return target + dist * np.array(
        [math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)]
    )


def _make_montage(out_dir: Path, labels, cols: int = 5, tile=(480, 360)) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    tw, th = tile
    n = len(labels)
    rows = (n + cols - 1) // cols
    montage = Image.new("RGB", (cols * tw, rows * th), (25, 25, 25))
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for i, label in enumerate(labels):
        p = out_dir / f"view_{i + 1:02d}.png"
        if not p.exists():
            continue
        x, y = (i % cols) * tw, (i // cols) * th
        montage.paste(Image.open(p).convert("RGB").resize((tw, th)), (x, y))
        draw.rectangle([x, y, x + tw, y + 16], fill=(0, 0, 0))
        draw.text((x + 4, y + 3), label, fill=(255, 235, 90), font=font)
    path = out_dir / "montage.png"
    montage.save(path)
    return path


def render_camera_views(
    scene_dir: str | Path,
    out_dir: str | Path | None = None,
    samples: int = 160,
    device: str = "GPU",
    width: int = 960,
    height: int = 720,
) -> Path:
    """Render one scene from `_CAMERA_VIEWS` to explore natural tabletop angles.

    Writes view_NN.png + a labelled montage.png into <scene>/camera_views/.
    """
    scene_dir = Path(scene_dir)
    usd = next(scene_dir.glob("data/*_physics.usdc"))
    out_dir = Path(out_dir) if out_dir is not None else scene_dir / "camera_views"
    out_dir.mkdir(parents=True, exist_ok=True)

    views, labels = [], []
    for i, (az, el, dist, fov, desc) in enumerate(_CAMERA_VIEWS, start=1):
        eye = _orbit_eye(_VIEW_TARGET, az, el, dist)
        intr = CameraIntrinsics(
            width=width,
            height=height,
            fov_degree=fov,
            eye=eye,
            target=_VIEW_TARGET.copy(),
        )
        views.append(intrinsics_to_dict(intr))
        labels.append(f"{i:02d} {desc} (az{az} el{el} d{dist})")

    job = {
        "usd": str(usd),
        "names": scene_names(scene_dir),
        "hdri": str(DEFAULT_HDRI),
        "out_dir": str(out_dir),
        "samples": samples,
        "device": device,
        "world_strength": 1.0,
        "hdri_rotation": 0.0,
        "view_transform": "AgX",
        "views": views,
    }
    blender_runner.run_render_views(job)
    montage = _make_montage(out_dir, labels)
    print(f"[views] {scene_dir.name} -> {out_dir} (montage: {montage})")
    return out_dir


# FOV sweep (degrees, vertical) applied to every camera position: telephoto -> wide.
_FOV_VALUES = [20, 27, 34, 41, 48, 55, 62, 70, 78, 85]


def _make_grid_montage(
    out_dir: Path, names, fovs, descs, rows, cols, tile=(192, 144)
) -> Path:
    """Master grid: one row per camera position, one column per FOV value."""
    from PIL import Image, ImageDraw, ImageFont

    tw, th = tile
    montage = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for idx, name in enumerate(names):
        r, c = idx // cols, idx % cols
        p = out_dir / f"{name}.png"
        if not p.exists():
            continue
        montage.paste(Image.open(p).convert("RGB").resize((tw, th)), (c * tw, r * th))
        draw.rectangle([c * tw, r * th, c * tw + tw, r * th + 13], fill=(0, 0, 0))
        draw.text(
            (c * tw + 3, r * th + 2), f"fov {fovs[c]}", fill=(255, 235, 90), font=font
        )
        if c == 0:
            draw.text(
                (3, r * th + th - 11), descs[r][:20], fill=(120, 220, 255), font=font
            )
    path = out_dir / "grid_montage.png"
    montage.save(path)
    return path


def render_fov_grid(
    scene_dir: str | Path,
    out_dir: str | Path | None = None,
    samples: int = 96,
    device: str = "GPU",
    width: int = 640,
    height: int = 480,
) -> Path:
    """Render the 10 camera positions x 10 FOV values = 100 options, saving a
    reproducible config JSON per option + a 10x10 montage (rows=position, cols=fov)."""
    scene_dir = Path(scene_dir)
    usd = next(scene_dir.glob("data/*_physics.usdc"))
    out_dir = (
        Path(out_dir)
        if out_dir is not None
        else scene_dir / "camera_views" / "fov_grid"
    )
    cfg_dir = out_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    views, names = [], []
    descs = [v[4] for v in _CAMERA_VIEWS]
    for pi, (az, el, dist, _fov, desc) in enumerate(_CAMERA_VIEWS, start=1):
        eye = _orbit_eye(_VIEW_TARGET, az, el, dist)
        for fi, fov in enumerate(_FOV_VALUES, start=1):
            oid = f"p{pi:02d}_f{fi:02d}"
            intr = CameraIntrinsics(
                width=width,
                height=height,
                fov_degree=fov,
                eye=eye,
                target=_VIEW_TARGET.copy(),
            )
            d = intrinsics_to_dict(intr)
            views.append(d)
            names.append(oid)
            (cfg_dir / f"{oid}.json").write_text(
                json.dumps(
                    {
                        "id": oid,
                        "position": {
                            "index": pi,
                            "desc": desc,
                            "azimuth_deg": az,
                            "elevation_deg": el,
                            "distance_m": dist,
                        },
                        "fov_degree": fov,
                        "width": width,
                        "height": height,
                        "eye": d["eye"],
                        "target": d["target"],
                        "up": d["up"],
                    },
                    indent=2,
                )
            )

    job = {
        "usd": str(usd),
        "names": scene_names(scene_dir),
        "hdri": str(DEFAULT_HDRI),
        "out_dir": str(out_dir),
        "samples": samples,
        "device": device,
        "world_strength": 1.0,
        "hdri_rotation": 0.0,
        "view_transform": "AgX",
        "views": views,
        "view_names": names,
    }
    blender_runner.run_render_views(job)
    montage = _make_grid_montage(
        out_dir,
        names,
        _FOV_VALUES,
        descs,
        rows=len(_CAMERA_VIEWS),
        cols=len(_FOV_VALUES),
    )
    print(
        f"[fov-grid] {scene_dir.name} -> {out_dir} ({len(names)} options; montage: {montage})"
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Render generated scenes via Blender.")
    ap.add_argument(
        "--scenes", nargs="*", default=[], help="scene names, e.g. scene001"
    )
    ap.add_argument(
        "--all", action="store_true", help="render every scene in the dataset"
    )
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--hdri", default=str(DEFAULT_HDRI))
    ap.add_argument("--view-transform", default="AgX")
    ap.add_argument(
        "--camera-config",
        help="path to a camera-layout JSON (e.g. a fov_grid option or default_camera) "
        "to use as the viewpoint; defaults to default_camera",
    )
    args = ap.parse_args()

    intr = (
        load_camera_config(args.camera_config) if args.camera_config else default_camera
    )

    if args.all:
        dirs = sorted(p for p in SCENES_ROOT.glob("scene*") if p.is_dir())
    else:
        dirs = [SCENES_ROOT / s for s in args.scenes]
    if not dirs:
        ap.error("specify --scenes <name...> or --all")

    for d in dirs:
        print(f"[render] {d.name} ...")
        out = render_scene(
            d,
            intr=intr,
            samples=args.samples,
            device=args.device,
            hdri=args.hdri,
            view_transform=args.view_transform,
        )
        print(f"[render] {d.name} -> {out}")


if __name__ == "__main__":
    main()
