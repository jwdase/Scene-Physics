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
from pathlib import Path

from scene_physics.configs.camera import default_camera
from scene_physics.visualization import blender_runner, segmentation
from scene_physics.visualization.blender_runner import intrinsics_to_dict

ROOT = Path(__file__).resolve().parents[3]
SCENES_ROOT = ROOT / "resources" / "generated_scenes"
DEFAULT_HDRI = ROOT / "resources" / "hdri" / "studio_small_08_2k.hdr"


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
    world_strength: float = 0.5,
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
        "view_transform": view_transform,
    }
    blender_runner.run_render_scene(job)
    segmentation.write_outputs(str(out_dir))
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
    args = ap.parse_args()

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
            samples=args.samples,
            device=args.device,
            hdri=args.hdri,
            view_transform=args.view_transform,
        )
        print(f"[render] {d.name} -> {out}")


if __name__ == "__main__":
    main()
