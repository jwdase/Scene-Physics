"""uv-env-side helpers for driving Blender as a subprocess.

Blender ships its own Python (no numpy, can't import this package), so we talk to
it across the process boundary: hand it a script + a JSON job, read JSON/image
files back. This module centralizes binary discovery and invocation so tests and
the render pipeline share one path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

BLENDER_DIR = Path(__file__).resolve().parent / "blender"


def blender_bin() -> str:
    """Path to the Blender executable (override with $BLENDER)."""
    cand = os.environ.get("BLENDER") or shutil.which("blender")
    if not cand or not Path(cand).exists():
        raise FileNotFoundError(
            "Blender executable not found. Install Blender or set $BLENDER."
        )
    return cand


def run_script(
    script_name: str, args: list[str], timeout: int = 600
) -> subprocess.CompletedProcess:
    """Run blender/<script_name> headless with trailing `-- args`."""
    script = BLENDER_DIR / script_name
    cmd = [blender_bin(), "--background", "--python", str(script), "--", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Blender script {script_name} failed (rc={proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\n\nSTDERR:\n{proc.stderr[-4000:]}"
        )
    return proc


def intrinsics_to_dict(intr) -> dict:
    """Serialize a CameraIntrinsics into the plain dict Blender scripts expect."""
    return {
        "eye": list(map(float, np.asarray(intr.eye).tolist())),
        "target": list(map(float, np.asarray(intr.target).tolist())),
        "up": list(map(float, np.asarray(intr.up).tolist())),
        "fov_degree": float(intr.fov_degree),
        "width": int(intr.width),
        "height": int(intr.height),
        "max_depth": float(intr.max_depth),
    }


def project_points(intr, points: np.ndarray) -> np.ndarray:
    """Project world points via Blender's camera; returns (N,2) pixels (NaN = off)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    with tempfile.TemporaryDirectory() as d:
        job = Path(d) / "job.json"
        out = Path(d) / "out.json"
        job.write_text(
            json.dumps(
                {
                    "intrinsics": intrinsics_to_dict(intr),
                    "points": points.tolist(),
                }
            )
        )
        run_script("project_points.py", [str(job), str(out)])
        pixels = json.loads(out.read_text())["pixels"]
    return np.array([[np.nan, np.nan] if p is None else p for p in pixels], dtype=float)


def dump_usd_poses(usd_path: str, names: list[str]) -> dict:
    """Import the USD in Blender and return {name: {"pos":[xyz], "quat_xyzw":[...]}}."""
    with tempfile.TemporaryDirectory() as d:
        job = Path(d) / "job.json"
        out = Path(d) / "out.json"
        job.write_text(json.dumps({"usd": usd_path, "names": names}))
        run_script("dump_usd_poses.py", [str(job), str(out)])
        return json.loads(out.read_text())


def run_render_scene(job: dict) -> None:
    """Drive blender/render_scene.py with a job dict (writes into job['out_dir'])."""
    with tempfile.TemporaryDirectory() as d:
        job_path = Path(d) / "job.json"
        job_path.write_text(json.dumps(job))
        run_script("render_scene.py", [str(job_path)], timeout=1800)


def run_render_views(job: dict) -> None:
    """Drive blender/render_views.py (multi-camera beauty render) with a job dict."""
    with tempfile.TemporaryDirectory() as d:
        job_path = Path(d) / "job.json"
        job_path.write_text(json.dumps(job))
        run_script("render_views.py", [str(job_path)], timeout=1800)


def render_boxes_mask(intr, boxes: list[dict]) -> np.ndarray:
    """Render boxes in Blender; return the (H,W) bool silhouette (alpha>0)."""
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory() as d:
        job = Path(d) / "job.json"
        out = Path(d) / "out.png"
        job.write_text(
            json.dumps(
                {
                    "intrinsics": intrinsics_to_dict(intr),
                    "boxes": boxes,
                }
            )
        )
        run_script("render_boxes.py", [str(job), str(out)])
        img = imageio.imread(out)
    return img[..., 3] > 127  # alpha channel
