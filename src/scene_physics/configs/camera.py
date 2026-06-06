"""
Shared camera-perspective defaults.

Single source of truth for the viewpoint + intrinsics used to render scenes: the
dataset generator's occlusion/visibility checks (data_gen/scene_gen.py) and the
sampling run's GT render + likelihood camera (simulation/sim_sampling.py) both
pull `default_camera` from here, so a generated scene's occlusion gate and the
downstream sampling render see the scene from the exact same viewpoint.

Z-up convention: `eye` looks from -Y toward the origin; the table top is the XY
plane. The Camera machinery (SensorTiledCamera wrappers) lives in sim_sampling.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Viewpoint (Z-up): eye looks from -Y at the origin; table top is the XY plane.
EYE = np.array([0.0, -1.5, 1.5])
TARGET = np.zeros(3)
UP = np.array([0, 0, 1])

# Default intrinsics
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FOV_DEGREE = 60
CAM_MAX_DEPTH = 4.0


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fov_degree: float
    max_depth: float = CAM_MAX_DEPTH

    eye: np.ndarray = field(default_factory=lambda: EYE)
    target: np.ndarray = field(default_factory=lambda: TARGET)
    up: np.ndarray = field(default_factory=lambda: UP)

    @property
    def fov_rad(self):
        return np.radians(self.fov_degree)


# The canonical perspective shared by dataset generation and the sampling run.
default_camera = CameraIntrinsics(
    width=CAM_WIDTH, height=CAM_HEIGHT, fov_degree=CAM_FOV_DEGREE
)


def camera_to_dict(
    intr: "CameraIntrinsics", name: str | None = None, description: str | None = None
) -> dict:
    """Serialize a CameraIntrinsics to a plain JSON-able camera-layout dict.

    The schema is a superset of the per-option configs written by
    render_pipeline.render_fov_grid, so those files load back via camera_from_dict.
    """
    d = {
        "width": int(intr.width),
        "height": int(intr.height),
        "fov_degree": float(intr.fov_degree),
        "max_depth": float(intr.max_depth),
        "eye": [float(x) for x in np.asarray(intr.eye).ravel()],
        "target": [float(x) for x in np.asarray(intr.target).ravel()],
        "up": [float(x) for x in np.asarray(intr.up).ravel()],
    }
    if description is not None:
        d = {"description": description, **d}
    if name is not None:
        d = {"name": name, **d}
    return d


def camera_from_dict(d: dict) -> "CameraIntrinsics":
    """Build a CameraIntrinsics from a camera-layout dict (extra keys like
    `id`/`position` from grid configs are ignored)."""
    return CameraIntrinsics(
        width=int(d["width"]),
        height=int(d["height"]),
        fov_degree=float(d["fov_degree"]),
        max_depth=float(d.get("max_depth", CAM_MAX_DEPTH)),
        eye=np.asarray(d["eye"], dtype=float),
        target=np.asarray(d["target"], dtype=float),
        up=np.asarray(d["up"], dtype=float),
    )


def save_camera_config(
    intr: "CameraIntrinsics",
    path,
    name: str | None = None,
    description: str | None = None,
) -> Path:
    """Write a camera layout to a JSON config file. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(camera_to_dict(intr, name, description), indent=2))
    return path


def load_camera_config(path) -> "CameraIntrinsics":
    """Load a camera-layout JSON (e.g. a render_fov_grid option, or default_camera)
    into a CameraIntrinsics that the render pipeline can use directly."""
    return camera_from_dict(json.loads(Path(path).read_text()))
