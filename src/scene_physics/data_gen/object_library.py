"""
Object library loader for the scene-generation pipeline.

Reads raw `.obj` assets from `resources/objects/objects/`, converts them from
the Blender export convention (Y-up) into the pipeline's Z-up convention, and
wraps them as `newton.Mesh` instances at their native `.obj` scale.

Y-up -> Z-up is the +90 deg rotation about X that `add_usd` effectively
applies on import: (x, y, z) -> (x, -z, y). It is a pure rotation, so object
dimensions are preserved.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import newton

# resources/objects/objects lives at the repository root, four parents up from
# this file (.../src/scene_physics/simulation/object_library.py).
DEFAULT_LIBRARY_DIR = (
    Path(__file__).resolve().parents[3] / "resources" / "objects" / "objects"
)


def _yup_to_zup(vertices: np.ndarray) -> np.ndarray:
    """Rotate Y-up vertices into Z-up: (x, y, z) -> (x, -z, y)."""
    out = np.empty_like(vertices)
    out[:, 0] = vertices[:, 0]
    out[:, 1] = -vertices[:, 2]
    out[:, 2] = vertices[:, 1]
    return out


@dataclass(frozen=True)
class LoadedObject:
    """A library object in the Z-up frame at native `.obj` scale."""

    name: str
    vertices: np.ndarray  # (N, 3) float32, Z-up
    indices: np.ndarray   # (3 * T,) int32 triangle list
    mesh: newton.Mesh

    @property
    def aabb_min(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def aabb_max(self) -> np.ndarray:
        return self.vertices.max(axis=0)

    @property
    def extents(self) -> np.ndarray:
        return self.aabb_max - self.aabb_min

    @property
    def center(self) -> np.ndarray:
        """Local AABB center, used as the object's geometric center."""
        return 0.5 * (self.aabb_min + self.aabb_max)

    @property
    def min_z(self) -> float:
        return float(self.vertices[:, 2].min())

    @property
    def height(self) -> float:
        return float(self.extents[2])


def _obj_path(name: str, library_dir: Path) -> Path:
    path = Path(library_dir) / f"{name}.obj"
    if not path.exists():
        raise FileNotFoundError(f"No .obj named {name!r} in {library_dir}")
    return path


@functools.lru_cache(maxsize=None)
def load_object(name: str, library_dir: Path = DEFAULT_LIBRARY_DIR) -> LoadedObject:
    """Load and cache a library object as a Z-up `newton.Mesh`."""
    mesh_pv = pv.read(_obj_path(name, library_dir)).triangulate()

    faces = np.asarray(mesh_pv.faces).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError(f"{name}: triangulation did not yield pure triangles")
    indices = np.ascontiguousarray(faces[:, 1:].reshape(-1), dtype=np.int32)

    vertices = _yup_to_zup(np.asarray(mesh_pv.points, dtype=np.float32))
    mesh = newton.Mesh(vertices, indices)

    return LoadedObject(name=name, vertices=vertices, indices=indices, mesh=mesh)


def available_objects(library_dir: Path = DEFAULT_LIBRARY_DIR) -> list[str]:
    """Sorted list of every object name available in the library."""
    return sorted(p.stem for p in Path(library_dir).glob("*.obj"))


def sample_objects(
    n: int,
    rng: np.random.Generator,
    pool: list[str] | None = None,
    exclude: tuple[str, ...] = (),
    library_dir: Path = DEFAULT_LIBRARY_DIR,
) -> list[str]:
    """Sample `n` distinct object names from `pool` (or the whole library)."""
    candidates = [c for c in (pool or available_objects(library_dir)) if c not in exclude]
    if n > len(candidates):
        raise ValueError(f"Requested {n} objects but only {len(candidates)} available")
    return list(rng.choice(candidates, size=n, replace=False))
