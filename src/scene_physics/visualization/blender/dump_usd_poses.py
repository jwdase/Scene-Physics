"""Blender entry-point: import a scene USD and dump each object's world pose.

Used by the USD-import fidelity test to confirm Blender places objects at the
settled truth.json poses.

    blender --background --python dump_usd_poses.py -- <job.json> <out.json>

job.json: {"usd": "<path>", "names": ["obj1", ...]}
out.json: {name: {"pos":[x,y,z], "quat_xyzw":[qx,qy,qz,qw]}, ...}
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _scene import import_scene_usd, reset_scene  # noqa: E402


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    job_path, out_path = argv[0], argv[1]
    with open(job_path) as f:
        job = json.load(f)

    reset_scene()
    mapping = import_scene_usd(job["usd"], job["names"])

    out = {}
    for name, obj in mapping.items():
        m = obj.matrix_world
        loc = m.translation
        q = m.to_quaternion()  # mathutils Quaternion is (w, x, y, z)
        out[name] = {
            "pos": [loc.x, loc.y, loc.z],
            "quat_xyzw": [q.x, q.y, q.z, q.w],
        }

    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
