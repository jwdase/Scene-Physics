"""
This document loads all of the default setup for simulation Scene01
"""

import os
import warp as wp
import newton

from scene_physics.properties.shapes import MeshBody
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material

# Root places us in correct file folder
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Setup Defaults: Model, Ground-Plan
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
builder.add_ground_plane()

def scene01():
    """
    Setup scene01
        Dynamic: Bowl, Coffee
        Stable: Table
        Sample_Over: Bowl
    """

    ROOT = os.path.join(PACKAGE_ROOT, "objects", "scene01")

    Bowl = MeshBody(builder=builder, body=f"{ROOT}/BOWL.obj", material=Dynamic_Material)
    Coffee = MeshBody(builder=builder, body=f"{ROOT}/COFFEE.obj", material=Dynamic_Material)
    Table = MeshBody(builder=builder, body=f"{ROOT}/TABLE.obj", material=Still_Material)

    return builder, Bowl, Coffee, Table
