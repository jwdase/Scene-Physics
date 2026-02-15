"""
This document loads all of default setup for different situations of testing
"""

import warp as wp
import newton

from scene_physics.properties.shapes import MeshBody
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material

# Setup Defaults, add ground plane
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
builder.add_ground_plane()

# Setup Materials
Wedge_material = Dynamic_Material
Ramp_material = Still_Material

# Objects
paths = [f"objects/stable_scene/{val}" for val in ["table.obj", "rectangle.obj"]]

# Placing objects
table = MeshBody(
    builder=builder,
    body=paths[0],
    solid=True,
    position=wp.vec3(0.0, 0.0, 0.0),
    mass=0.0,
    material=Wedge_material,
)

rectangle = MeshBody(
    builder=builder,
    body=paths[1],
    solid=True,
    position=wp.vec3(0.0, 0.0, 0.0),
    mass=0.0,
    material=Ramp_material,
)

