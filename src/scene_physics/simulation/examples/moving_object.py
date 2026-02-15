# Package Imports
import pyvista as pv
import warp as wp
import numpy as np

# Newton Library
import newton
from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD

# Files
from scene_physics.properties.shapes import MeshBody
from scene_physics.properties.basic_materials import Dynamic_Material, Still_Material
from scene_physics.visualization.scene import PyVistaVisualizer

# Setup Defaults
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Add Plane
builder.add_ground_plane()

# Material
Ball_material = Dynamic_Material
Ramp_material = Still_Material

# Objects
paths = [f'objects/stable_scene/{val}' for val in ['table.obj', 'rectangle.obj']]

# Placing objects
table = MeshBody(
    builder=builder,
    body=paths[0],
    solid=True,
    scale=1.0,
    position=wp.vec3(0., 0., 0.),
    mass=0.0,
    material=Ramp_material,
    name='table'
)

rectangle = MeshBody(
    builder=builder,
    body=paths[1],
    solid=True,
    scale=1.0,
    position=wp.vec3(0., 0., 0.),
    mass=0.0,
    material=Ball_material,
    name='rectangle'
)

# List out bodies, camera for visualizer
bodies = [table, rectangle]
camera = [
    (1, 1.5, 3),
    (0, 1, 0),
    (0, 1, 0),
]

# Generate visualizer
visualizer = PyVistaVisualizer(bodies, camera_position=camera)

# Move objects and view changes
for i, movement in enumerate(np.linspace(-.5, .5, 10)):
    rectangle.update_position(movement, 0.)
    visualizer.gen_png(f'recordings/movement/image0{i}.png')

print('complete')



