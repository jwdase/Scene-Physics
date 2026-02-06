import copy

import warp as wp
import newton
import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation  
import jax.numpy as jnp

from scene_physics.properties.material import Material

class Body:
    def __init__(self, builder, position, mass, material=None, quat=None, name=None):
        # Loads the builder
        self.builder = builder

        # Loads: Location / Mass / material properties
        self.mass = mass 
        self.position = wp.vec3(*position) # Starting position
        self.quat = quat if quat is not None else wp.quat_identity() # Rotation
        self.cfg = Material() if material is None else material.to_cfg(builder)
        
        # Adds body index to access later
        self.shape_start_index = len(self.builder.body_q) 

        # Adds body to builder
        self.body = self.builder.add_body(
            xform=wp.transform(self.position, self.quat),
            mass=self.mass
        )
        
        # Used for object tracking
        self.name = name
        self.model = None

    def location(self):
        """Prints out body name and location"""
        return f"Position of body {self.name} is {self.get_location()}"

    def get_location(self):
        return tuple(self.position.numpy().to_list())

    def to_pyvista_png(self):
        """ Turns a single body into a pyvista png photo"""
        transform = np.array(self.builder.body_q)[self.shape_start_index]
        return self.pyvista_body(transform)


    def to_pyvista(self, state):
        """ Get's the information (location, rotation) and sends to animator"""
        transform = state['body_q'].numpy()[self.shape_start_index]
        return self.pyvista_body(transform)

    def pyvista_body(self, transform):
        """ Used to generate shape information for pyvista animator """
        mesh = self.pv_mesh.copy() 

        position = transform[0:3]  # [x, y, z]  
        quat_xyzw = transform[3:7]  # [qx, qy, qz, qw] 

        # Switch systems
        rotation = Rotation.from_quat(quat_xyzw)  
        rotation_matrix = rotation.as_matrix()   

        # Apply Transformation
        transform_matrix = np.eye(4)  
        transform_matrix[:3, :3] = rotation_matrix  
        transform_matrix[:3, 3] = position

        assert np.all(np.isfinite(transform_matrix)), "Transform matrix contains non-finite values!"

        # Apply rotation first (around origin)  
        mesh.transform(transform_matrix, inplace=True)  
        
        return mesh

    def move_position_wp(self, state, x, z, quat=None):
        """
        Updating the position of the object once simulation has begun
        - Code sets a complete new location
        - Must have builder complete to do this

        """
        assert isinstance(x, float) and isinstance(z, float), "x and z must be floats"
        assert self.model is not None, "Builder must be finalized"
        
        # New positions
        new_quat = quat if quat is not None else wp.quat_identity()
        new_pos = wp.vec3(x, 0., z)

        # state.body_q is a wp.array — item assignment not supported,
        # so round-trip through numpy to update the transform
        body_q = state.body_q.numpy()
        body_q[self.shape_start_index] = [
            new_pos[0], new_pos[1], new_pos[2],
            new_quat[0], new_quat[1], new_quat[2], new_quat[3],
        ]
        state.body_q = wp.array(body_q, dtype=wp.transformf)

        return None

    def update_position(self, x, z, quat=None):
        """From a proposal updates position in pyvista"""

        # Ensures correctly converted to float
        assert isinstance(x, float) and isinstance(z, float), "x and z must be floats"

        # Get rotation
        new_quat = quat if quat is not None else wp.quat_identity()
        new_pos = wp.vec3(x, 0., z)

        # Update shape transform (shape to body frame)  
        self.builder.body_q[self.shape_start_index] = wp.transform(new_pos, new_quat)

    def translate(self, positions):
        pass

    def add_shape(self):
        raise NotImplementedError

    def __str__(self):
        return self.name

class MeshBody(Body):
    def __init__(self, builder, body, solid, scale=1.0, stable=False, **kwargs):
        """
        Takes in file path to mesh and then builds/adds mesh
        object
        """
        super().__init__(builder, **kwargs)

        # Checks if object is stable / solid
        self.data_test(stable)
        self.solid = solid
        
        # Parses mesh and saves it
        self.pv_mesh = self.load_mesh(body)
        self.nt_mesh = self.convert_mesh()
        
        # Add shape to the builder
        self.add_shape()

    def data_test(self, stable):
        """Checks if mesh is stable"""
        assert not stable or self.cfg.density == 0.0, "Density must be 0"

    def load_mesh(self, file):
        return file if isinstance(file, pv.core.pointset.PolyData) else pv.read(file)

    def convert_mesh(self):
        """ Transfrom mesh into newton type"""
        self.pv_mesh.compute_normals(inplace=True)
        verts = self.pv_mesh.points.astype(np.float32)
        faces = self.pv_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

        return newton.Mesh(verts, faces, compute_inertia=True, is_solid=self.solid, maxhullvert=64)

    def add_shape(self):
        """ Add shape"""
        self.builder.add_shape_mesh(
            body=self.body,
            xform=wp.transform(),  # identity transform in body space
            mesh=self.nt_mesh,
            scale=wp.vec3(1.0, 1.0, 1.0),
            cfg=self.cfg,
        )
