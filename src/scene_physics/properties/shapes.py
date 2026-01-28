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
        self.builder = builder

        # Location / Mass
        self.mass = mass 
        self.position = wp.vec3(*position) # Starting position
        self.quat = quat if quat is not None else wp.quat_identity() # Rotation

        # Give body matieral properties
        if material is None:
            self.cfg = Material()
        else:
            self.cfg = material.to_cfg(builder)
        
        # Add body to builder list
        self.shape_start_index = len(self.builder.body_q) 

        # Add body number
        self.body = self.builder.add_body(
            xform=wp.transform(self.position, self.quat),
            mass=self.mass
        )

        self.name = name

    def to_pyvista_png(self):
        """ Used for extracting start position"""

        transform = np.array(self.builder.body_q)[self.shape_start_index]

        return self.pyvista_body(transform)


    def to_pyvista(self, state):
        """ Class used for animating position of objects through simulation"""

        # Extract full transform (position + rotation)
        transform = state['body_q'].numpy()[self.shape_start_index]

        return self.pyvista_body(transform)

    def pyvista_body(self, transform):
        """ Used to generate the shape """
        
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

    def update_position(self, x, z, quat=None):
        """
        Update position of mesh to in space
        """

        # Ensures correctly converted to float
        assert isinstance(x, float) and isinstance(z, float), "x and z must be floats"

        # Get rotation
        new_quat = quat if quat is not None else wp.quat_identity()
        new_pos = wp.vec3(x, 0., z)

        # New transformation
        new_transform = wp.transform(new_pos, new_quat)

        # Update shape transform (shape to body frame)  
        self.builder.shape_transform[self.shape_start_index] = new_transform 
        self.builder.body_q[self.shape_start_index] = new_transform

    def translate(self, positions):
        pass

    def add_shape(self):
        raise NotImplementedError

    def __str__(self):
        return self.name

class Sphere(Body):
    def __init__(self, builder, radius, **kwargs):
        super().__init__(builder, **kwargs)
        self.radius = radius

        self.pv_mesh = pv.Sphere(radius=self.radius, center=np.array([0, 0, 0]))

        # Call initilization
        self.add_shape()

    def add_shape(self):
        self.builder.add_shape_sphere(
            body=self.body, radius=self.radius, cfg=self.cfg,
        )

class Box(Body):
    def __init__(self, builder, half_extends, **kwargs):
        super().__init__(builder, **kwargs)

        self.half_extends = half_extends

        self.pv_mesh = pv.Box(bounds=(
            self.position[0] - self.half_extends[0], self.position[0] + self.half_extends[0],
            self.position[1] - self.half_extends[1], self.position[1] + self.half_extends[1],
            self.position[2] - self.half_extends[2], self.position[2] + self.half_extends[2],
        ))

        self.add_shape()

    def add_shape(self):
        self.builder.add_shape_box(
            body=self.body, hx=self.half_extends[0], hy=self.half_extends[1], hz=self.half_extends[2], cfg=self.cfg
        )
        

class MeshBody(Body):
    def __init__(self, builder, body, solid, scale=1.0, stable=False, **kwargs):
        """
        Takes in file path to mesh and then builds/adds mesh
        object
        """
        super().__init__(builder, **kwargs)

        self.data_test(stable)

        self.solid = solid
        self.scale = scale

        self.pv_mesh = self.load_mesh(body)
        self.nt_mesh = self.convert_mesh()

        self.add_shape()

        self.pv_mesh = self.pv_mesh.scale(scale)

    def data_test(self, stable):
        if stable:
            assert self.cfg.density == 0.0 


    def load_mesh(self, file):
        if isinstance(file, pv.core.pointset.PolyData):
            return file
        else:
            return pv.read(file)

    def convert_mesh(self):
        # Transform into object for Newton
        self.pv_mesh.compute_normals(inplace=True)
        verts = self.pv_mesh.points.astype(np.float32)
        faces = self.pv_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

        return newton.Mesh(verts, faces, compute_inertia=True, is_solid=self.solid, maxhullvert=64)

    def add_shape(self):
        self.builder.add_shape_mesh(
            body=self.body,
            xform=wp.transform(),  # identity transform in body space
            mesh=self.nt_mesh,
            scale=wp.vec3(np.ones(3) * self.scale),
            cfg=self.cfg,
        )

class StableMesh(MeshBody):
    def __init__(self, builder, **kwargs):
        kwargs['mass'] = 0.0
        kwargs['stable'] = True
        super().__init__(builder, **kwargs)
        self.rotation = Rotation.from_quat(self.quat)

    # def to_pyvista(self, state):
    #     mesh = self.pv_mesh.copy()
    #     return mesh.rotate(self.rotation, inplace=False)

class SoftMesh:
    def __init__(self, builder, path, mass, position, material):
        self.pv_mesh = self.load_mesh(path)
        self.builder = builder

        # General Values
        self.mass = mass
        self.position = position
        self.material = material

        # Particle Info
        self.particle_start_index = None  
        self.num_particles = None

        self.add_shape()

    def load_mesh(self, file):
        if isinstance(file, pv.UnstructuredGrid):
            return file
        else:
            return pv.read(file)

    def load_vertices(self):  
        vertices = self.pv_mesh.points.astype(np.float32).tolist()  
        
        cells = self.pv_mesh.cells.reshape(-1, 5)[:, 1:] 
        tet_indices = cells.flatten().astype(np.int32).tolist()  
        
        return vertices, tet_indices

    def add_shape(self):
        self.particle_start_index = len(self.builder.particle_q)

        vertices, tet_indices = self.load_vertices()

        self.num_particles = len(vertices) 

        print(f"Density: {self.material.density}")

        self.builder.add_soft_mesh(
            pos=wp.vec3(*self.position),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=tet_indices,
            density=1e3, # self.material.density,
            k_mu=1e8, # Resistance to shear deformation (first Lame param)
            k_lambda=1e8, # Resistance to volume change   (second Lame param)
            k_damp=1e3, # Damping coefficent (stiffness against damping)
        )

    def to_pyvista(self, state):      
        # Extract particles, from state
        end_index = self.particle_start_index + self.num_particles  
        particle_positions = state['particle_q'][self.particle_start_index:end_index]  
          
        # Update mesh with deformed positions  
        updated_mesh = self.pv_mesh.copy()  

        updated_mesh.points = particle_positions.numpy()

        return updated_mesh
