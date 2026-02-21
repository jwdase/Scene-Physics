import warp as wp
import newton
import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation

from scene_physics.properties.material import Material


class Parallel_Mesh:
    """
    A mesh object that can be inserted into multiple parallel worlds.

    Each world gets its own instance of this body in the simulation. The
    `allocs` list tracks the body_q index for each world this object was
    inserted into, allowing positions to be read and written by world index.
    """

    def __init__(self, body_file, material=None, mass=None, position=None, quat=None, name=None):
        # Physical properties
        self.mass = 0.0 if mass is None else mass
        self.position = wp.vec3(0., 0., 0.) if position is None else wp.vec3(*position)
        self.quat = quat if quat is not None else wp.quat_identity()
        self.cfg = Material().to_cfg() if material is None else material.to_cfg()
        self.name = name

        # Load and convert mesh representations
        self.pv_mesh = self._load_mesh(body_file)   # PyVista mesh for visualization
        self.nt_mesh = self._convert_mesh()          # Newton mesh for simulation

        # Parallel world tracking — populated by insert_object calls
        self.finalized = False
        self.mw = None      # Reference to the finalized MultiWorld
        self.allocs = []    # body_q index for each world this object was inserted into

    # ------------------------------------------------------------------
    # World management
    # ------------------------------------------------------------------

    def give_finalized_world(self, mw_f):
        """
        Called after all worlds are built and the MultiWorld is finalized.
        Locks in the world reference and converts allocs to a numpy array
        for efficient indexing.
        """
        self.finalized = True
        self.mw = mw_f
        self.allocs = np.array(self.allocs)

    def insert_object(self, mw, i):
        """
        Insert this body into world `i` of a MultiWorld builder.
        Records the body_q index in `allocs` so positions can be retrieved
        later by world index.

        Args:
            mw : Newton MultiWorld builder
            i  : World index to insert into
        """
        mw.current_world = i
        self.allocs.append(len(mw.body_q))

        body = mw.add_body(xform=wp.transform(self.position, self.quat))
        mw.add_shape_mesh(body=body, mesh=self.nt_mesh, cfg=self.cfg)

    # ------------------------------------------------------------------
    # Position access and manipulation
    # ------------------------------------------------------------------

    def _get_positions(self):
        """Return body transforms for all worlds from the finalized MultiWorld."""
        return self.mw.body_q.numpy()[self.allocs]

    def move_6dof_wp(self, body_q_np, prop_pos):
        """
        Update this object's position across all worlds in a body_q array.

        Args:
            body_q_np : np.array [total_bodies, 7] — full flattened body state
            prop_pos  : np.array [num_worlds, 7] — proposed transforms, one per world

        Returns:
            body_q_np with this object's entries updated
        """
        body_q_np[self.allocs] = prop_pos
        return body_q_np

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def _print_positions(self, pos):
        """Print each world's position for this object."""
        print(f"Object '{self.name}' positions across worlds:")
        for i, p in enumerate(pos):
            print(f"  World {i}: {p}")

    def print_positions_final(self):
        """Print positions from the finalized MultiWorld state."""
        pos = self.mw.body_q.numpy()[self.allocs]
        self._print_positions(pos)

    def print_positions_scene(self, numpy_bd_q):
        """Print positions from an external body_q numpy array."""
        pos = numpy_bd_q[self.allocs]
        self._print_positions(pos)

    # ------------------------------------------------------------------
    # Mesh loading and conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mesh(file):
        """Accept either a file path or an already-loaded PyVista mesh."""
        return file if isinstance(file, pv.core.pointset.PolyData) else pv.read(file)

    def _convert_mesh(self):
        """Convert the PyVista mesh into a Newton collision mesh."""
        self.pv_mesh.compute_normals(inplace=True)
        verts = self.pv_mesh.points.astype(np.float32)
        faces = self.pv_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
        return newton.Mesh(verts, faces, compute_inertia=True, is_solid=True, maxhullvert=64)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def pyvista_body(self, transform):
        """
        Return a copy of this mesh transformed to the given pose.

        Args:
            transform : (7,) array — [x, y, z, qx, qy, qz, qw]

        Returns:
            Transformed pv.PolyData mesh
        """
        mesh = self.pv_mesh.copy()

        position = transform[0:3]       # [x, y, z]
        quat_xyzw = transform[3:7]      # [qx, qy, qz, qw]

        rotation = Rotation.from_quat(quat_xyzw)
        rotation_matrix = rotation.as_matrix()

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position

        assert np.all(np.isfinite(transform_matrix)), "Transform matrix contains non-finite values!"

        mesh.transform(transform_matrix, inplace=True)
        return mesh

    def to_pyvista(self, numpy_bd_q, world_id):
        """
        Return the mesh positioned for a given world.

        Args:
            numpy_bd_q : np.array [total_bodies, 7]
            world_id   : which world to render
        """
        pos = numpy_bd_q[self.allocs[world_id]]
        return self.pyvista_body(pos)


class Parallel_Static_Mesh(Parallel_Mesh):
    """
    A mesh shared across all worlds — inserted once globally rather than
    once per world. Useful for static scene elements like floors or walls.
    """

    def insert_object_static(self, mw):
        """
        Insert this body as a global (world -1) object shared by all worlds.
        Unlike Parallel_Mesh.insert_object, this is called once only.

        Args:
            mw : Newton MultiWorld builder
        """
        mw.current_world = -1
        self.allocs = [len(mw.body_q)]     # wrap in list so give_finalized_world works

        body = mw.add_body(xform=wp.transform(self.position, self.quat))
        mw.add_shape_mesh(body=body, mesh=self.nt_mesh, cfg=self.cfg)

    def insert_object(self, mw, i):
        raise TypeError("Use insert_object_static — static meshes are inserted once for all worlds.")

    def to_pyvista(self, numpy_bd_q, world_id=None):
        """
        Return the mesh at its single global position.
        world_id is accepted for interface compatibility but ignored.
        """
        pos = numpy_bd_q[self.allocs[0]]
        return self.pyvista_body(pos)


class MeshBody(Parallel_Mesh):
    """
    A Parallel_Mesh restricted to a single world.

    Simplifies the interface when parallel worlds are not needed —
    insert_object takes no world index, and to_pyvista always uses world 0.
    """

    def insert_object(self, mw):
        """
        Insert this body into world 0.

        Args:
            mw : Newton MultiWorld builder
        """
        super().insert_object(mw, 0)

    def to_pyvista(self, numpy_bd_q, world_id=0):
        """
        Return the mesh at its single-world position.
        world_id is accepted for interface compatibility but always uses world 0.
        """
        return super().to_pyvista(numpy_bd_q, 0)
