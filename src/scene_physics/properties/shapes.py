import warp as wp
import newton
import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation

from scene_physics.properties.material import Material
from scene_physics.properties.priors import Priors

DEFAULT_MASS = 0.0
DEFAULT_POSITION = wp.vec3(0., 0., 0.)
DEFAULT_QUAT = wp.quat_identity()
DEFAULT_PRIOR = Priors()


class Parallel_Mesh:
    """
    A mesh object that can be inserted into multiple parallel worlds.

    Each world gets its own instance of this body in the simulation. The
    `allocs` list tracks the body_q index for each world this object was
    inserted into, allowing positions to be read and written by world index.
    """

    OFF_POSITION = np.array((0., -1_000., 0.))

    def __init__(self, body_file, material, name, mass=DEFAULT_MASS, position=DEFAULT_POSITION, quat=DEFAULT_QUAT, prior=DEFAULT_PRIOR):
        """
        Declared for each body to manage itself across many different worlds.
        Key functionality is moving the body across many different worlds.

        Args:
            body_file (str): Path to location of mesh.
            material (Material): Specified in materials class.
            name (str): Name of object.
            mass (float, optional): Mass additional to density calc of object.
            position (wp.vec3, optional): Target position of the object.
            quat (wp.quat, optional): Target rotation of the object.
            prior (Priors, optional): Tells us how to sample.
        """
        # Physical properties
        self.mass = mass
        self.position = position
        self.quat = quat
        self.cfg = material.to_cfg()
        self.name = name
        self.priors = prior

        # Load and convert mesh representations
        self.pv_mesh = self._load_mesh(body_file)
        self.nt_mesh = self._convert_mesh()

        # Parallel world tracking — populated by insert_object calls
        self.finalized = False
        self.mw = None
        self.allocs = []
        self.num_worlds = None

        # Saved physics properties for freeze/unfreeze
        self.inv_mass = None
        self.inv_inertia = None
        self.final_position = None

        # Body state flags
        self.is_frozen = False
        self.is_locked = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def target_position(self):
        """Position where this object should be placed."""
        return self.position

    @property
    def target_quat(self):
        """Rotation this object should have at its target."""
        return self.quat

    @property
    def target_quaternian(self):
        """Deprecated: use target_quat instead."""
        return self.quat

    @property
    def target_pose(self):
        """7-element [x, y, z, qx, qy, qz, qw] array for the target transform."""
        return np.concatenate((np.array(self.position), np.array(self.quat)))

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
        self.num_worlds = self.mw.num_worlds

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

    def freeze_finalized_body(self):
        """
        Zeros mass and inertia and moves all bodies to OFF_POSITION,
        effectively removing them from the physics simulation.
        """
        assert self.finalized, "Must finalize body before freezing"
        assert not self.is_frozen, "Body is already frozen"
        assert not self.is_locked, "Body is locked"

        # Save and zero inv_mass
        inv_mass = self.mw.body_inv_mass.numpy()
        self.inv_mass = inv_mass[self.allocs].copy()
        inv_mass[self.allocs] = 0.0
        self.mw.body_inv_mass = wp.array(inv_mass, dtype=wp.float32, device="cuda")

        # Save and zero inv_inertia
        inv_inertia = self.mw.body_inv_inertia.numpy()
        self.inv_inertia = inv_inertia[self.allocs].copy()
        inv_inertia[self.allocs] = 0.0
        self.mw.body_inv_inertia = wp.array(inv_inertia, dtype=wp.mat33, device="cuda")

        # Move to off-screen position
        bodies = self.mw.body_q.numpy()
        bodies[self.allocs, 0:3] = self.OFF_POSITION
        self.mw.body_q = wp.array(bodies, dtype=wp.transformf, device="cuda")
        self.is_frozen = True

    def unfreeze_finalized_body(self):
        """Restores mass and inertia, moves bodies to random positions."""
        assert self.finalized, "Must finalize body before unfreezing"
        assert self.inv_mass is not None, "Must freeze before unfreezing"
        assert not self.is_locked, "Body is locked"

        # Restore inv_mass
        inv_mass = self.mw.body_inv_mass.numpy()
        inv_mass[self.allocs] = self.inv_mass
        self.mw.body_inv_mass = wp.array(inv_mass, dtype=wp.float32, device="cuda")

        # Restore inv_inertia
        inv_inertia = self.mw.body_inv_inertia.numpy()
        inv_inertia[self.allocs] = self.inv_inertia
        self.mw.body_inv_inertia = wp.array(inv_inertia, dtype=wp.mat33, device="cuda")

        # Move to random positions
        bodies = self.mw.body_q.numpy()
        bodies[self.allocs, 0:3] = np.random.normal(0., scale=1., size=(len(self.allocs), 3))
        self.mw.body_q = wp.array(bodies, dtype=wp.transformf, device="cuda")
        self.is_frozen = False

    def move_to_target(self, scene):
        """Move this object to its target pose in the given scene."""
        assert self.finalized, "Can only move to target once finalized"
        self.move_6dof_wp(self.target_pose, scene)

    def place_final_position(self, world_index, scene):
        """Lock this object at the position from the given world."""
        assert self.finalized, "Can only do on a finalized body"
        assert not self.is_locked, "Body has already been locked"
        self.is_locked = True
        self.final_position = scene.body_q.numpy()[self.allocs[world_index]]

    # ------------------------------------------------------------------
    # Position access and manipulation
    # ------------------------------------------------------------------

    def set_proposal(self):
        """Returns (priors, num_worlds) for the proposal generator."""
        return self.priors, self.num_worlds

    def get_positions(self, state):
        """Return body transforms for all worlds from the given state."""
        return state.body_q.numpy()[self.allocs]

    def move_6dof_wp(self, prop_pos, scene):
        """
        Write proposed positions into the scene's body_q array.

        Args:
            prop_pos : np.array of positions to place body at
            scene    : Newton State
        """
        assert not self.is_locked, "Body is locked"
        bodies = scene.body_q.numpy()
        bodies[self.allocs] = prop_pos
        scene.body_q = wp.array(bodies, dtype=wp.transformf, device="cuda")

    # ------------------------------------------------------------------
    # Mesh loading and conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mesh(file):
        """Accept either a file path or an already-loaded PyVista mesh."""
        return file if isinstance(file, pv.core.pointset.PolyData) else pv.read(file)

    def _convert_mesh(self, triangulate=True, compute_inertia=True, is_solid=True, maxhullvert=10_000):
        """Convert the PyVista mesh into a Newton collision mesh."""
        mesh = self.pv_mesh.extract_surface().clean()
        if triangulate:
            mesh = mesh.triangulate()
        mesh.compute_normals(inplace=True, consistent_normals=True, auto_orient_normals=True)
        verts = mesh.points.astype(np.float32)
        faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
        mesh_kwargs = dict(compute_inertia=compute_inertia, is_solid=is_solid)
        if maxhullvert is not None:
            mesh_kwargs["maxhullvert"] = maxhullvert
        return newton.Mesh(verts, faces, **mesh_kwargs)

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
        position = transform[0:3]
        quat_xyzw = transform[3:7]

        if np.linalg.norm(quat_xyzw) < 1e-10:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])

        rotation = Rotation.from_quat(quat_xyzw)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation.as_matrix()
        transform_matrix[:3, 3] = position

        assert np.all(np.isfinite(transform_matrix)), "Transform matrix contains non-finite values!"
        mesh.transform(transform_matrix, inplace=True)
        return mesh

    def set_final_position_to_target(self):
        """Set final_position to the target pose for rendering."""
        self.final_position = self.target_pose

    def to_pyvista_final(self, *_):
        """Return mesh at its final position after sampling."""
        assert self.final_position is not None, "Position must be finalized first"
        return self.pyvista_body(self.final_position)

    def to_pyvista(self, numpy_bd_q, world_id):
        """
        Return the mesh positioned for a given world.

        Args:
            numpy_bd_q : np.array [total_bodies, 7]
            world_id   : which world to render
        """
        return self.pyvista_body(numpy_bd_q[self.allocs[world_id]])

    def __str__(self):
        return self.name


class Parallel_Static_Mesh(Parallel_Mesh):
    """
    A mesh shared across all worlds — inserted once globally rather than
    once per world. Useful for static scene elements like floors or walls.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_position = np.array([0., 0., 0., 0., 0., 0., 1.])

    def _convert_mesh(self):
        """Use triangle mesh collision for static bodies.

        Static bodies have complex geometry requiring exact collision
        normals. compute_inertia is unnecessary since they don't move.
        """
        return super()._convert_mesh(
            triangulate=False, compute_inertia=False,
            is_solid=False, maxhullvert=None,
        )

    def insert_object_static(self, mw):
        """
        Insert as a global (world -1) object shared by all worlds.

        Args:
            mw : Newton MultiWorld builder
        """
        mw.current_world = -1
        self.allocs = [len(mw.body_q)]
        body = mw.add_body(xform=wp.transform(self.position, self.quat))
        mw.add_shape_mesh(body=body, mesh=self.nt_mesh, cfg=self.cfg)

    def insert_object(self, mw, i):
        raise TypeError("Use insert_object_static — static meshes are inserted once for all worlds.")

    def set_final_position_to_target(self):
        raise TypeError("Cannot use set_final_position_to_target on static mesh")

    def to_pyvista(self, numpy_bd_q, world_id):
        """Return mesh at its single global position (world_id ignored)."""
        return self.pyvista_body(numpy_bd_q[self.allocs[0]])
