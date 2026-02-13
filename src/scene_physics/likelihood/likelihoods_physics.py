"""
This is used to build the physics likelihood function, it integrates forward physics and our jax likelihood function
"""

import warp as wp
from newton import State
from newton.solver import SolverXPBD

from scene_physics.likelihood.likelihoods import compute_likelihood_score
from scene_physics.kernels.image_process import render_point_cloud
from scene_physics.utils.io import render_bio


class Likelihood_Physics:
    """
    Likelihood function written over a forward run of
    of a physics engine
    """

    def __init__(
        self,
        correct_state,
        model,
        camera,
        sensor,
        height=None,
        width=None,
        dt=None,
        likelihood=None,
        frames=None,
        solver=None,
    ):
        self.correct_pointcloud = correct_state
        self.baseline_score = (
            self._compute_baseline()
            if likelihood is None
            else likelihood(correct_pointcloud)
        )

        # Information for Physics Engine
        self.model = model
        self.control = model.control
        self.dt = 0.05 if dt is None else dt
        self.solver = self._get_solver() if solver is None else solver
        self.frames = 300 if frames is None else solver

        # Information for Rendering Point Cloud
        self.height = 480 if height is None else height
        self.width = 640 if width is None else width
        self.camera = camera
        self.sensor = sensor
        self.allocate_points = self._initial_buff()

    def _initial_buff(self):
        """Code to setup initial information on scene"""
        
        # Create output buffers
        depth_image = self.sensor.create_depth_image_output()
        color_image = self.sensor.create_color_image_output()
        self.sensor.render(
            self.state,
            camera_transforms,
            camera_rays,
            depth_image=depth_image,
            color_image=color_image,
        )

        return render_bio(depth_image, HEIGHT, WIDTH)

    def _get_solver(self):
        """Returns solver with some built in capacities"""

        solver = SolverXPBD(
            self.model,
            rigid_contact_relaxation=0.9,
            iterations=100,
            angular_damping=0.1,
            enable_restitution=False,
        )

        return solver

    def _get_render_fn(self):

        def render_fn(state):
            return render_point_cloud(
                    self.sensor,
                    state,
                    self.camera,
                    self.camera_rays,
                    self.depth_image,
                    self.points_gpu,
                    self.height,
                    self.width,
                    self.max_depth,
                    )

        return render_fn

    def _compute_baseline(self):
        """Compute the self-comparison score used as normalization baseline."""
        return compute_likelihood_score(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=self.correct_pointcloud,
        )

    def new_proposal_likelihood(self, scene):
        """Runs a forward pass on physics to compute likelihood"""

        likelihoods = []  # Collect data on likelihoods

        state_0 = State().assign(scene)  # Assign first physics frame
        state_1 = State().assign(scene)  # Assign second physics frame
        
        # Allocate points for GPU
        points_gpu = wp.empty((*self.allocate_points), dtype=wp.vec3f)

        for frame in self.frames:
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, self.control, contacts, self.dt)
            state_0, state_1 = state_1, state_0
