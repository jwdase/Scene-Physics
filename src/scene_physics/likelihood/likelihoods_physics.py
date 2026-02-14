"""
This is used to build the physics likelihood function, it integrates forward physics and our jax likelihood function
"""

import warp as wp
from newton import State
from newton.solvers import SolverXPBD

from scene_physics.likelihood.likelihoods import compute_likelihood_score
from scene_physics.kernels.image_process import render_point_cloud


class Likelihood_Physics:
    """
    Likelihood function written over a forward run of
    of a physics engine
    """

    def __init__(
        self,
        target_state,
        model,
        camera,
        max_depth=None,
        likelihood_f=None,
        height=None,
        width=None,
        dt=None,
        likelihood=None,
        frames=None,
        solver=None,
    ):
        self.target_state = target_state

        # Information for likelihood function
        self.likelihood_f = (
            compute_likelihood_score if likelihood is None else likelihood
        )

        # Information for Physics Engine
        self.model = model
        self.control = model.control()
        self.dt = 0.05 if dt is None else dt
        self.solver = self._get_solver() if solver is None else solver
        self.frames = 300 if frames is None else frames

        # Information for Rendering Point Cloud
        self.height = 480 if height is None else height
        self.width = 640 if width is None else width
        self.max_depth = 10.0 if max_depth is None else max_depth
        self.sensor = camera["sensor"]
        self.camera_transforms = camera["camera_transforms"]
        self.camera_rays = camera["camera_rays"]


        # Setup warp buffers for rendering
        self.depth_image = self.sensor.create_depth_image_output()      # Warp depth buffer (1, 1, H*W)
        self.points_gpu = wp.empty(self.depth_image.shape, dtype=wp.vec3f)  # Warp points buffer (1, 1, H*W)

        # Render correct scene to get baseline point cloud
        self.correct_pointcloud = self._render_target()                 # jnp (H, W, 3)
        self.render_fn = self._get_render_fn()                          # Define rendering function
        self.baseline_score = self._compute_baseline()                  # Baseline score


    def _render_target(self):
        """Render the target state into a (H, W, 3) point cloud."""
        return render_point_cloud(
            self.sensor,
            self.target_state,
            self.camera_transforms,
            self.camera_rays,
            self.depth_image,
            self.points_gpu,
            self.height,
            self.width,
            self.max_depth,
        )

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
                self.camera_transforms,
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
        return self._calc_likelihood(self.correct_pointcloud)

    def _calc_likelihood(self, point_cloud):
        """Used to calculate a likelihood in a given situation"""
        return self.likelihood_f(
                observed_xyz=self.correct_pointcloud,
                rendered_xyz=point_cloud
                )

    def new_proposal_likelihood(self, scene):
        """Runs a forward pass on physics to compute likelihood"""

        likelihoods = []  # Collect data on likelihoods
        
        # Copies states
        state_0, state_1 = self.model.state(), self.model.state()
        state_0.assign(scene)
        state_1.assign(scene)

        # Run forward physics
        for frame in range(self.frames):
            print(f"Complete Frame: {frame}")
            state_0.clear_forces()
            contacts = self.model.collide(state_0)

            # Solver integrates from state_0 --> state_1
            self.solver.step(state_0, state_1, self.control, contacts, self.dt)

            # Get likelihood value from render_fn
            likelihoods.append(self._calc_likelihood(self.render_fn(state_1)))
            state_0, state_1 = state_1, state_0

        # Perform calculation on new likelihood
        return self.scaling_likelihood_f(likelihoods)

    def scaling_likelihood_f(self, likelihoods):
        """Create an aggregate of likelihood value"""
        return sum(likelihoods) / len(likelihoods)



