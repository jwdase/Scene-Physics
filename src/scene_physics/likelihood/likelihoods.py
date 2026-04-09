"""
This is used to build the physics likelihood function, it integrates forward physics and
likelihood function defined in JAX.
"""

import os
import numpy as np
import warp as wp
from newton.solvers import SolverXPBD

from scene_physics.likelihood.likelihoods_functions import (
    compute_likelihood_score,
    compute_likelihood_score_batch,
)
from scene_physics.kernels.image_process import (
    render_point_cloud,
    render_point_clouds_batch,
)
from scene_physics.visualization.camera import setup_depth_camera
from scene_physics.utils.io import save_point_cloud_ply

# RENDERING INFORMATION
DEFAULT_FOV_DEGREES = 120
DEFAULT_RENDER_DEPTH = 4.0
DEFAULT_RENDER_HEIGHT = 480
DEFAULT_RENDER_WIDTH = 640

# PHYSICS SIMULATION
SOLVER = SolverXPBD
DEFAULT_RIGID_CONTACT_RELAXATION = 0.75
DEFAULT_SOLVER_ITERATIONS = 16
DEFAULT_ANGULAR_DAMPING = 0.2
DEFAULT_ENABLE_RESTITUTION = False

# LIKELIHOOD FUNCTION INFORMATION
DEFAULT_EVAL_EVERY = 20
DEFAULT_FORWARD_FRAMES = 50
DEFAULT_SIM_DELTA_TIME = 0.05
DEFAULT_LIKELIHOOD_FUNCTION = compute_likelihood_score

class ParallelPhysicsLikelihood:
    """
    Across proposals in $n$ worlds returns a likelihood value for each of those
    worlds relative to baseline. 
    """

    def __init__(
        self,
        model,
        objects,
        wp_eye,
        wp_target,
        num_worlds,
        name,
        max_depth=DEFAULT_RENDER_DEPTH,
        fov_degrees=DEFAULT_FOV_DEGREES,
        likelihood_f=DEFAULT_LIKELIHOOD_FUNCTION,
        height=DEFAULT_RENDER_HEIGHT,
        width=DEFAULT_RENDER_WIDTH,
        dt=DEFAULT_SIM_DELTA_TIME,
        frames=DEFAULT_FORWARD_FRAMES,
        solver=None,
        eval_every=DEFAULT_EVAL_EVERY,
    ):
        self.config = self._build_config(
            name=name, objects=objects, model=model,
            num_worlds=num_worlds, max_depth=max_depth,
            fov_degrees=fov_degrees, height=height, width=width,
            wp_eye=wp_eye, wp_target=wp_target,
            dt=dt, frames=frames, eval_every=eval_every,
            likelihood_f=likelihood_f,
        )

        self.name = name
        self.num_worlds = num_worlds
        self.likelihood_f = likelihood_f
        self.model = model

        # Camera, Simulator, and Buffer setup
        self._build_default_camera()        # Setup Camera for Rendering
        self._build_render_buffers()        # Buffers for rendering
        self.target_point_cloud = (
            self._get_target_state()        # Target Pointcloud
        )
        self._build_physics_solver(solver)  # Build the physics solver
        self._build_batch_buffers()         # Leaves space for rendering

        # Rendering
        self.baseline_score = self._compute_baseline()
        self._save_target()

    @property
    def _num_eval_points(self):
        return self.frames // self.eval_every


    @staticmethod
    def _build_config(**kwargs):
        return {
            "name": kwargs["name"],
            "objects": kwargs["objects"],
            "model": kwargs["model"],

            # Render Information
            "num_worlds": kwargs["num_worlds"],
            "max_depth": kwargs["max_depth"],
            "fov_degrees": kwargs["fov_degrees"],
            "height": kwargs["height"],
            "width": kwargs["width"],
            "wp_eye": kwargs["wp_eye"],
            "wp_target": kwargs["wp_target"],

            # Likelihood information
            "dt": kwargs["dt"],
            "frames": kwargs["frames"],
            "eval_every": kwargs["eval_every"],
            "likelihood_f": kwargs["likelihood_f"],
        }

    def _build_default_camera(self):
        """
        SETUP: Build the camera sensor to get the 3d point clouds that
        enable the application of a likelihood function
        """
        self.height = self.config["height"]
        self.width = self.config["width"]
        self.max_depth = self.config["max_depth"]

        camera_intrinsics = setup_depth_camera(
            self.config["model"], 
            self.config["wp_eye"], 
            self.config["wp_target"],
            self.config["width"],
            self.config["height"],
            self.config["num_worlds"],
            self.config["fov_degrees"]
        )
        
        self.sensor = camera_intrinsics["sensor"]
        self.camera_transforms = camera_intrinsics["camera_transforms"]
        self.camera_rays = camera_intrinsics["camera_rays"]

    def _build_render_buffers(self):
        """
        SETUP: Build rendering buffers so pointclouds can be stored
        """
        self.depth_image = self.sensor.create_depth_image_output()
        self.points_gpu = wp.empty(self.depth_image.shape, dtype=wp.vec3f)

    def _get_target_state(self):
        """
        SETUP: Moves all the objects into the correct position
        and returns a state variable on that position
        """

        target_scene = self.config["model"].state()

        for obj in self.config["objects"].all_sampled:
            obj.move_to_target(target_scene)

        point_cloud = render_point_cloud(
            self.sensor,
            target_scene,
            self.camera_transforms,
            self.camera_rays,
            self.depth_image,
            self.points_gpu,
            self.height,
            self.width,
            self.max_depth,
        )

        return point_cloud

    def _build_physics_solver(self, solver=None):
        """
        SETUP: Builds solver for the given states
        """
        self.control = self.config["model"].control()
        self.dt = self.config["dt"]
        self.frames = self.config["frames"]
        self.eval_every = self.config["eval_every"]

        if solver is not None:
            self.solver = solver
        else:
            self.solver = SOLVER(
                self.config["model"],
                rigid_contact_relaxation=DEFAULT_RIGID_CONTACT_RELAXATION,
                iterations=DEFAULT_SOLVER_ITERATIONS,
                angular_damping=DEFAULT_ANGULAR_DAMPING,
                enable_restitution=DEFAULT_ENABLE_RESTITUTION,
            )

    def _build_batch_buffers(self):
        """
        SETUP: Builds the buffers to accelerate speed of 
        rendering
        """

        # State holder for simulation
        self._state_0 = self.config["model"].state()
        self._state_1 = self.config["model"].state()

        body_q_shape = self._state_0.body_q.numpy().shape
        self._eval_states = np.empty((self._num_eval_points,) + body_q_shape)

        # State holder for rendering
        self._render_state = self.config["model"].state()

    def _save_target(self):
        os.makedirs(self.config["name"], exist_ok=True)
        save_point_cloud_ply(
            self.target_point_cloud, f"{self.name}/target.ply"
        )


    ###############################
    ##       FUNCTIONAL CODE    ###
    ###############################

    def _render_batch(self, state):
        """Render all worlds, returning (num_worlds, H, W, 3)."""
        return render_point_clouds_batch(
            self.sensor,
            state,
            self.camera_transforms,
            self.camera_rays,
            self.depth_image,
            self.points_gpu,
            self.height,
            self.width,
            self.max_depth,
            self.num_worlds,
        )

    def _compute_baseline(self):
        return float(
            self.likelihood_f(
                observed_xyz=self.target_point_cloud,
                rendered_xyz=self.target_point_cloud,
            )
        )

    def new_proposal_likelihood_physics_batch(self, scene):
        """Run forward physics on all worlds, then batch render + compute likelihoods.

        Worlds whose initial proposal placement causes a rigid-body collision are
        assigned -inf and skipped in score averaging. Physics still runs over all
        worlds (the solver is a single batched kernel), but colliding worlds are
        excluded from results.

        Args:
            scene: Newton state with all worlds' body_q already set to proposals

        Returns:
            numpy array of shape (num_worlds,) with likelihood scores (relative to baseline),
            -inf for any world with an initial collision.
        """
        state_0, state_1 = self._state_0, self._state_1
        state_0.assign(scene)
        state_1.assign(scene)

        # Phase 1: Forward physics — single solver loop processes all worlds
        eval_idx = 0
        for frame in range(self.frames):
            state_0.clear_forces()
            contacts = self.model.collide(state_0)
            self.solver.step(state_0, state_1, self.control, contacts, self.dt)

            if (frame + 1) % self.eval_every == 0:
                self._eval_states[eval_idx] = state_1.body_q.numpy()
                eval_idx += 1

            state_0, state_1 = state_1, state_0

        # Phase 2: Batch render + batch likelihood on eval states
        total_scores = np.zeros(self.num_worlds)
        for i in range(eval_idx):

            # Load snapshot into reusable render state
            self._render_state.body_q = wp.array(
                self._eval_states[i], dtype=wp.transformf, device="cuda"
            )
            batch_clouds = self._render_batch(self._render_state)
            scores = compute_likelihood_score_batch(
                observed_xyz=self.target_point_cloud,
                rendered_xyz_batch=batch_clouds,
            )
            total_scores += np.asarray(scores)

        # Average across eval points and subtract baseline
        avg_scores = (total_scores / eval_idx) - self.baseline_score

        return avg_scores

    def new_proposal_likelihood_still_batch(self, scene):
        """Batch render all worlds at their current positions, compute likelihoods.

        No physics simulation — evaluates proposals exactly as placed.

        Args:
            scene: Newton state with all worlds' body_q already set to proposals

        Returns:
            numpy array of shape (num_worlds,) with likelihood scores (relative to baseline)
        """

        batch_clouds = self._render_batch(scene)
        scores = compute_likelihood_score_batch(
            observed_xyz=self.target_point_cloud,
            rendered_xyz_batch=batch_clouds,
        )

        # Rescale for Collision
        scores = np.asarray(scores) - self.baseline_score
        return scores