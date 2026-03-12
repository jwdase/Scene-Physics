"""
This is used to build the physics likelihood function, it integrates forward physics and
likelihood function defined in JAX.
"""

import os
import numpy as np
import warp as wp
from newton.solvers import SolverXPBD

from scene_physics.likelihood.likelihoods import (
    compute_likelihood_score,
    compute_likelihood_score_batch,
)
from scene_physics.kernels.image_process import (
    render_point_cloud,
    render_point_clouds_batch,
)
from scene_physics.visualization.camera import setup_depth_camera
from scene_physics.utils.io import save_point_cloud_ply


class Likelihood_Physics_Parallel:
    """
    Physics-informed likelihood for N parallel worlds.

    Runs forward physics once (single solver.step loop processes all worlds),
    then batch-renders and batch-computes likelihoods across all worlds.
    """

    def __init__(
        self,
        target_state,
        model,
        wp_eye,
        wp_target,
        num_worlds,
        name,
        max_depth=None,
        likelihood_f=None,
        height=None,
        width=None,
        dt=None,
        frames=None,
        solver=None,
        eval_every=None,
    ):
        self.name = name
        self.target_state = target_state
        self.num_worlds = num_worlds

        # Likelihood function
        self.likelihood_f = (
            compute_likelihood_score if likelihood_f is None else likelihood_f
        )

        # Physics engine
        self.model = model
        self.control = model.control()
        self.dt = 0.05 if dt is None else dt
        self.solver = self._get_solver() if solver is None else solver
        self.frames = 50 if frames is None else frames
        self.eval_every = 20 if eval_every is None else eval_every

        # Rendering
        self.height = 480 if height is None else height
        self.width = 640 if width is None else width
        self.max_depth = 10.0 if max_depth is None else max_depth

        # Camera setup
        camera_intrinsics = setup_depth_camera(
            self.model, wp_eye, wp_target, self.width, self.height, self.num_worlds
        )
        self.sensor = camera_intrinsics["sensor"]
        self.camera_transforms = camera_intrinsics["camera_transforms"]
        self.camera_rays = camera_intrinsics["camera_rays"]

        # Warp buffers for batch rendering
        self.depth_image = self.sensor.create_depth_image_output()
        self.points_gpu = wp.empty(self.depth_image.shape, dtype=wp.vec3f)

        # Render target point cloud (single-world render of the ground truth)
        self.correct_pointcloud = self._render_target()
        self.baseline_score = self._compute_baseline()

        # Pre-allocate physics states (reused across proposals)
        self._state_0 = self.model.state()
        self._state_1 = self.model.state()

        # Allocate a CPU body_q snapshot array (num_eval_points, *body_q_shape)
        self._num_eval_points = self.frames // self.eval_every
        body_q_shape = self._state_0.body_q.numpy().shape
        self._eval_states = np.empty((self._num_eval_points,) + body_q_shape)

        # Pre-allocate a single reusable GPU render buffer for Phase 2
        self._render_state = self.model.state()

        # Save target
        os.makedirs(f"recordings/{self.name}", exist_ok=True)
        save_point_cloud_ply(
            self.correct_pointcloud, f"recordings/{self.name}/target.ply"
        )

    def _get_solver(self):
        return SolverXPBD(
            self.model,
            rigid_contact_relaxation=0.9,
            iterations=4,
            angular_damping=0.1,
            enable_restitution=False,
        )

    def _render_target(self):
        """Render target state — uses single-world render."""
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
                observed_xyz=self.correct_pointcloud,
                rendered_xyz=self.correct_pointcloud,
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
        # Accumulate likelihoods across eval points
        total_scores = np.zeros(self.num_worlds)
        for i in range(eval_idx):

            # Load snapshot into reusable render state
            self._render_state.body_q = wp.array(
                self._eval_states[i], dtype=wp.transformf, device="cuda"
            )
            batch_clouds = self._render_batch(self._render_state)
            scores = compute_likelihood_score_batch(
                observed_xyz=self.correct_pointcloud,
                rendered_xyz_batch=batch_clouds,
            )
            total_scores += np.asarray(scores)

        # Average across eval points and subtract baseline
        avg_scores = total_scores / eval_idx
        avg_scores -= self.baseline_score

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
            observed_xyz=self.correct_pointcloud,
            rendered_xyz_batch=batch_clouds,
        )

        # Rescale for Collision
        scores = np.asarray(scores) - self.baseline_score

        return scores
