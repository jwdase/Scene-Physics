"""
This is used to build the physics likelihood function, it integrates forward physics and our jax likelihood function
"""
import os
import warp as wp
from newton.solvers import SolverXPBD
from newton._src.utils.recorder import RecorderModelAndState

from scene_physics.likelihood.likelihoods import compute_likelihood_score
from scene_physics.kernels.image_process import render_point_cloud
from scene_physics.visualization.scene import VideoVisualizer
from scene_physics.visualization.camera import setup_depth_camera
from scene_physics.utils.io import save_point_cloud_ply



class Likelihood_Physics:
    """
    Likelihood function written over a forward run of
    of a physics engine
    """

    def __init__(
        self,
        target_state,
        model,
        wp_eye,
        wp_target,
        pv_eye,
        pv_target,
        bodies,
        name,
        max_depth=None,
        likelihood_f=None,
        height=None,
        width=None,
        dt=None,
        likelihood=None,
        frames=None,
        solver=None,
        check_likelihood=None,
        eval_every=None,
    ):

        # Basic information
        self.name = name
        self.target_state = target_state

        # Information for likelihood function
        self.likelihood_f = (
            compute_likelihood_score if likelihood is None else likelihood
        )
        self.check_likelihood = 50 if check_likelihood is None else check_likelihood

        # Information for Physics Engine
        self.model = model
        self.control = model.control()
        self.dt = 0.05 if dt is None else dt
        self.solver = self._get_solver() if solver is None else solver
        self.frames = 50 if frames is None else frames
        self.eval_every = 5 if eval_every is None else eval_every
        self.fps = int(1 / self.dt)

        # Information for Rendering Point Cloud
        self.height = 480 if height is None else height
        self.width = 640 if width is None else width
        self.max_depth = 10.0 if max_depth is None else max_depth
        self.wp_eye = wp_eye
        self.wp_target = wp_target
        self.pv_eye = pv_eye
        self.pv_target = pv_target

        # Build Camera View Settings
        camera_intrinsics = setup_depth_camera(self.model, self.wp_eye, self.wp_target, self.width, self.height)
        self.sensor = camera_intrinsics["sensor"]
        self.camera_transforms = camera_intrinsics["camera_transforms"]
        self.camera_rays = camera_intrinsics["camera_rays"]

        # Setup warp buffers for rendering
        self.depth_image = self.sensor.create_depth_image_output()      
        self.points_gpu = wp.empty(self.depth_image.shape, dtype=wp.vec3f)

        # Render correct scene to get baseline point cloud
        self.correct_pointcloud = self._render_target()
        self.render_fn = self._get_render_fn()
        self.baseline_score = self._compute_baseline()

        # Build recorder for replay
        self.recorder = RecorderModelAndState()
        self.visualizer = VideoVisualizer(self.recorder, bodies, self.fps, camera_position=self._get_camera())

        # Save the target point cloud
        self._save_target()

        # Pre-allocate physics states (reused across proposals)
        self._state_0 = self.model.state()
        self._state_1 = self.model.state()

        # Pre-allocate states for likelihood evaluation at eval points
        self._num_eval_points = self.frames // self.eval_every
        self._eval_states = [self.model.state() for _ in range(self._num_eval_points)]

    def _save_target(self):
        """Saves a copy of target point cloud"""
        os.makedirs(f"recordings/{self.name}", exist_ok=True)
        save_point_cloud_ply(self.correct_pointcloud, f"recordings/{self.name}/target.ply")

    def _get_camera(self):
        """Builds Pyvista Camera Settings"""
        return [tuple(self.pv_eye), tuple(self.pv_target), (0, 1, 0)]

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
            iterations=4,
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

    def new_proposal_likelihood(self, scene, view=False, name=None):
        """Runs a forward pass on physics, then evaluates likelihood on saved states.

        Phase 1: Run all physics frames, saving states every eval_every frames.
        Phase 2: Render + compute likelihood only on saved eval states.
        """

        if view: self.recorder.history.clear()

        # Setup states
        state_0, state_1 = self._state_0, self._state_1
        state_0.assign(scene)
        state_1.assign(scene)

        # Phase 1: Forward physics — save snapshots at eval points
        eval_idx = 0
        for frame in range(self.frames):
            state_0.clear_forces()
            contacts = self.model.collide(state_0)
            self.solver.step(state_0, state_1, self.control, contacts, self.dt)

            if view: self.recorder.record(state_1)

            # Save state at eval points (every eval_every frames)
            if (frame + 1) % self.eval_every == 0:
                self._eval_states[eval_idx].assign(state_1)
                eval_idx += 1

            state_0, state_1 = state_1, state_0

        # Phase 2: Render + likelihood on saved eval states only
        likelihoods = []
        for i in range(eval_idx):
            likelihoods.append(self._calc_likelihood(self.render_fn(self._eval_states[i])))

        if view: self._build_recording(name)

        return self.scaling_likelihood_f(likelihoods)

    def _build_recording(self, name):
        """Code to build rendering of simulation forward pass"""
        self.visualizer.render(f"recordings/{self.name}/{name}.mp4")

    def scaling_likelihood_f(self, likelihoods):
        """Create an aggregate of likelihood value"""
        return sum(likelihoods) / len(likelihoods)
