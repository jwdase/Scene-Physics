"""
Physics-based likelihood for parallel proposals.

Camera + target point cloud are produced upstream (SingleWorldCamera /
MultiWorldCamera) and injected here. This class owns only what must persist
across the inner sampling loop: the XPBD solver, pre-allocated state buffers,
and the baseline score.
"""

import jax
import numpy as np
import warp as wp
from scipy.stats import rankdata
from newton.solvers import SolverXPBD

from scene_physics.likelihood.likelihoods_functions import (
    compute_likelihood_score,
    compute_likelihood_score_batch,
)
from scene_physics.properties.structs import Object_Collection
from scene_physics.properties.shapes import Dynamic
from scene_physics.configs.likely import *

# PHYSICS SIMULATION
DEFAULT_SOLVER_ITERATIONS = 16

# LIKELIHOOD FUNCTION INFORMATION
DEFAULT_EVAL_EVERY = 20
DEFAULT_FORWARD_FRAMES = 50
DEFAULT_SIM_DELTA_TIME = 0.05
DEFAULT_SUBSTEPS = 4


class ParallelPhysicsLikelihood:
    """Score parallel proposals against an observed point cloud.

    The camera (a MultiWorldCamera) and the target point cloud are passed in.
    Two scoring modes:
      - still(scene):   render proposals as-placed, score vs. target.
      - physics(scene): forward-sim N frames, score snapshots, average.
    """

    def __init__(
        self,
        camera,                       # MultiWorldCamera
        target_point_cloud: jax.Array,
        model,
        *,
        dt: float = DEFAULT_SIM_DELTA_TIME,
        frames: int = DEFAULT_FORWARD_FRAMES,
        substeps: int = DEFAULT_SUBSTEPS,
        eval_every: int = DEFAULT_EVAL_EVERY,
        likelihood_f=compute_likelihood_score,
        solver=None,
    ):
        self.camera = camera
        self.target_point_cloud = target_point_cloud
        self.model = model
        self.num_worlds = camera.num_worlds

        self.dt = dt
        self.frames = frames
        self.substeps = substeps
        self.sub_dt = dt / substeps
        self.eval_every = eval_every
        self.likelihood_f = likelihood_f

        self._build_solver(solver)
        self._build_state_buffers()

        self.baseline_score = float(
            likelihood_f(
                observed_xyz=target_point_cloud,
                rendered_xyz=target_point_cloud,
            )
        )

    @property
    def _num_eval_points(self):
        return self.frames // self.eval_every

    def _build_solver(self, solver):
        self.control = self.model.control()
        if solver is not None:
            self.solver = solver
        else:
            self.solver = SolverXPBD(
                self.model,
                iterations=DEFAULT_SOLVER_ITERATIONS,
            )

    def _build_state_buffers(self):
        # Ping-pong states for stepping
        self._state_0 = self.model.state()
        self._state_1 = self.model.state()

        # CPU snapshots of body_q at each eval point
        body_q_shape = self._state_0.body_q.numpy().shape
        self._eval_states = np.empty((self._num_eval_points,) + body_q_shape)

        # Reusable state for replaying snapshots through the camera
        self._render_state = self.model.state()

    def still(self, scene):
        """Render proposals as-placed, return (num_worlds,) scores - baseline."""
        clouds = self.camera.render(scene)
        scores = compute_likelihood_score_batch(
            observed_xyz=self.target_point_cloud,
            rendered_xyz_batch=clouds,
        )
        return np.asarray(scores) - self.baseline_score
    
    def _get_penetration(self, scene, object_collection : Object_Collection):
        # TODO build sampler

        return np.zeros(self.num_worlds)

    def run_simulation(self, scene):
        state_0, state_1 = self._state_0, self._state_1
        state_0.assign(scene)
        state_1.assign(scene)

        for frame in range(self.frames):
            for _ in range(self.substeps):
                state_0.clear_forces()
                contacts = self.model.collide(state_0)
                self.solver.step(
                    state_0, state_1, self.control, contacts, self.sub_dt,
                )
                state_0, state_1 = state_1, state_0

        return state_0

    def physics(self, scene, object_collection : Object_Collection):
        """Generates likelihood for each world preserving order"""
        get_rank = lambda a : rankdata(a, method="average") - 1.0

        # Get still render: Want HIGH simularity
        sim = get_rank(self.still(scene))

        # Get initial contacts: Want LOW contacts
        pen = get_rank(-self._get_penetration(scene, object_collection)) # Want LOW 

        # Get physics results
        disp, rot = self._get_physics_results(scene, object_collection)
        disp = get_rank(-disp)
        rot = get_rank(-rot)

        return (
            LAMBDA_POINT * sim / sim.sum() +
            LAMBDA_PENETRATION * pen / pen.sum() +
            LAMBDA_MOVEMENT * disp / disp.sum() +
            LAMBDA_ROT * rot / rot.sum()
            )


    def _get_physics_results(self, scene, object_collection : Object_Collection):
        """ Run's physics scene and calculates final displacement"""
        # Get pre_physics positions (world, body, 7)
        initial_q = np.stack([
            obj.get_positions(scene)
            for obj in 
            object_collection.objects.values()
            if isinstance(obj, Dynamic)
        ], axis=1)

        # Run Physics Simulation
        final_scene = self.run_simulation(scene)

        # Get post simulation positions (world, body, 7)
        final_q = np.stack([
            obj.get_positions(final_scene)
            for obj in
            object_collection.objects.values()
            if isinstance(obj, Dynamic)
        ], axis=1)

        return (
            self._total_displacement(initial_q, final_q),
            self._total_rotation(initial_q, final_q)
        )

    def _total_displacement(self, initial, final):
        displacement = np.abs(initial[..., :3] - final[..., :3]) # (world, body, 3)
        l2_distance = np.linalg.norm(displacement, axis=-1) # (world, body)

        return l2_distance.sum(axis=-1) # (num_worlds,)

    def _total_rotation(self, initial, final):
        # Rotational distance between two quaternians is <q1, q2> = cos(\theta)
        # \theta = 2 * arccos(|<q1, q2>|)

        dot = np.abs(np.sum(initial[..., 3:7] * final[..., 3:7], axis=-1))  # (world, body)
        theta = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))

        return theta.sum(axis=-1)  # (world,)
