"""
Generates sampling for 6DoF for each object. It uses a scheduler and applies quaternian through small angle proposed changes.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import softmax


def linear_decay(iteration, total_iterations):
    """Linear decay from 1.0 to 0.1 over total_iterations."""
    return max(0.1, 1.0 - 0.9 * iteration / total_iterations)


def exponential_decay(iteration, half_life=50):
    """Exponential decay with configurable half-life."""
    return max(0.1, np.exp(-0.693 * iteration / half_life))


class SixDOFProposal:
    """"""

    def __init__(self, obj, x_lower_bound=-1.0, x_upper_bound=1.0, z_lower_bound=-1.0, z_upper_bound=1.0, pos_std=0.01, rot_std=0.01, schedule=None, prob_f=None):
        # Object informaion
        self.obj = obj
        self.num = self.obj.num_worlds

        # Values for initilization
        self._init_mean = 0.00 # TODO better way to set
        self._init_std = 0.05 # TODO better way to set

        # for next proposals
        self.pos_std_base = pos_std
        self.rot_std_base = rot_std
        self.schedule = schedule

        # bounding box for distibution
        self.x_bounds = {"lower" : x_lower_bound, "upper" : x_upper_bound}
        self.z_bounds = {"lower" : z_lower_bound, "upper" : z_upper_bound}

        # Selection function
        self.prob_f = self._rank_proposals if prob_f is None else prob_f

    def get_std(self, iteration, total_iterations=None):
        """Get current (pos_std, rot_std) after applying schedule."""
        if self.schedule is None:
            return self.pos_std_base, self.rot_std_base
        scale = self.schedule(iteration, total_iterations)
        return self.pos_std_base * scale, self.rot_std_base * scale

    def initial_positions(self):
        """
        Initilize positions for object being sampled

        - Positions are on Gaussian
        - Quaternian is vertical for right now
        """

        positions = np.zeros((self.num, 7))
        positions[:, :3] = np.random.normal(loc=self._init_mean, scale=self._init_std, size=(self.num, 3))

        # Ensure Y axis > 0, place vertical
        positions[:, 1] = np.abs(positions[:, 1])
        positions[:, 6] = 1.0

        return positions

    def _rank_proposals(self, scores):
        """
        Returns:
            positions: (self.num_world,) to add noise to
        """
        ranks = np.argsort(np.argsort(scores))
        return np.array(softmax(ranks.astype(float)))

    def _raw_prob_proposals(self, scores):
        """
        Returns:
            positions: [N, 7] to add noise to
        """
        # TODO implement
        pass


    def propose_batch(self, pos, scores, cur_it, total_it):
        """
        Generate 6DOF proposals for the current object

        Args:
            current_pos: [5, 7] array - top 5 current (x, y, z, qx, qy, qz, qw)
            scores: (n, ) numpy array of relative scores
            iterations: current MCMC iterations (for scheduling)

        Returns:
            positions: [N, 7] numpy array of options
        """
        num_proposals = self.num 

        print(scores)

        # Calculate probabilites and select randomly
        idx = np.random.choice(len(pos), size=self.num, p=self.prob_f(scores))
        positions = pos[idx]
        
        # Apply Gaussian Noise to placement
        pos_std, rot_std = self.get_std(cur_it, total_it)
        positions[:, :3] = positions[:, :3] + np.random.normal(0, pos_std, size=(num_proposals, 3))

        
        # TODO make this in parrallel somehow
        for i in range(num_proposals):
            positions[i, 3:] = self._perturb_rotation(positions[i, 3:].squeeze(), rot_std)

        return positions

    @staticmethod
    def _perturb_rotation(current_quat, rot_std):
        """Apply a small random rotation perturbation to a quaternion.

        Args:
            current_quat: (4,) array in (qx, qy, qz, qw) format
            rot_std: standard deviation of axis-angle perturbation (radians)

        Returns:
            (4,) numpy array in (qx, qy, qz, qw) format, unit normalized
        """
        # Sample small axis-angle perturbation
        axis_angle = np.random.normal(0, rot_std, size=3)
        perturbation = Rotation.from_rotvec(axis_angle)

        # Compose with current rotation
        current_rot = Rotation.from_quat(np.asarray(current_quat))
        new_rot = perturbation * current_rot

        # Return as (qx, qy, qz, qw) — scipy default
        return new_rot.as_quat()
