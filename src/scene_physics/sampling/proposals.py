"""
Generates sampling for 6DoF for each object. It uses a scheduler and applies quaternian through small angle proposed changes.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def linear_decay(iteration, total_iterations):
    """Linear decay from 1.0 to 0.1 over total_iterations."""
    return max(0.1, 1.0 - 0.9 * iteration / total_iterations)


def exponential_decay(iteration, half_life=50):
    """Exponential decay with configurable half-life."""
    return max(0.1, np.exp(-0.693 * iteration / half_life))


class SixDOFProposal:
    """"""

    def __init__(self, obj, pos_std=0.05, rot_std=0.1, schedule=None):
        self.pos_std_base = pos_std
        self.rot_std_base = rot_std
        self.schedule = schedule
        self.obj = obj
        self.num = self.obj.num_worlds
        self._init_mean = 0.00 # TODO better way to set
        self._init_std = 0.05 # TODO better way to set

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
        positions[:, 6] = 1.0

        return positions


    def propose_batch(self, pos, scores, cur_it, total_it):
        """
        Generate 6DOF proposals for the current object

        Args:
            current_pos: [5, 7] array - top 5 current (x, y, z, qx, qy, qz, qw)
            iterations: current MCMC iterations (for scheduling)

        Returns:
            positions: [N, 7] numpy array of options
        """
        n = 5
        num_proposals = self.num 
        
        # Get top n positions and then sample forward
        indices = np.argpartition(scores, -n)[-n:]
        positions = np.repeat(pos[indices], ((num_proposals // n) + 1), axis=0)
        positions = positions[:num_proposals]
        
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
