"""
6DOF proposal distribution with variance scheduling for MCMC sampling.

Generates batched position + rotation proposals for parallel world evaluation.
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
    """
    Generates batched 6DOF proposals (position + rotation) for parallel MH sampling.

    Position proposals: Gaussian random walk on (x, y, z).
    Rotation proposals: Small axis-angle perturbation composed with current rotation.

    Args:
        pos_std: initial standard deviation for position perturbation (meters)
        rot_std: initial standard deviation for rotation perturbation (radians)
        schedule: callable(iteration, total_iterations) -> scale_factor in (0, 1]
            Applied to both pos_std and rot_std. None = no scheduling.
    """

    def __init__(self, pos_std=0.05, rot_std=0.1, schedule=None):
        self.pos_std_base = pos_std
        self.rot_std_base = rot_std
        self.schedule = schedule

    def get_std(self, iteration, total_iterations=None):
        """Get current (pos_std, rot_std) after applying schedule."""
        if self.schedule is None:
            return self.pos_std_base, self.rot_std_base
        scale = self.schedule(iteration, total_iterations)
        return self.pos_std_base * scale, self.rot_std_base * scale

    def propose_single(self, current_pos, current_quat, iteration=0, total_iterations=None):
        """Generate a single 6DOF proposal around the current state.

        Args:
            current_pos: (3,) array — current (x, y, z)
            current_quat: (4,) array — current (qx, qy, qz, qw)
            iteration: current MCMC iteration (for scheduling)
            total_iterations: total iterations planned (for scheduling)

        Returns:
            new_pos: (3,) numpy array
            new_quat: (4,) numpy array in (qx, qy, qz, qw) format
        """
        pos_std, rot_std = self.get_std(iteration, total_iterations)

        # Position: Gaussian random walk
        new_pos = np.array(current_pos, dtype=np.float64) + np.random.normal(0, pos_std, size=3)

        # Rotation: small axis-angle perturbation composed with current
        new_quat = self._perturb_rotation(current_quat, rot_std)

        return new_pos.astype(np.float32), new_quat.astype(np.float32)

    def propose_batch(self, current_pos, current_quat, num_proposals, iteration=0, total_iterations=None):
        """Generate num_proposals 6DOF proposals around the current state.

        Args:
            current_pos: (3,) array — current (x, y, z)
            current_quat: (4,) array — current (qx, qy, qz, qw)
            num_proposals: number of proposals to generate
            iteration: current MCMC iteration (for scheduling)
            total_iterations: total iterations planned (for scheduling)

        Returns:
            positions: (num_proposals, 3) numpy array
            quats: (num_proposals, 4) numpy array in (qx, qy, qz, qw) format
        """
        pos_std, rot_std = self.get_std(iteration, total_iterations)

        # Batch position proposals: Gaussian random walk
        current_pos = np.asarray(current_pos, dtype=np.float64)
        positions = current_pos[None, :] + np.random.normal(0, pos_std, size=(num_proposals, 3))

        # Batch rotation proposals: axis-angle perturbations
        quats = np.empty((num_proposals, 4), dtype=np.float64)
        for i in range(num_proposals):
            quats[i] = self._perturb_rotation(current_quat, rot_std)

        return positions.astype(np.float32), quats.astype(np.float32)

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
