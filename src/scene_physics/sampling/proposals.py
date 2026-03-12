"""
Generates sampling for 6DoF for each object. It uses a scheduler and applies quaternian through small angle proposed changes.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import softmax

from scene_physics.properties.priors import Priors


def linear_decay(iteration, total_iterations):
    """Linear decay from 1.0 to 0.1 over total_iterations."""
    return max(0.1, 1.0 - 0.9 * iteration / total_iterations)


def exponential_decay(iteration, half_life=50):
    """Exponential decay with configurable half-life."""
    return max(0.1, np.exp(-0.693 * iteration / half_life))


def no_decay(iteration, half_life):
    return 1.0


class SixDOFProposal:
    """"""

    schedules = {"linear": linear_decay, "exp": exponential_decay, "no_decay": no_decay}

    def __init__(self, priors, num_worlds, seed, schedule="no_decay"):
        assert isinstance(priors, Priors), "Expecting object priors passed in"

        # Values for sampling
        self.num = num_worlds
        self.priors = priors

        # TODO Random Seed

        # Our schedulers
        self.schedulers = self._get_scheduler(schedule)
        self.cur_iters = 0

        # Saves values for rot_std and pos_std
        self.save_rot_std = []
        self.save_pos_std = []
        self.epoch_num = []

    def _get_scheduler(self, schedule_type):
        """Choooses our schedulers for 6DoF"""

        assert schedule_type in list(
            self.schedules.keys()
        ), "Scheduler {schedule_type} is not an option"
        return self.schedules[schedule_type]

    def get_std(self):
        """Get current (pos_std, rot_std) after applying schedule."""

        scale = self.schedulers(self.cur_iters, self.priors.total_iter)
        return self.priors.pos_std * scale, self.priors.rot_std * scale

    def initial_positions(self):
        """
        Initilize positions for object being sampled

        - Positions are on Gaussian
        - Quaternian is vertical for right now
        """

        positions = np.zeros((self.num, 7))
        positions[:, :3] = np.random.normal(
            loc=self.priors.init_mean, scale=self.priors.init_std, size=(self.num, 3)
        )

        # Ensure Y axis > 0, place vertical
        positions[:, 1] = np.abs(positions[:, 1])
        positions[:, 6] = 1.0

        return positions

    def _update_epochs(self, pos_std, rot_std, epoch_num):
        """Saves values for epoch in plotting"""

        self.save_pos_std.append(pos_std)
        self.save_rot_std.append(rot_std)
        self.epoch_num.append(epoch_num)

    def propose_general(self, positions, epoch_num, count):
        """
        Add some noise to positions - does not add noise to index=0 where top
        index from previous run is stored.
        """

        pos_std, rot_std = self.get_std()
        positions[1:, :3] = positions[1:, :3] + np.random.normal(
            0, pos_std, size=(self.num - 1, 3)
        )
        positions[:, 0] = np.clip(
            positions[:, 0], a_min=self.priors.x_min, a_max=self.priors.x_max
        )
        positions[:, 2] = np.clip(
            positions[:, 2], a_min=self.priors.z_min, a_max=self.priors.z_max
        )

        for i in range(1, self.num):
            positions[i, 3:] = self._perturb_rotation(
                positions[i, 3:].squeeze(), rot_std
            )

        # Save values in object
        if count is True:
            self._update_epochs(pos_std, rot_std, epoch_num)

        # Update Iterations
        self.cur_iters += 1

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
