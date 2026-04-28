"""
Generates sampling for 6DoF for each object. It uses a scheduler and applies quaternion through small angle proposed changes.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def linear_decay(iteration, total_iterations):
    """Linear decay from 1.0 to 0.1 over total_iterations."""
    return max(0.1, 1.0 - 0.9 * iteration / total_iterations)


def exponential_decay(iteration, half_life=50):
    """Exponential decay with configurable half-life."""
    return max(0.1, np.exp(-0.693 * iteration / half_life))


class Prior:
    def __init__(self, json : dict):
        self.pos_x = json["position"][0]
        self.pos_y = json["position"][1]
        self.pos_z = json["position"][2]

        self.quat_x = json["position"][3]
        self.quat_y = json["position"][4]
        self.quat_z = json["position"][5]
        self.quat_w = json["position"][6]

        self.init_std_pos = json["pos_std"]
        self.init_std_rot = json["rot_std"]

        self.x_max = json["x_max"]
        self.x_min = json["x_min"]
        self.y_max = json["y_max"]
        self.y_min = json["y_min"]

    @property
    def xform(self):
        return [
                self.pos_x, self.pos_y, self.pos_z,
                self.quat_x, self.quat_y, self.quat_z,
                self.quat_w
                ]

class Proposer:
    """
    Proposer Class must be able to generate initial proposals
    and proposals after initial values
    """
    def __init__(self, rand_seed, prior : Prior):
        self.rng = np.random.default_rng(rand_seed)
        self.prior = prior
        
        self._cur_pos_std = prior.init_std_pos
        self._cur_rot_std = prior.init_std_rot

    def _gen_rotations(self, quat : np.array):
        n, _ = quat.shape
        angles = self.rng.normal(0.0, self._cur_rot_std, size=(n, 3))

        perturbation = Rotation.from_rotvec(angles)
        current = Rotation.from_quat(quat)

        return (perturbation * current).as_quat()

    def _update(self):
        self._update_pos_std()
        self._update_rot_std()

    def _update_pos_std(self):
        raise NotImplementedError("Not Implemented in General Class")

    def _update_rot_std(self):
        raise NotImplementedError("Not Implemented in General Class")

    def _apply_bounds(self, proposals):
        proposals[:, 0] = np.clip(proposals[:, 0], a_min=self.prior.x_min, a_max=self.prior.x_max)
        proposals[:, 1] = np.clip(proposals[:, 1], a_min=self.prior.y_min, a_max=self.prior.y_max)
        return proposals

    def initial_proposal(self, num_worlds : int):
        np_xform = np.array(self.prior.xform)
        proposals = np.tile(np_xform, (num_worlds, 1))

        proposals[1:, :3] += self.rng.normal(loc=0.0, scale=self._cur_pos_std, size=(num_worlds - 1, 3))
        proposals[1:, 3:] = self._gen_rotations(proposals[1:, 3:])
        
        self._update()

        return self._apply_bounds(proposals)

    def propose(self, positions, likelihood):
        n = len(likelihood)
        proposals = np.zeros((n, 7))

        # Keep Top Likelihood
        proposals[0, :] = positions[np.argmax(likelihood)]
        
        weights = likelihood / np.sum(likelihood)
        idx = self.rng.choice(n, size=n-1, p=weights, replace=True)
        proposals[1:, :] = positions[idx]

        # Add Gaussian Noise
        proposals[1:, :3] += self.rng.normal(loc=0.0, scale=self._cur_pos_std, size=(n - 1, 3))
        proposals[1:, 3:] = self._gen_rotations(proposals[1:, 3:])

        self._update()

        return self._apply_bounds(proposals)

class NoDecayProposal(Proposer):
    def _update_pos_std(self):
        pass
    
    def _update_rot_std(self):
        pass

