"""
Generates sampling for 6DoF for each object. It uses a scheduler and applies quaternion through small angle proposed changes.
"""

import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, rand_seed, num_worlds, prior : Prior, iterations : int):
        self.rng = np.random.default_rng(rand_seed)
        self.prior = prior
        self.num_worlds = num_worlds
        self.iterations = iterations

        self._cur_pos_std = prior.init_std_pos
        self._cur_rot_std = prior.init_std_rot
        self._cur_iteration = 0
        
        # Save for plotting
        self._pos_std_list = []
        self._rot_std_list = []

    def _gen_rotations(self, quat : np.array):
        n, _ = quat.shape
        angles = self.rng.normal(0.0, self._cur_rot_std, size=(n, 3))

        perturbation = Rotation.from_rotvec(angles)
        current = Rotation.from_quat(quat)

        return (perturbation * current).as_quat()
    
    def propose(self, _, __):
        raise NotImplementedError("Propose must be implemented in child class")

    def _update(self):
        self._pos_std_list.append(self._cur_pos_std)
        self._rot_std_list.append(self._cur_rot_std)

        # Update iteration and stds
        self._cur_iteration += 1
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
    
    def gen_plots(self, path):
        self._plot_pos_std(path)
        self._plot_rot_std(path)

    def _plot_pos_std(self, path):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self._pos_std_list)
        plt.title("Position Std Dev Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Position Std Dev")
        plt.grid()

        plt.savefig(f"{path}/pos_std.png")
    
    def _plot_rot_std(self, path):

        plt.figure()
        plt.plot(self._rot_std_list)
        plt.title("Rotation Std Dev Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Rotation Std Dev")
        plt.grid()
        plt.savefig(f"{path}/rot_std.png")

class XYProposer(Proposer):
    def __init__(self, rand_seed, num_worlds, prior, iterations):
        super().__init__(rand_seed, num_worlds, prior, iterations)

        self._cur_rot_std = 0.0  # No rotation for XYProposer

    def initial_proposal(self):
        np_xform = np.array(self.prior.xform)
        proposals = np.tile(np_xform, (self.num_worlds, 1))

        proposals[1:, :2] += self.rng.normal(loc=0.0, scale=self._cur_pos_std, size=(self.num_worlds - 1, 2))
        
        self._update()

        return self._apply_bounds(proposals)    
    
    def propose(self, positions, weights):
        proposals = np.zeros((self.num_worlds, 7))

        # Keep Top Likelihood - Go into allocs to find top for that object
        proposals[0, :] = positions[np.argmax(weights)]
        
        idx = self.rng.choice(self.num_worlds, size=self.num_worlds-1, p=weights, replace=True)
        proposals[1:, :] = positions[idx]

        # Add Gaussian Noise
        proposals[1:, :2] += self.rng.normal(loc=0.0, scale=self._cur_pos_std, size=(self.num_worlds - 1, 2))

        self._update()

        return self._apply_bounds(proposals)

class XYZQProposer(Proposer):
    def initial_proposal(self):
        np_xform = np.array(self.prior.xform)
        proposals = np.tile(np_xform, (self.num_worlds, 1))

        proposals[1:, :3] += self.rng.normal(loc=0.0, scale=self._cur_pos_std, size=(self.num_worlds - 1, 3))
        proposals[1:, 3:] = self._gen_rotations(proposals[1:, 3:])
        
        self._update()

        return self._apply_bounds(proposals)

    def propose(self, positions, likelihood):
        proposals = np.zeros((self.num_worlds, 7))

        # Keep Top Likelihood - Go into allocs to find top for that object
        proposals[0, :] = positions[np.argmax(likelihood)]
        
        ranks = np.argsort(np.argsort(likelihood)) + 1  # 1-based so worst != 0
        weights = ranks / ranks.sum()

        idx = self.rng.choice(self.num_worlds, size=self.num_worlds-1, p=weights, replace=True)
        proposals[1:, :] = positions[idx]

        # Add Gaussian Noise
        proposals[1:, :3] += self.rng.normal(loc=0.0, scale=self._cur_pos_std, size=(self.num_worlds - 1, 3))
        proposals[1:, 3:] = self._gen_rotations(proposals[1:, 3:])

        self._update()

        return self._apply_bounds(proposals)

class NoDecayProposal(XYProposer):
    def _update_pos_std(self):
        pass
    
    def _update_rot_std(self):
        pass

class ExpDecayProposal(XYProposer):
    NUM_HALVING = 2

    def __init__(self, rand_seed, num_worlds, prior, iterations):
        super().__init__(rand_seed, num_worlds, prior, iterations)

        self.multiplier = 0.5 ** (self.NUM_HALVING / self.iterations)

    def _update_pos_std(self):
        self._cur_pos_std *= self.multiplier
    
    def _update_rot_std(self):
        self._cur_rot_std *= self.multiplier
