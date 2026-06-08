import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from scene_physics.sampling.proposals import Proposer, Prior

class Body:
    def __init__(self, name, num_worlds):
        self.name = name
        self.allocs = []
        self.num_worlds = num_worlds

        self.correct = None

    def add(self, i):
        self.allocs.append(i)

    def finalize(self, model):
        self.allocs = np.array(self.allocs)

    def __str__(self):
        return f"Body Name: {self.name}"
    

class Static(Body):
    pass

class Dynamic(Body):
    def __init__(self, name, num_worlds):
        super().__init__(name, num_worlds)

        self.proposer = None
        self.prior = None
        self.plots = []

        # Best score: (-inf, None) → (likelihood, value)
        self.best = (-np.inf, None)


    def set_proposer(self, rand_seed, prior_dict, proposer, iterations):
        self.prior = Prior(prior_dict)
        self.proposer = proposer(rand_seed, self.num_worlds, self.prior, iterations)

    def initialize(self, scene):
        proposals = self._warp_to_numpy(scene)
        proposals[self.allocs] = self.proposer.initial_proposal()
        scene.body_q = self._numpy_to_warp(proposals)

        return scene
    
    def propose(self, scene, likelihood):
        proposals = self._warp_to_numpy(scene)

        # Index Position, Likelihood, Then propose
        positions, likelihoods = proposals[self.allocs], likelihood

        # Save the highest likelihood
        self._update_best(likelihoods, positions)

        proposals[self.allocs] = self.proposer.propose(positions, likelihoods)

        # Save a copy of highest likelihood location (allocs[0] = best world's body)
        self.plots.append(proposals[self.allocs].copy())

        scene.body_q = self._numpy_to_warp(proposals)
        return scene
    
    def get_positions(self, state):
        return self._warp_to_numpy(state)[self.allocs]
    
    def _update_best(self, likelihood, positions):
        max_likelihood = likelihood.max()
        max_index = likelihood.argmax()

        if max_likelihood > self.best[0]:
            self.best = (max_likelihood, positions[max_index])

    
    def get_final_location(self):
        return self.best[1]
    
    def _warp_to_numpy(self, state):
        return state.body_q.numpy()
    
    def _numpy_to_warp(self, proposals):
        return wp.array(proposals, dtype=wp.transformf)
    
    def gen_plots(self, save_dir):
        self._plot_xy_position(save_dir)
        self.proposer.gen_plots(save_dir)
        self._save_positions(save_dir)
        self._save_best(save_dir)

    def _save_positions(self, save_dir):
        positions = np.array(self.plots)
        np.save(f"{save_dir}/{self.name}_positions.npy", positions)

    def _save_best(self, save_dir):
        with open(f"{save_dir}/{self.name}_best.txt", 'w') as f:
            f.write(f"Best Likelihood: {self.best[0]}\n")
            f.write(f"Best Position: {self.best[1]}\n")

    def _plot_xy_position(self, save_dir):
        assert self.correct is not None, "Correct value must be assigned to generate plots"

        positions = np.array(self.plots)
        n, nw, size = positions.shape

        flat = positions.reshape(-1, size)
        iters = np.repeat(np.arange(n), nw)

        fig, ax = plt.subplots()
        sc = ax.scatter(flat[:, 0], flat[:, 1], c=iters, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="Iteration")

        # Plot the correct position
        ax.scatter(self.correct[0], self.correct[1], marker=(7, 1, 0), s=50, color='red')


        ax.set_title(f"XY Position of {self.name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.savefig(f"{save_dir}/{self.name}_position.png")
        plt.close(fig)


class Observed(Dynamic):
    pass

class Hidden(Dynamic):
    pass
