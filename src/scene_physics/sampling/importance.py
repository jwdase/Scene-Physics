import matplotlib.pyplot as plt
import numpy as np

from scene_physics.properties.shapes import Object_Collection
from scene_physics.likelihood.likelihoods import ParallelPhysicsLikelihood


class ImportanceSampler:
    def __init__(self, objects : Object_Collection, likelihood_fn : ParallelPhysicsLikelihood, state, save_dir : str):
        self.objects = objects
        self.likelihood_fn = likelihood_fn
        self.state = state

        # Values Saved throughout sampling
        self.likelihood = []

        # Generate sampling.txt
        self.sampling_path = f"{save_dir}/sampling.txt"

        with open(self.sampling_path, 'w') as f:
            f.write("Iteration,Object,Likelihood\n")

    def initialize(self):
        """ Initialize world from priors"""
        self.objects.initialize(self.state)

    def sample(self, obj, likelihood):
        """Sample a new object pose based on the state and likelihood function"""

        # Propose and update location in state
        obj.propose(self.state, likelihood)

        # Get likelihood function
        likelihood = self.likelihood_fn.physics(self.state)

        return likelihood
    
    def gibb_sample(self, iterations):
        """Gibbs Sample over all objects in the scene"""

        likelihood = self.likelihood_fn.physics(self.state)

        for i in range(iterations):
            obj = self.objects.get_random()
            likelihood = self.sample(obj, likelihood)

            self.likelihood.append(likelihood)
            print(f"Iteration {i}, Likelihood: {likelihood.max()}")

            with open(self.sampling_path, 'a') as f:
                f.write(f"{i},{obj.name},{likelihood.max()}\n")


    def gen_plots(self, save_dir):
        self._plot_top_likelihood(save_dir)
        self._plot_avg_likelihood(save_dir)


    def _plot_top_likelihood(self, save_dir):
        max_likelihood = [max(sim_round) for sim_round in self.likelihood]
        plt.plot(list(range(len(max_likelihood))), max_likelihood)
        
        plt.title("Max Likelihood Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Max Likelihood")
        plt.savefig(f"{save_dir}/max_likelihood.png")
        plt.clf()

    def _plot_avg_likelihood(self, save_dir):
        avg_likelihood = [np.mean(sim_round) for sim_round in self.likelihood]
        plt.plot(list(range(len(avg_likelihood))), avg_likelihood)
        
        plt.title("Avg Likelihood Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Avg Likelihood")
        plt.savefig(f"{save_dir}/avg_likelihood.png")
        plt.clf()