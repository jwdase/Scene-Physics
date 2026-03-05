"""
Parallel Metropolis-Hastings sampler for multi-body 6DOF scene reconstruction.

Evaluates num_worlds proposals in parallel using Newton's multi-world GPU capability.
Objects are placed sequentially: converge object 1, freeze, then sample object 2, etc.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from scene_physics.sampling.proposals import SixDOFProposal


class ParallelPhysicsMHSampler:
    """"""

    def __init__(
        self,
        model,
        likelihood,
        objects,
        proposal=None,
        iter_per_obj=None,
        convergence_threshold=0.8,
        visualization=None,
        interval=None,
        name=None,
    ):
        self.sample_state = model.state()
        self.likelihood = likelihood
        self.objects = objects

        # Information for proposals
        self.proposals = self._gen_proposals(SixDOFProposal if proposal is None else proposal)
        self.iter_per_obj = 200 if iter_per_obj is None else iter_per_obj

        # Data saving information
        self.visualization = visualization
        self.plot_interval = 5 if interval is None else interval
        self.name = name
        self.likelihoods = []

    def _gen_proposals(self, proposal):
        """
        Generate a proposor for each object as a dict

        Returns:
            Dict[hast(obj) : Proposor]
        """
        proposals = {}
        
        # Loop through get attributes and create proposal
        for obj in self.objects["observed"] + self.objects["unobserved"]:
            attributes = obj.set_proposal()
            proposals[hash(obj)] = proposal(attributes)

        return proposals


    def run_single_body_sampling(self, obj, total_iter, object_num, physics=False, init_positions=None, debug=False):
        """
        Run sequential placement with parallel across 1 objectn

        Args:
            iterations_per_object: number of MH iterations per body

        Returns:
            Object correctly placed
        """

        # Generate the proposal method, and get initial positions
        proposor = self.proposals[hash(obj)]

        # Move positions in the scene and get scores
        if init_positions is None:
            prev_positions = proposor.initial_positions()
        else:
            # TODO Implement This
            raise NotImplementedError("Unsure how to handle init_positions") 
        
        # Move position in the scene and get score
        obj.move_6dof_wp(prev_positions, self.sample_state)
        prev_scores = self.likelihood.new_proposal_likelihood_still_batch(self.sample_state)

        # Run Importance Sampling
        for iteration in range(total_iter):
            new_positions = proposor.propose_batch(prev_positions, prev_scores, iteration, total_iter)
            obj.move_6dof_wp(new_positions, self.sample_state)
            
            # Save values for new iteration
            if physics:
                prev_scores = self.likelihood.new_proposal_likelihood_physics_batch(self.sample_state)
            else:
                prev_scores = self.likelihood.new_proposal_likelihood_still_batch(self.sample_state)
            
            # Update for next loop
            prev_positions = new_positions

            # Plot out representation of all proposals
            if debug and (self.visualization is not None) and (iteration % self.plot_interval) == 0:
                number = total_iter * object_num + iteration
                location = f"{self.name}/epoch_{number}"
                self._save_proposals(location, self.sample_state)
            
            # Save proposals and values
            if debug: self.likelihoods.append(prev_scores)

                
        # Save best position (TODO Build it for n bodies)
        obj.place_final_position(prev_positions, prev_scores, self.sample_state)

    def _save_proposals(self, location, state):
        """
        Clears the folder to put proposals inside of it for visualization
        """

        assert self.visualization is not None, "Must specify visualization before"

        # Make directory and then save 
        os.makedirs(location, exist_ok=True)
        self.visualization.gen_multi_world_png(state, location)

    def run_sampling_gibbs(self, debug=False):
        """
        Run gibbs sampling on scene so that we do each object proposals 
        and then move to next object
        """
        # TODO implement
        pass



    def run_sampling_linear_print(self, debug=False):
        """
        Run importance sampling with parrallel proposal evalutation
        for a variety of objects
        """
        object_num = 0
        
        # Insert observed objects
        print("============")
        print("Non Physics Sampling")
        for obj in self.objects["observed"]:
            print(f"Working on obj: {obj.name}")
            self.run_single_body_sampling(obj, self.iter_per_obj, object_num, debug=debug, physics=False)
            object_num += 1
        
        # Insert unobserved objects
        print("============")
        print("Physics Sampling")
        for obj in self.objects["unobserved"]:
            print(f"Working on obj: {obj.name}")
            self.run_single_body_sampling(obj, self.iter_per_obj, object_num, physics=True)
            object_num += 1


    def print_results(self):
        """Prints final position of each object"""
        for obj_type, obj_list in self.objects.items():
            for obj in obj_list:
                print(f"Object: {obj.name} was placed at {obj.final_position}")


    # ========== CLAUDE SECTION ============== 

    def plot_proposal_scores(self):
        """
        Plot likelihood scores across worlds over sampling iterations.

        self.likelihoods is a list of (num_worlds,) arrays, one per iteration.
        Each world's trace is drawn in light grey; the per-iteration max and
        mean are overlaid as bold lines so convergence is easy to read.

        Args:
            save_path : file path to save the figure (e.g. "run/scores.png").
                        If None, the plot is displayed interactively.
        """
        assert len(self.likelihoods) > 0, "Need to record likelihoods in simulation"

        # Shape: (num_iterations, num_worlds)
        scores = np.stack(self.likelihoods, axis=0)
        iterations = np.arange(len(self.likelihoods))

        # scores shape: (num_iterations, num_worlds)
        # heatmap expects (num_worlds, num_iterations) — worlds on Y, iterations on X
        fig, ax = plt.subplots(figsize=(12, max(4, scores.shape[1] // 4)))

        im = ax.imshow(
            scores.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )

        fig.colorbar(im, ax=ax, label="Likelihood score (relative to baseline)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("World")
        ax.set_title("Proposal scores per world across sampling iterations")
        fig.tight_layout()

        os.makedirs(os.path.dirname(self.name) or ".", exist_ok=True)
        fig.savefig(f"{self.name}_scores.png", dpi=150)

        plt.close(fig)
