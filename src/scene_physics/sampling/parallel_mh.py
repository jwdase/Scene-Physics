"""
Class for Importance Sampling for Objects across states.
- Evaluates likelihood across N worlds and then updates position from function

Main Uses
- Using run_sampling_linear_print runs a linear placement of each object
- Using run_sampling_gibbs runs a placement of each object with choice random
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from scipy.special import softmax

from scene_physics.sampling.proposals import SixDOFProposal


class ImportanceSampling:

    def __init__(
        self,
        model,
        likelihood,
        objects,
        decay=None,
        proposal=None,
        iter_per_obj=None,
        convergence_threshold=0.8,
        visualization=None,
        interval=None,
        name=None,
        master_seed=42,
    ):
        self.sample_state = model.state()
        self.likelihood = likelihood
        self.objects = objects
        self.object_list = self._get_objects()

        # Numpy seed for proposals
        self.np_seed = np.random.SeedSequence(master_seed)

        # Specify Decay before
        self.decay = "no_decay" if decay is None else decay

        # Information for proposals
        self.proposals = self._gen_proposals(
            SixDOFProposal if proposal is None else proposal
        )
        self.iter_per_obj = 200 if iter_per_obj is None else iter_per_obj

        # Data saving information
        self.visualization = visualization
        self.plot_interval = 5 if interval is None else interval
        self.name = name
        self.likelihoods = []

    def _get_objects(self):
        return self.objects["observed"] + self.objects["unobserved"]

    def _gen_proposals(self, proposal):
        """
        Generate a proposor for each object as a dict

        Returns:
            Dict[hast(obj) : Proposor]
        """
        proposals = {}
        children = self.np_seed.spawn(len(self.object_list))

        # Loop through get attributes and create proposal
        for i, obj in enumerate(self.object_list):
            priors, num_worlds = obj.set_proposal()
            proposals[hash(obj)] = proposal(
                priors, num_worlds, children[i], schedule=self.decay
            )

        return proposals

    def _generate_positions(self, position, scores):
        """
        takes in positions and scores. Returns an [N, 7] matrix
        ranking the top scores
        """

        # Get Worlds w/ Probability
        num_worlds = len(scores)
        probs = softmax(scores)

        # Select top world to continue on
        top_world = position[np.argmax(probs)][np.newaxis]  # (1, 7)

        # Randomly select others from probs
        rng = np.random.default_rng(self.np_seed.spawn(1)[0])
        resampled_indices = rng.choice(
            num_worlds, size=num_worlds - 1, replace=True, p=probs
        )
        new_positions = position[resampled_indices]

        # Stack over positions
        return np.concatenate([top_world, new_positions], axis=0)

    def run_single_body_sampling(
        self,
        obj,
        total_iter,
        object_num,
        physics=False,
        init_positions=None,
        debug=False,
        count=True,
    ):
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
        prev_scores = self.likelihood.new_proposal_likelihood_still_batch(
            self.sample_state
        )

        # Run Importance Sampling
        for iteration in range(total_iter):

            # Get a ranking of positions and then sample
            new_positions = self._generate_positions(prev_positions, prev_scores)
            new_positions = proposor.propose_general(new_positions, iteration, count)

            # Move the object
            obj.move_6dof_wp(new_positions, self.sample_state)

            # Save values for new iteration
            if physics:
                prev_scores = self.likelihood.new_proposal_likelihood_physics_batch(
                    self.sample_state
                )
            else:
                prev_scores = self.likelihood.new_proposal_likelihood_still_batch(
                    self.sample_state
                )

            # Update for next loop
            prev_positions = new_positions

            # Plot out representation of all proposals
            if (
                debug
                and (self.visualization is not None)
                and (iteration % self.plot_interval) == 0
            ):
                number = total_iter * object_num + iteration
                location = f"{self.name}/epoch_{number}"
                self._save_proposals(location, self.sample_state)

            # Save proposals and values
            self.likelihoods.append(prev_scores)

        self._update_all_worlds(scores=prev_scores)

    def _update_all_worlds(self, scores):
        """
        Takes scores computes softmax for relative likelihood. It then
        copies the worlds with respect to their softmax probability across the
        n allocated worlds in self.sample_scene
        """
        probs = softmax(scores)
        num_worlds = len(scores)

        # Get top score and put at world 0
        top_index = np.argmax(scores)

        # Multinomial resample: high-score worlds duplicated, low-score worlds dropped
        rng = np.random.default_rng(self.np_seed.spawn(1)[0])
        resampled_indices = rng.choice(
            num_worlds, size=num_worlds - 1, replace=True, p=probs
        )
        resampled_indices = np.concatenate([[top_index], resampled_indices])

        bodies = self.sample_state.body_q.numpy()
        new_bodies = bodies.copy()

        for obj in self.object_list:
            new_bodies[obj.allocs] = bodies[obj.allocs[resampled_indices]]

        self.sample_state.body_q = wp.array(
            new_bodies, dtype=wp.transformf, device="cuda"
        )

    def _give_final_positions(self):
        """
        Gives final positions to all the bodies so we can do simulation
        """

        # Get top world
        prev_scores = self.likelihood.new_proposal_likelihood_physics_batch(
            self.sample_state
        )
        top_world = np.argsort(prev_scores)[-1:][::-1][0]

        # Place it for all worlds
        for obj in self.object_list:
            obj.place_final_position(top_world, self.sample_state)

    def _save_proposals(self, location, state):
        """
        Clears the folder to put proposals inside of it for visualization
        """

        assert self.visualization is not None, "Must specify visualization before"

        # Make directory and then save
        os.makedirs(location, exist_ok=True)
        self.visualization.gen_multi_world_png(state, location)

    def run_single_sample(self, obj, epoch, debug, count=True):
        """
        Code to run one forward run of sampling on an objects. We will
        always use physics for this pass
        """
        # Get proposor
        proposer = self.proposals[hash(obj)]

        # Generate sample
        current_positions = obj.get_positions(self.sample_state)  # (N, 7)
        new_positions = proposer.propose_general(
            current_positions, epoch, count
        )  # (N, 7)
        obj.move_6dof_wp(new_positions, self.sample_state)

        # Create visualization
        if (
            debug
            and (self.visualization is not None)
            and (epoch % self.plot_interval) == 0
        ):
            location = f"{self.name}/epoch_{epoch}"
            self._save_proposals(location, self.sample_state)

        # Calculate likelihood function
        scores = self.likelihood.new_proposal_likelihood_physics_batch(
            self.sample_state
        )
        self.likelihoods.append(scores)
        self._update_all_worlds(scores)

    def run_sampling_gibbs(self, iters=100, debug=False, burn_in=30, seed=42):
        """
        Run gibbs sampling on scene so that we do each object proposals
        and then move to next object
        """

        # Randomly sample and place these objects
        objects = self.object_list
        rng = np.random.default_rng(self.np_seed.spawn(1)[0])

        # Place Objects for Burn In
        object_num = 0
        print(f"Beginning the Burn in on {len(objects)} objects")
        for obj in objects:
            print(f"Working on obj: {obj}")
            self.run_single_body_sampling(
                obj, burn_in, object_num, physics=False, count=False
            )
            object_num += 1

        # Running the Gibbs Sampling
        print(f"Burn in Complete")
        print(f"Beginning the Physics Sampling")
        for i in range(iters):
            choice = rng.integers(low=0, high=len(objects))
            obj = objects[choice]
            self.run_single_sample(obj, epoch=i, debug=debug, count=True)

            if i % 10 == 0:
                print(f"Epoch: {i}, object: {obj} ")

        # Give final positions to objects
        self._give_final_positions()

    def run_sampling_linear_print(self, debug=False):
        """
        Run importance sampling with parrallel proposal evalutation
        for a variety of objects
        """

        assert len(self.likelihoods) == 0, "Please clear likelihoods before a new run"

        object_num = 0

        # Insert observed objects
        print("============")
        print("Non Physics Sampling")
        for obj in self.objects["observed"]:
            print(f"Working on obj: {obj.name}")
            self.run_single_body_sampling(
                obj, self.iter_per_obj, object_num, debug=debug, physics=False
            )
            object_num += 1

        # Insert unobserved objects
        print("============")
        print("Physics Sampling")
        for obj in self.objects["unobserved"]:
            print(f"Working on obj: {obj.name}")
            self.run_single_body_sampling(
                obj, self.iter_per_obj, object_num, debug=debug, physics=True
            )
            object_num += 1

        # Give Final Positions
        self._give_final_positions()

    def print_results(self):
        """Prints final position of each object"""
        for obj_type, obj_list in self.objects.items():
            for obj in obj_list:
                print(f"Object: {obj.name} was placed at {obj.final_position}")

    # ========== CLAUDE SECTION ==============

    def save_likelihoods_txt(self, path):
        """
        Write self.likelihoods to a plain-text file.

        Each row corresponds to one sampling iteration; each column is one world.
        Shape written: (num_iterations, num_worlds).

        Args:
            path: file path to write (e.g. "run/likelihoods.txt")
        """
        assert (
            len(self.likelihoods) > 0
        ), "No likelihoods recorded — run sampling with debug=True first"
        scores = np.stack(self.likelihoods, axis=0)  # (num_iterations, num_worlds)
        np.savetxt(
            path,
            scores,
            fmt="%.6f",
            header=f"shape={scores.shape} rows=iterations cols=worlds",
        )

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
        fig.savefig(f"{self.name}/scores.png", dpi=150)

        plt.close(fig)

    def plot_avg_score(self):
        """
        Plot mean likelihood score across all worlds at each iteration.

        self.likelihoods is a list of (num_worlds,) arrays covering both the
        burn-in phase (run_single_body_sampling) and the Gibbs phase
        (run_single_sample). The mean and max across worlds are plotted so
        convergence is easy to read.

        Saves to <self.name>/avg_score.png.
        """
        assert len(self.likelihoods) > 0, "No likelihoods recorded — run sampling first"

        scores = np.stack(self.likelihoods, axis=0)  # (num_iterations, num_worlds)
        iterations = np.arange(len(self.likelihoods))
        mean_scores = scores.mean(axis=1)
        max_scores = scores.max(axis=1)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(iterations, mean_scores, label="mean", linewidth=2)
        ax.plot(iterations, max_scores, label="max", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Likelihood score")
        ax.set_title("Average likelihood score across worlds over time")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        os.makedirs(os.path.dirname(self.name) or ".", exist_ok=True)
        fig.savefig(f"{self.name}/avg_score.png", dpi=150)
        plt.close(fig)

    def plot_proposal_stds(self):
        """
        Plot pos_std and rot_std for each object's proposer over sampling iterations.

        Each object contributes two lines (one per subplot): the schedule-scaled
        position and rotation standard deviations recorded in SixDOFProposal at
        every call to propose_general.

        Saves to <self.name>/proposal_stds.png.
        """
        fig, (ax_pos, ax_rot) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

        for obj in self.object_list:
            proposer = self.proposals[hash(obj)]
            epochs = proposer.epoch_num
            if len(epochs) == 0:
                continue
            ax_pos.plot(epochs, proposer.save_pos_std, label=obj.name)
            ax_rot.plot(epochs, proposer.save_rot_std, label=obj.name)

        ax_pos.set_ylabel("pos_std")
        ax_pos.set_title("Position proposal std across iterations")
        ax_pos.legend(loc="upper right")
        ax_pos.grid(True, alpha=0.3)

        ax_rot.set_ylabel("rot_std (rad)")
        ax_rot.set_xlabel("Iteration")
        ax_rot.set_title("Rotation proposal std across iterations")
        ax_rot.legend(loc="upper right")
        ax_rot.grid(True, alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(self.name) or ".", exist_ok=True)
        fig.savefig(f"{self.name}/proposal_stds.png", dpi=150)
        plt.close(fig)
