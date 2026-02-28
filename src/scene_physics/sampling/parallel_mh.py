"""
Parallel Metropolis-Hastings sampler for multi-body 6DOF scene reconstruction.

Evaluates num_worlds proposals in parallel using Newton's multi-world GPU capability.
Objects are placed sequentially: converge object 1, freeze, then sample object 2, etc.
"""

import numpy as np
import warp as wp

from scene_physics.sampling.proposals import SixDOFProposal


class ParallelPhysicsMHSampler:
    """
    GPU-parallel MH sampler with sequential object placement.

    For each body in placement_order:
        1. Generate num_worlds 6DOF proposals
        2. Write each proposal into its world's body_q
        3. Run forward physics (single solver loop, all worlds in parallel)
        4. Batch render all worlds
        5. Batch compute likelihoods
        6. Select best proposal (greedy) or MH accept/reject
        7. Freeze accepted position, move to next body

    Args:
        model: Newton model (finalized, with all parallel worlds)
        likelihood: Likelihood_Physics_Parallel instance
        placement_order: list of body names in order of placement
        world_bodies: list[dict] from build_parallel_worlds — per-world body mappings
        body_index_map: dict (world_idx, body_name) -> body_q index
        num_worlds: number of parallel worlds
        proposal: SixDOFProposal instance
        convergence_threshold: normalized score threshold to stop early (0-1 range)
    """

    def __init__(
        self,
        model,
        likelihood,
        objects,
        proposal=None,
        iter_per_obj=None,
        convergence_threshold=0.8,
    ):
        self.sample_state = model.state()
        self.likelihood = likelihood
        self.objects = objects
        self.proposal = SixDOFProposal if proposal is None else proposal

        self.iter_per_obj = 40 if iter_per_obj is None else iter_per_obj

    def run_single_body_sampling(self, obj, total_iter, physics=False, init_positions=None, debug=False):
        """
        Run sequential placement with parallel across 1 objectn

        Args:
            iterations_per_object: number of MH iterations per body

        Returns:
            Object correctly placed
        """

        # Generate the proposal method, and get initial positions
        proposor = self.proposal(obj)

        # Move positions in the scene and get scores
        if init_positions is None:
            prev_positions = proposor.initial_positions()
        else:
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

        # Save best position (TODO Build it for n bodies)
        obj.place_final_position(prev_positions, prev_scores, self.sample_state) 


    def run_sampling(self, debug=False):
        """
        Run importance sampling with parrallel proposal evalutation
        for a variety of objects
        """

        for obj in self.objects["observed"]:
            self.run_single_body_sampling(obj, self.iter_per_obj, physics=False)

        for obj in self.objects["unobserved"]:
            self.run_single_body_sampling(obj, self.iter_per_obj, physics=True)

