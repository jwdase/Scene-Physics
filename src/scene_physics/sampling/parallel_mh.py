"""
Parallel Metropolis-Hastings sampler for multi-body 6DOF scene reconstruction.

Evaluates num_worlds proposals in parallel using Newton's multi-world GPU capability.
Objects are placed sequentially: converge object 1, freeze, then sample object 2, etc.
"""

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
    ):
        self.sample_state = model.state()
        self.likelihood = likelihood
        self.objects = objects
        self.proposal = SixDOFProposal if proposal is None else proposal

        self.iter_per_obj = 200 if iter_per_obj is None else iter_per_obj

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

        # Save best position (TODO Build it for n bodies)
        obj.place_final_position(prev_positions, prev_scores, self.sample_state) 


    def run_sampling(self, debug=False):
        """
        Run importance sampling with parrallel proposal evalutation
        for a variety of objects
        """
        
        # Insert observed objects
        print("============")
        print("Non Physics Sampling")
        for obj in self.objects["observed"]:
            print(f"Working on obj: {obj.name}")
            self.run_single_body_sampling(obj, self.iter_per_obj, physics=False)
        
        # Insert unobserved objects
        print("============")
        print("Physics Sampling")
        for obj in self.objects["unobserved"]:
            print(f"Working on obj: {obj.name}")
            self.run_single_body_sampling(obj, self.iter_per_obj, physics=True)


    def print_results(self):
        """Prints final position of each object"""

        for obj_type, obj_list in self.objects.items():
            for obj in obj_list:
                print(f"Object: {obj.name} was placed at {obj.final_position}")

