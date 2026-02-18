"""
Parallel Metropolis-Hastings sampler for multi-body 6DOF scene reconstruction.

Evaluates num_worlds proposals in parallel using Newton's multi-world GPU capability.
Objects are placed sequentially: converge object 1, freeze, then sample object 2, etc.
"""

import numpy as np
import warp as wp


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
        placement_order,
        world_bodies,
        body_index_map,
        num_worlds,
        proposal,
        convergence_threshold=0.8,
    ):
        self.model = model
        self.likelihood = likelihood
        self.placement_order = placement_order
        self.world_bodies = world_bodies
        self.body_index_map = body_index_map
        self.num_worlds = num_worlds
        self.proposal = proposal
        self.convergence_threshold = convergence_threshold

    def run_sampling(self, iterations_per_object, init_positions=None, debug=False):
        """
        Run sequential placement with parallel proposal evaluation.

        Args:
            iterations_per_object: number of MH iterations per body
            init_positions: optional dict body_name -> (position, quat) for initialization.
                If not provided, samples from a Gaussian prior.
            debug: print progress info

        Returns:
            results: dict mapping body_name -> {
                'position': final (3,) array,
                'quat': final (4,) array,
                'scores': list of scores per iteration,
                'positions': list of (pos, quat) per iteration,
            }
        """
        # Create a shared initial state
        state = self.model.state()
        results = {}

        for body_name in self.placement_order:
            if debug:
                print(f"\n=== Sampling body: {body_name} ===")

            # Initialize current position
            if init_positions and body_name in init_positions:
                current_pos, current_quat = init_positions[body_name]
                current_pos = np.asarray(current_pos, dtype=np.float32)
                current_quat = np.asarray(current_quat, dtype=np.float32)
            else:
                current_pos = np.random.normal(0, 0.2, size=3).astype(np.float32)
                current_pos[1] = abs(current_pos[1])  # Keep y positive (above ground)
                current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            # Set initial position in all worlds
            self._set_body_all_worlds(state, body_name, current_pos, current_quat)

            # Compute initial likelihood (use world 0 as reference)
            prev_best_score = None
            score_history = []
            position_history = []

            for iteration in range(iterations_per_object):
                # Generate batch of proposals
                positions, quats = self.proposal.propose_batch(
                    current_pos, current_quat, self.num_worlds,
                    iteration=iteration, total_iterations=iterations_per_object,
                )

                # Write proposals into each world's body_q
                body_q_np = state.body_q.numpy()
                for world_idx in range(self.num_worlds):
                    bq_idx = self.body_index_map[(world_idx, body_name)]
                    pos = positions[world_idx]
                    quat = quats[world_idx]
                    body_q_np[bq_idx] = [
                        pos[0], pos[1], pos[2],
                        quat[0], quat[1], quat[2], quat[3],
                    ]
                # Single CPU->GPU transfer
                state.body_q = wp.array(body_q_np, dtype=wp.transformf)

                # Evaluate all proposals in parallel
                scores = self.likelihood.new_proposal_likelihood_batch(state)

                # Select best proposal
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])

                # Accept if better (greedy selection from parallel proposals)
                if prev_best_score is None or best_score > prev_best_score:
                    current_pos = positions[best_idx].copy()
                    current_quat = quats[best_idx].copy()
                    prev_best_score = best_score

                # Update all worlds to current best for next iteration
                self._set_body_all_worlds(state, body_name, current_pos, current_quat)

                score_history.append(prev_best_score)
                position_history.append((current_pos.copy(), current_quat.copy()))

                if debug and iteration % 10 == 0:
                    print(
                        f"  Iter {iteration}: best_score={prev_best_score:.4f}, "
                        f"pos=({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})"
                    )

            # Freeze this body at its accepted position in all worlds
            self._set_body_all_worlds(state, body_name, current_pos, current_quat)

            results[body_name] = {
                'position': current_pos,
                'quat': current_quat,
                'scores': score_history,
                'positions': position_history,
            }

            if debug:
                print(f"  Final: pos=({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}), "
                      f"score={prev_best_score:.4f}")

        return results

    def _set_body_all_worlds(self, state, body_name, position, quat):
        """Set the same position/quat for a body across all worlds."""
        body_q_np = state.body_q.numpy()
        for world_idx in range(self.num_worlds):
            bq_idx = self.body_index_map[(world_idx, body_name)]
            body_q_np[bq_idx] = [
                position[0], position[1], position[2],
                quat[0], quat[1], quat[2], quat[3],
            ]
        state.body_q = wp.array(body_q_np, dtype=wp.transformf)
