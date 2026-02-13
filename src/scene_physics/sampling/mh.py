"""
Contains all the different Metropolis-Hasting Sampling Codes
"""

import numpy as np



class XZ_MH_Sampler:
    """
    Used to sample (x, z) positions of likelihood for a given object

    self.body [Type Body] - Refers to the body we are sampling over
    self.model [Warp Model] - Refers to the Netwon model
    self.likelihood [Likelihood Class] - Refers to a likelihood class
    self.proposal_std [int] - Standard Deviation of Proposals
    self.render_fun [render_fun] - Creates the point cloud rendered on

    """
    def __init__(self, body, model, likelihood, proposal_std, render_fn):
        self.body = body
        self.model = model
        self.likelihood = likelihood
        self.proposal_std = proposal_std
        self.render_fn = render_fn

    def initial_sample(self):
        """Sample initial position from standard normal prior."""
        x = float(np.random.normal(scale=.2))
        z = float(np.random.normal(scale=.2))
        return x, z

    def propose(self, x_curr, z_curr):
        """Propose new position via Gaussian random walk."""
        x_new = float(np.random.normal(loc=x_curr, scale=self.proposal_std))
        z_new = float(np.random.normal(loc=z_curr, scale=self.proposal_std))
        return x_new, z_new

    def run_sampling(self, iterations, debug=False):
        """ Run Metropolis-Hastings sampling"""
        state0, state1 = self.model.state(), self.model.state()

        # Initilize Prior on state
        x_curr, z_curr = self.initial_sample()
        state0.body_q = self.body.move_position_wp(state0, x_curr, z_curr)
        state1.body_q = self.body.move_position_wp(state1, x_curr, z_curr)

        # Generate likelihood
        prev_likelihood = self.likelihood.new_proposal_likelihood(
            self.render_fn(state0)
        )

        # Track histories
        positions, scores = [], []

        for i in range(iterations):
            x_prop, z_prop = self.propose(x_curr, z_curr)
            state1.body_q = self.body.move_position_wp(state0, x_prop, z_prop)
            new_likelihood = self.likelihood.new_proposal_likelihood(
                self.render_fn(state1))

            log_alpha = new_likelihood - prev_likelihood
            if np.log(np.random.uniform()) < log_alpha:
                x_curr, z_curr = x_prop, z_prop
                state0.assign(state1)
                prev_likelihood = new_likelihood


            positions.append((x_curr, z_curr))
            scores.append(float(prev_likelihood))

            if debug and i % 10 == 0:
                print(f"Iteration {i}: likelihood={prev_likelihood:.2f}, pos=({x_curr:.3f}, {z_curr:.3f})")
                save_point_cloud_ply(self.render_fn(state0), f"recordings/{i}_cloud.ply")
                print(log_alpha)
                

        return positions, scores
