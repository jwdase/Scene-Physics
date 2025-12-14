# Add correct path for me
from typing import Any
from utils.io import setup_path
setup_path('jonathan')

# Package Import
import pyvista
import warp as wp
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# GenJax
from genjax import ChoiceMapBuilder as C
from genjax import gen, normal, pretty, uniform
from genjax import Diff, NoChange, UnknownChange
import genjax

# Newton Library
import newton
from newton._src.utils.recorder import RecorderModelAndState
from newton.solvers import SolverXPBD

# Bayes 3D Import
from b3d.chisight.dense.likelihoods.image_likelihoods import threedp3_likelihood_per_pixel_old
from b3d.camera import unproject_depth, Intrinsics
import b3d

# Files
from properties.shapes import Sphere, Box, MeshBody, SoftMesh, StableMesh
from properties.material import Material
from visualization.scene import PyVistaVisuailzer
from utils.io import plot_point_maps

# Setup Defaults
vec6f = wp.types.vector(length=6, dtype=float)
builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)

# Add Plane
builder.add_ground_plane()

# Material
Ball_material = Material(mu=0.8, restitution=.3, contact_ke=2e5, contact_kd=5e3, density=1e3)
Ramp_material = Material(density=0.0)

# Objects
paths = [f'objects/stable_scene/{val}' for val in ['table.obj', 'rectangle.obj']]

# Placing objects
table = MeshBody(
    builder=builder,
    body=paths[0],
    solid=True,
    scale=1.0,
    position=wp.vec3(0., 0., 0.),
    mass=0.0,
    material=Ramp_material,
)

rectangle = MeshBody(
    builder=builder,
    body=paths[1],
    solid=True,
    scale=1.0,
    position=wp.vec3(0., 0., 0.),
    mass=0.0,
    material=Ramp_material,
)

# List out bodies, camera for visualizer
bodies = [table, rectangle]
camera = [
    (1, 1.5, 3),
    (0, 1, 0),
    (0, 1, 0),
]

# Create visualizer and set intrinsics
visualizer = PyVistaVisuailzer(bodies, camera_position=camera)
visualizer.set_intrinsics(Intrinsics)

# Generate png - for correct placing and point cloud
visualizer.gen_png('recordings/mc/initial.png')
point_cloud = visualizer.point_cloud(unproject_depth)
visualizer.plot_point_maps(point_cloud, 'recordings/mc/initial_point_cloud.png')

class Likelihood:
    def __init__(self, initial_point_cloud):
        self.correct_pointcloud = initial_point_cloud
        self.control = self.get_control()

    def get_control(self):
        """ Gets the control value"""
        likelihood_control = threedp3_likelihood_per_pixel_old(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=self.correct_pointcloud,
            variance=0.001,
            outlier_prob=0.001,
            outlier_volume=1.0,
            filter_size=3,
        )

        return self.get_likelihood(likelihood_control)

    def get_likelihood(self, result):
        """ Code to cleanup likelihood from B3D result"""
        likelihood = result["pix_score"]
        clean = np.array([arr[~np.isnan(arr)] for arr in likelihood])
        score = clean.sum()
        return score

    def new_proposal_likelihood(self, proposal):
        """ Returns likelihood of new control"""
        new_score = threedp3_likelihood_per_pixel_old(
            observed_xyz=self.correct_pointcloud,
            rendered_xyz=proposal,
            variance=0.001,
            outlier_prob=0.001,
            outlier_volume=1.0,
            filter_size=3,
        )

        # [TODO] DOUBLE CHECK IF CORRECT
        return (self.get_likelihood(new_score)) / self.control

likelihood_func = Likelihood(point_cloud)

# --- 1. THE BRIDGE (Fixing the Float Issue) ---
def physics_step_python(x_val, z_val, x=[]):
    """ Runs Physics Step in Python"""

    # 1. Cast JAX arrays to standard python floats
    x_float = float(x_val)
    z_float = float(z_val)

    # 2. Run your imperative/object-oriented code
    rectangle.update_position(x_float, z_float)
    point_cloud = visualizer.point_cloud(unproject_depth)

    # Draw proposal
    visualizer.gen_png(f'recordings/mc/proposal_{len(x)}.png')
    visualizer.plot_point_maps(point_cloud, f'recordings/mc/proposal_{len(x)}_point_cloud.png')

    # So we can count # of calls to the physics step
    x.append('a')
    
    # 3. Calculate likelihood (score)
    # Ensure this returns a single float
    result = likelihood_func.new_proposal_likelihood(point_cloud)
    
    return float(result)

def physics_bridge(x, z):
    """ Moves the JAX tracers to the CPU """
    result = jax.pure_callback(
        physics_step_python,  # The python function to call
        jax.ShapeDtypeStruct((), jnp.float32),  # Scalar with float32 dtype
        x, z                  # Arguments to pass
    )
    return result

# --- 2. THE GENJAX MODEL ---
@gen
def model():
    """Defines our model with B3D likelihood"""
    # Sample latent variables
    x = uniform(-3.0, 3.0) @ 'x'
    z = uniform(-3.0, 3.0) @ 'z'

    # Call the bridge (this handles the "Tracer" vs "Float" issue)
    # We pass the JAX tracer variables directly here.
    likelihood_score = physics_bridge(x, z)

    # Update log_likelihood
    # We treat the returned score as the log probability of an observation.
    # We observe '0.0' with a 'likelihood_score' offset to inject the density.
    # (Note: This is a hack common in simulator-inference).
    log_likely = normal(likelihood_score, 1.0) @ 'll_score'

    return x, z

# --- 3. MCMC KERNEL (Manual MH Implementation) ---
# GenJAX doesn't have a built-in metropolis_hastings function.
# We implement it manually following the pattern from b3d/demos.
PROPOSAL_STD = 0.1  # Standard deviation for the random walk proposal

def mh_step(key, trace):
    """
    Perform one Metropolis-Hastings step with a symmetric Gaussian random walk proposal.
    """
    key, key_prop_x, key_prop_z, key_accept = jax.random.split(key, 4)
    
    # Get current values from trace
    current_x = trace.get_choices()['x']
    current_z = trace.get_choices()['z']
    
    # Propose new values (symmetric Gaussian random walk)
    proposed_x = genjax.normal.sample(key_prop_x, current_x, PROPOSAL_STD)
    proposed_z = genjax.normal.sample(key_prop_z, current_z, PROPOSAL_STD)
    
    # For symmetric proposals: q(x'|x) = q(x|x'), so q_fwd = q_bwd and they cancel
    # Thus we only need the model probability ratio (p_ratio from update)
    
    # Update trace with proposed values
    proposed_trace, p_ratio, _, _ = trace.update(
        key,
        C["x"].set(proposed_x).merge(C["z"].set(proposed_z)),
        genjax.Diff.tree_diff_no_change(trace.get_args()),
    )
    
    # MH acceptance ratio (symmetric proposal, so just p_ratio)
    log_alpha = jnp.minimum(p_ratio, 0.0)  # min(p_ratio, 1) in log space
    alpha = jnp.exp(log_alpha)
    
    # Accept or reject
    accept = jax.random.bernoulli(key_accept, alpha)
    new_trace = jax.lax.cond(accept, lambda _: proposed_trace, lambda _: trace, None)
    
    return new_trace, accept

def metropolis_hastings_step(carry, key):
    trace = carry
    new_trace, _ = mh_step(key, trace)
    return new_trace, new_trace

# --- 4. INFERENCE LOOP ---
def run_inference(num_samples, key):
    key, subkey_init, subkey_scan = jax.random.split(key, 3)

    # 1. Constrain the likelihood score to be observed
    # We observe 0.0 for the dummy likelihood variable
    observations = C["ll_score"].set(0.0)

    # 2. Generate initial trace using importance sampling
    # The model takes no arguments, so we pass ()
    initial_trace, _ = model.importance(subkey_init, observations, ())

    # 3. Run the MCMC Chain
    keys = jax.random.split(subkey_scan, num_samples)
    
    # Scan carries just the trace through the loop
    final_trace, chain = jax.lax.scan(
        metropolis_hastings_step,
        initial_trace,
        keys
    )
    
    return chain

# --- 5. EXECUTION ---
if __name__ == "__main__":
    key = jax.random.key(42)
    
    # JIT Compile the inference loop for speed
    # Note: The 'pure_callback' will prevent full fusion, but the logic 
    # around it will still be optimized.
    jit_inference = jax.jit(run_inference, static_argnums=(0,))
    
    print("Compiling and running inference...")
    traces = jit_inference(100, key)
    
    # Extract results from the chain of traces
    # GenJAX traces use .get_choices() to access sampled values
    xs = traces.get_choices()['x']
    zs = traces.get_choices()['z']
    ll_scores = traces.get_choices()['ll_score']

    
    print(f"Sampled {len(xs)} points.")
    print(f"X values: {xs}")
    print(f"Z values: {zs}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Scatter plot of x vs z (posterior samples)
    axes[0].scatter(xs, zs, alpha=0.5, s=10)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')
    axes[0].set_title('Posterior Samples (x vs z)')
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.3)
    
    # 2. Trace plot for x
    axes[1].plot(xs)
    axes[1].set_xlabel('MCMC Iteration')
    axes[1].set_ylabel('x')
    axes[1].set_title('Trace Plot: x')
    
    # 3. Trace plot for z
    axes[2].plot(zs)
    axes[2].set_xlabel('MCMC Iteration')
    axes[2].set_ylabel('z')
    axes[2].set_title('Trace Plot: z')
    
    plt.tight_layout()
    plt.savefig('recordings/mc/results.png', dpi=150)
    plt.show()
    
    print("Plot saved to recordings/mcmc_results.png")

    # Plot trace plot for ll_scores
    axes[3].plot(ll_scores)
    axes[3].set_xlabel('MCMC Iteration')
    axes[3].set_ylabel('ll_score')
    axes[3].set_title('Trace Plot: ll_score')
    
    plt.tight_layout()
    plt.savefig('recordings/mc/results_ll_scores.png', dpi=150)
    plt.show()
    
# Flow of Information
# jax.lax.scan (lines 239-243)
#     → metropolis_hastings_step() (line 218)
#         → mh_step() (line 184)
#             → trace.update()  ← MODEL RE-EXECUTED HERE
#                 → model()
#                     → physics_bridge(x, z)
#                         → physics_step_python()
#                             → likelihood_func.new_proposal_likelihood()
#                                 → threedp3_likelihood_per_pixel_old()

