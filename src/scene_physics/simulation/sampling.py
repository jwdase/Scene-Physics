"""
End-to-end importance sampling.

Single function call to run the full pipeline: build worlds, compute
likelihoods, sample, plot, and render.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

from scene_physics.utils.setup import build_worlds
from scene_physics.utils.parallel_builder import allocate_worlds
from scene_physics.likelihood.likelihoods import ParallelPhysicsLikelihood
from scene_physics.sampling.parallel_mh import ImportanceSampling
from scene_physics.visualization.scene import PyVistaVisualizer, PhysicsVideoVisualizer


# Defaults
DEFAULT_CAMERA = [(4., 4., 4.), (0., 0., 0.), (0, 1, 0)]
DEFAULT_EYE = np.array([1., 1.5, 1.])
DEFAULT_TARGET = np.array([0., 0., 0.])
DEFAULT_DEGREES = 60

def run_importance_sampling(
    objects,
    location,
    method="gibbs",
    num_worlds=50,
    iter_per_obj=5,
    total_iterations=20,
    decay="exp",
    burn_in=30,
    width=320,
    height=240,
    max_depth=4.0,
    wp_eye=DEFAULT_EYE,
    wp_target=DEFAULT_TARGET,
    fov_degrees=DEFAULT_DEGREES,
    pyvista_camera=DEFAULT_CAMERA,
    sim_seconds=2,
    sim_fps=40,
    debug=False,
):
    """
    Runs the complete importance sampling pipeline on a SimulationObjects.

    Args:
        objects: SimulationObjects dataclass with observed, unobserved, static
        location: output directory for recordings (e.g. "recordings/my_experiment")
        num_worlds: number of parallel proposal worlds
        iter_per_obj: sampling iterations per object
        total_iterations: total Gibbs iterations after burn-in
        decay: proposal std schedule ("no_decay", "linear", "exp")
        burn_in: burn-in iterations for Gibbs sampling
        width: depth camera width
        height: depth camera height
        max_depth: max depth for point cloud
        wp_eye: camera eye position for depth rendering
        wp_target: camera look-at for depth rendering
        pyvista_camera: PyVista camera position for visualization
        sim_seconds: forward physics duration for final video
        sim_fps: physics FPS
        debug: save per-iteration proposal visualizations
    """

    os.makedirs(location, exist_ok=True)
    # Phase 1 — Build the world
    print("Building Model")
    worlds = allocate_worlds(num_worlds)
    model = build_worlds(worlds, objects)

    # Phase 2 — Build the likelihood
    print("Building Likelihood Function")
    visualizer = PyVistaVisualizer(objects, num_worlds=num_worlds, camera_pos=pyvista_camera)
    likelihood = ParallelPhysicsLikelihood(
        model=model,
        objects=objects,
        wp_eye=wp_eye,
        wp_target=wp_target,
        fov_degrees=fov_degrees,
        num_worlds=num_worlds,
        name=location,
        max_depth=max_depth,
        height=height,
        width=width,
    )

    # Phase 3 — Build and run the sampler
    print("Building Sampler")
    sampler = ImportanceSampling(
        model, likelihood, objects,
        iter_per_obj=iter_per_obj,
        visualization=visualizer,
        name=location,
        decay=decay,
    )

    if method == "gibbs":
        sampler.run_sampling_gibbs(iters=total_iterations, debug=debug, burn_in=burn_in)
    elif method == "occ":
        sampler.run_occluded_sampling(iters=total_iterations, debug=debug)
    else:
        raise ValueError(f"Sampling method {method} does not exist")

    # Results
    sampler.print_results()
    sampler.plot_proposal_scores()
    sampler.plot_proposal_stds()
    sampler.plot_avg_score()
    sampler.plot_proposal_scatter()

    print("Saving .png Final Result")
    visualizer.show_final_scene(f"{location}/final_position.png")

    print("Filming Final Result")
    video_visualizer = PhysicsVideoVisualizer(objects, FPS=sim_fps, camera_pos=pyvista_camera)
    video_visualizer.render_final_scene(f"{location}/final_state.mp4")

    print(f"Done — results in {location}/")
    return sampler
