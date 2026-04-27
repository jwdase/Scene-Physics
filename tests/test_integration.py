"""
GPU integration tests — exercises the full pipeline end-to-end.

Requires CUDA GPU. Uses PyVista-generated primitives (no mesh files needed).
Marked with @pytest.mark.gpu so they can be skipped on CPU-only machines:
    uv run pytest tests/ -m "not gpu"
"""

import numpy as np
import pyvista as pv
import pytest

# Skip entire module if no GPU
try:
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available", allow_module_level=True)
except Exception:
    pytest.skip("Warp/CUDA init failed", allow_module_level=True)

import newton
from scene_physics.properties.shapes import Parallel_Mesh, Parallel_Static_Mesh
from scene_physics.properties.material import Material
from scene_physics.properties.priors import Priors, SimulationObjects
from scene_physics.utils.parallel_builder import allocate_worlds
from scene_physics.utils.setup import build_worlds


# ── Helpers ─────────────────────────────────────────────────────────────────

NUM_WORLDS = 4

DYNAMIC_MAT = Material(mu=0.5, restitution=0.0, density=1000.0)
STATIC_MAT = Material(density=0.0)


def _make_cube(size=0.1):
    """Return a small PyVista cube mesh (no file I/O)."""
    return pv.Cube(x_length=size, y_length=size, z_length=size)


def _make_table():
    """Return a flat box as a table surface (triangulated for Newton)."""
    return pv.Box(bounds=(-0.5, 0.5, -0.02, 0.0, -0.5, 0.5)).triangulate()


def _make_objects():
    """Build a minimal SimulationObjects with 1 observed, 1 unobserved, 1 static."""
    cube_mesh = _make_cube()
    table_mesh = _make_table()

    observed = Parallel_Mesh(
        body_file=cube_mesh,
        material=DYNAMIC_MAT,
        name="obs_cube",
        position=wp.vec3(0.0, 0.2, 0.0),
        prior=Priors(x_min=-0.3, x_max=0.3, z_min=-0.3, z_max=0.3, total_iter=5),
    )
    unobserved = Parallel_Mesh(
        body_file=cube_mesh,
        material=DYNAMIC_MAT,
        name="unobs_cube",
        position=wp.vec3(0.2, 0.2, 0.0),
        prior=Priors(x_min=-0.3, x_max=0.3, z_min=-0.3, z_max=0.3, total_iter=5),
    )
    static = Parallel_Static_Mesh(
        body_file=table_mesh,
        material=STATIC_MAT,
        name="table",
    )

    return SimulationObjects(
        observed=[observed],
        unobserved=[unobserved],
        static=[static],
    )


def _build_model(objects):
    """Run the world-building pipeline and return the finalized model."""
    worlds = allocate_worlds(NUM_WORLDS)
    model = build_worlds(worlds, objects)
    return model


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def objects():
    return _make_objects()


@pytest.fixture(scope="module")
def model(objects):
    return _build_model(objects)


# ── Test: build_worlds ──────────────────────────────────────────────────────

@pytest.mark.gpu
class TestBuildWorlds:

    def test_model_finalizes(self, model):
        """build_worlds returns a finalized Newton model."""
        state = model.state()
        assert state.body_q is not None

    def test_allocs_populated(self, objects):
        """Every object has allocs after build_worlds."""
        for obj in objects.all_bodies:
            assert obj.finalized
            assert len(obj.allocs) > 0
            assert obj.num_worlds is not None

    def test_observed_has_correct_world_count(self, objects):
        """Dynamic objects should have one alloc per world."""
        for obj in objects.observed + objects.unobserved:
            assert len(obj.allocs) == NUM_WORLDS

    def test_static_has_single_alloc(self, objects):
        """Static objects are inserted once (world=-1)."""
        for obj in objects.static:
            assert len(obj.allocs) == 1

    def test_sampled_objects_are_frozen(self, objects):
        """All sampled objects should be frozen after build_worlds."""
        for obj in objects.all_sampled:
            assert obj.is_frozen


# ── Test: Parallel_Mesh lifecycle ───────────────────────────────────────────

@pytest.mark.gpu
class TestParallelMeshLifecycle:

    def test_unfreeze_restores_mass(self, objects, model):
        """Unfreezing an object restores its inv_mass to non-zero."""
        obj = objects.observed[0]
        saved_mass = obj.inv_mass.copy()

        obj.unfreeze_finalized_body()

        inv_mass = model.body_inv_mass.numpy()
        assert np.any(inv_mass[obj.allocs] != 0.0)

        # Re-freeze for other tests
        obj.freeze_finalized_body()

    def test_move_6dof_wp(self, objects, model):
        """move_6dof_wp writes positions into a scene state."""
        obj = objects.observed[0]
        obj.unfreeze_finalized_body()

        scene = model.state()
        target_pos = np.zeros((NUM_WORLDS, 7))
        target_pos[:, 0] = 0.5   # x
        target_pos[:, 1] = 0.3   # y
        target_pos[:, 6] = 1.0   # qw

        obj.move_6dof_wp(target_pos, scene)

        result = scene.body_q.numpy()[obj.allocs]
        assert np.allclose(result[:, 0], 0.5)
        assert np.allclose(result[:, 1], 0.3)

        obj.freeze_finalized_body()

    def test_get_positions(self, objects, model):
        """get_positions returns correct shape from a state."""
        obj = objects.observed[0]
        scene = model.state()
        pos = obj.get_positions(scene)
        assert pos.shape == (NUM_WORLDS, 7)

    def test_target_pose_property(self, objects):
        """target_pose returns a 7-element array."""
        obj = objects.observed[0]
        pose = obj.target_pose
        assert pose.shape == (7,)
        # qw should be 1.0 (identity quaternion)
        assert pose[6] == pytest.approx(1.0)


# ── Test: Likelihood ────────────────────────────────────────────────────────

@pytest.mark.gpu
class TestLikelihood:

    @pytest.fixture(scope="class")
    def likelihood(self, objects, model):
        from scene_physics.likelihood.likelihoods import ParallelPhysicsLikelihood

        return ParallelPhysicsLikelihood(
            model=model,
            objects=objects,
            wp_eye=np.array([1.0, 1.0, 1.0]),
            wp_target=np.array([0.0, 0.0, 0.0]),
            num_worlds=NUM_WORLDS,
            name="/tmp/test_likelihood",
            height=64,
            width=64,
            frames=10,
            eval_every=10,
        )

    def test_baseline_is_finite(self, likelihood):
        """Baseline score (target vs itself) should be finite."""
        assert np.isfinite(likelihood.baseline_score)

    def test_target_point_cloud_shape(self, likelihood):
        """Target point cloud should be (H, W, 3)."""
        assert likelihood.target_point_cloud.shape == (64, 64, 3)

    def test_still_batch_shape(self, likelihood, model):
        """still_batch returns (num_worlds,) scores."""
        scene = model.state()
        scores = likelihood.new_proposal_likelihood_still_batch(scene)
        assert scores.shape == (NUM_WORLDS,)
        assert np.all(np.isfinite(scores))

    def test_physics_batch_shape(self, likelihood, model):
        """physics_batch returns (num_worlds,) scores."""
        scene = model.state()
        scores = likelihood.new_proposal_likelihood_physics_batch(scene)
        assert scores.shape == (NUM_WORLDS,)
        assert np.all(np.isfinite(scores))


# ── Test: Sampler (one iteration smoke test) ────────────────────────────────

@pytest.mark.gpu
class TestSampler:

    @pytest.fixture(scope="class")
    def sampler(self, objects, model):
        from scene_physics.likelihood.likelihoods import ParallelPhysicsLikelihood
        from scene_physics.sampling.parallel_mh import ImportanceSampling

        likelihood = ParallelPhysicsLikelihood(
            model=model,
            objects=objects,
            wp_eye=np.array([1.0, 1.0, 1.0]),
            wp_target=np.array([0.0, 0.0, 0.0]),
            num_worlds=NUM_WORLDS,
            name="/tmp/test_sampler",
            height=64,
            width=64,
            frames=10,
            eval_every=10,
        )

        return ImportanceSampling(
            model, likelihood, objects,
            iter_per_obj=2,
            name="/tmp/test_sampler",
            decay="no_decay",
        )

    def test_sampler_constructs(self, sampler):
        """ImportanceSampling initializes without errors."""
        assert sampler.sample_state is not None
        assert len(sampler.proposals) > 0

    def test_single_body_sampling(self, sampler, objects):
        """run_single_body_sampling completes one object without crashing."""
        obj = objects.observed[0]
        obj.unfreeze_finalized_body()

        sampler.run_single_body_sampling(
            obj, total_iter=2, object_num=0, physics=False
        )

        assert len(sampler.likelihoods) >= 2
