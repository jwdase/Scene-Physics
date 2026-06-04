"""Invariants behind "why does the max-likelihood plot dip if we keep the best?"

Findings these tests pin down (scene001, 4 worlds, short rollout):

  * Render + scoring is bit-deterministic            -> test_render_score_deterministic
  * Worlds are collision-isolated (spacing=0 is safe) -> test_worlds_isolated
  * The elite (argmax) pose is carried into the scored
    world, so the top score reproduces exactly         -> test_elite_mapping_persists
  * The XPBD *solver* is NOT deterministic on GPU      -> test_physics_solver_deterministic [xfail]

The last one is the actual cause of the wandering max: recombination is fixed
(observed objects are identical across worlds now), isolation and elite mapping
are correct, but re-simulating the preserved elite gives a slightly different
settle each iteration, so its score jitters by a few units run-to-run.

Run on a CUDA box:  uv run pytest tests/test_sampling_invariants.py -v -s -m gpu
"""

import json
import os

# Small world count + no XLA preallocation BEFORE importing the GPU stack.
os.environ["NUM_WORLDS"] = "4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from scene_physics.simulation import sim_sampling as ss
from scene_physics.properties.shapes import Scene_Makeup, object_collection
from scene_physics.likelihood.likelihoods import ParallelPhysicsLikelihood
from scene_physics.sampling.proposals import ExpDecayProposal
from scene_physics.configs.camera import default_camera

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "resources" / "generated_scenes" / "scene001"
SCENE_USD = str(SCENE / "data" / "scene001_physics.usdc")
PRIORS = str(SCENE / "data" / "scene001_priors.json")
MAKEUP = str(SCENE / "data" / "scene001_makeup.json")

# Short rollout: these invariants are independent of frame count, so keep it cheap.
FRAMES = 10
EVAL_EVERY = 10


def _makeup() -> Scene_Makeup:
    mk = json.load(open(MAKEUP))
    return Scene_Makeup(
        static=set(mk["static"]),
        observed=set(mk["observed"]),
        hidden=set(mk["hidden"]),
    )


@pytest.fixture(scope="module")
def env():
    """Build the GPU model / camera / likelihood once for the module."""
    makeup = _makeup()
    model, _ = ss.build_worlds(SCENE_USD, makeup)
    camera = ss.MultiWorldCamera(default_camera, model, num_worlds=ss.NUM_WORLDS)
    point_cloud = ss.gen_point_cloud(SCENE_USD, default_camera)
    likelihood = ParallelPhysicsLikelihood(
        camera, point_cloud, model, frames=FRAMES, eval_every=EVAL_EVERY
    )
    return model, makeup, likelihood


@pytest.fixture
def initialized(env):
    """Fresh object collection + initialized state (cheap, per test)."""
    model, makeup, likelihood = env
    objects = object_collection(model, makeup, ss.NUM_WORLDS)
    objects.assign_priors(PRIORS, ExpDecayProposal, 10, np.random.default_rng(0))
    state = model.state()
    objects.initialize(state)
    return model, objects, likelihood, state


@pytest.mark.gpu
def test_render_score_deterministic(initialized):
    """still() (render + score, no solver) is bit-exact across calls."""
    _, _, likelihood, state = initialized
    s1 = np.asarray(likelihood.still(state))
    s2 = np.asarray(likelihood.still(state))
    np.testing.assert_array_equal(s1, s2)


@pytest.mark.gpu
def test_elite_mapping_persists(initialized):
    """propose() carries the argmax pose into world 0; with deterministic scoring
    the elite reproduces the previous top score exactly, so max cannot decrease.

    Uses still() so the result is not masked by solver non-determinism — this
    isolates the *recombination/mapping* logic, which is what was fixed.
    """
    _, objects, likelihood, state = initialized
    L0 = np.asarray(likelihood.still(state))
    m = int(np.argmax(L0))

    objects.dynamic[0].propose(state, L0)  # row 0 := argmax pose; observed untouched
    L1 = np.asarray(likelihood.still(state))

    assert L1[0] == pytest.approx(L0[m], abs=1e-3)  # elite world reproduces the best
    assert L1.max() >= L0.max() - 1e-3              # running max never decreases


@pytest.mark.gpu
def test_worlds_isolated(initialized):
    """Moving world 1's hidden body far away must not propagate to other worlds.

    Coupling would make the other worlds shift comparably to world 1; isolation
    keeps them at the solver-noise level (orders of magnitude smaller).
    """
    _, objects, likelihood, state = initialized
    base = likelihood.physics(state)

    hidden = objects.dynamic[0]
    q = state.body_q.numpy()
    q[hidden.allocs[1], 0] += 100.0  # fling world 1's hidden block 100 m in X
    state.body_q = wp.array(q, dtype=wp.transformf, device=state.body_q.device)
    after = likelihood.physics(state)

    delta = np.abs(after - base)
    other = [w for w in range(ss.NUM_WORLDS) if w != 1]
    assert delta[1] > 50.0                       # world 1 genuinely changed a lot
    assert delta[other].max() < 0.1 * delta[1]   # others stayed at noise level


@pytest.mark.gpu
@pytest.mark.xfail(
    reason="XPBD GPU solver is non-deterministic (atomic contact accumulation); "
    "this run-to-run settle jitter is the root cause of the wandering max plot.",
    strict=False,
)
def test_physics_solver_deterministic(initialized):
    """Desired (currently unmet) contract: same state -> same physics() score.

    Contrast with test_render_score_deterministic, which passes — proving the
    non-determinism lives in the solver, not the camera/scoring.
    """
    _, _, likelihood, state = initialized
    p1 = likelihood.physics(state)
    p2 = likelihood.physics(state)
    np.testing.assert_array_equal(p1, p2)
