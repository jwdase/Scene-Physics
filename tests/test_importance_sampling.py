"""
Tests for pure-numpy logic in sampling/parallel_mh.py — no GPU required.

ImportanceSampling._generate_positions is tested by calling the method on a
minimal stub that only sets the attributes the method actually reads.
"""

import numpy as np
import pytest
from numpy.linalg import norm


# ── Minimal stub — avoids importing Newton/Warp ──────────────────────────────

class _StubIS:
    """Exposes only the numpy-only methods of ImportanceSampling."""

    def __init__(self, seed=42):
        self.np_seed = np.random.SeedSequence(seed)

    # Copy the method under test verbatim so tests exercise the real logic.
    from scene_physics.sampling.parallel_mh import ImportanceSampling
    _generate_positions = ImportanceSampling._generate_positions


# ── _generate_positions ──────────────────────────────────────────────────────

NUM_WORLDS = 20


@pytest.fixture
def stub():
    return _StubIS(seed=42)


def _make_positions(n=NUM_WORLDS):
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((n, 7))
    # Normalise quaternion columns
    pos[:, 3:] /= norm(pos[:, 3:], axis=1, keepdims=True)
    return pos


def test_generate_positions_shape(stub):
    pos = _make_positions()
    scores = np.random.default_rng(1).standard_normal(NUM_WORLDS)
    out = stub._generate_positions(pos, scores)
    assert out.shape == (NUM_WORLDS, 7)


def test_generate_positions_top_is_first(stub):
    pos = _make_positions()
    scores = np.zeros(NUM_WORLDS)
    best_idx = 7
    scores[best_idx] = 100.0  # clear winner

    out = stub._generate_positions(pos, scores)

    assert np.allclose(out[0], pos[best_idx])


def test_generate_positions_rows_come_from_input(stub):
    """Every output row must be one of the input rows."""
    pos = _make_positions()
    scores = np.random.default_rng(2).standard_normal(NUM_WORLDS)
    out = stub._generate_positions(pos, scores)

    for row in out:
        assert any(np.allclose(row, p) for p in pos), \
            "Output row not found in input positions"


def test_generate_positions_deterministic_given_seed():
    pos = _make_positions()
    scores = np.random.default_rng(3).standard_normal(NUM_WORLDS)

    out1 = _StubIS(seed=99)._generate_positions(pos, scores)
    out2 = _StubIS(seed=99)._generate_positions(pos, scores)

    assert np.allclose(out1, out2)
