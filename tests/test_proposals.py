"""
Tests for sampling/proposals.py — pure-numpy, no GPU required.
"""

import numpy as np
import pytest
from numpy.linalg import norm

from scene_physics.properties.priors import Priors
from scene_physics.sampling.proposals import (
    SixDOFProposal,
    exponential_decay,
    linear_decay,
    no_decay,
)

NUM_WORLDS = 10
PRIORS = Priors(total_iter=50)


# ── Decay schedules ──────────────────────────────────────────────────────────

def test_linear_decay_starts_at_one():
    assert linear_decay(0, 100) == pytest.approx(1.0)


def test_linear_decay_floor():
    # Should never go below 0.1
    assert linear_decay(10_000, 100) == pytest.approx(0.1)


def test_linear_decay_monotone():
    vals = [linear_decay(i, 100) for i in range(0, 110, 10)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_exponential_decay_floor():
    assert exponential_decay(10_000, half_life=10) == pytest.approx(0.1)


def test_exponential_decay_monotone():
    vals = [exponential_decay(i, half_life=50) for i in range(0, 500, 50)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_no_decay_constant():
    for i in range(100):
        assert no_decay(i, half_life=50) == 1.0


# ── initial_positions ────────────────────────────────────────────────────────

@pytest.fixture
def proposal():
    return SixDOFProposal(PRIORS, NUM_WORLDS, seed=0)


def test_initial_positions_shape(proposal):
    pos = proposal.initial_positions()
    assert pos.shape == (NUM_WORLDS, 7)


def test_initial_positions_identity_quaternion(proposal):
    pos = proposal.initial_positions()
    # XYZW: qx=qy=qz=0, qw=1
    assert np.all(pos[:, 3:6] == 0.0)
    assert np.all(pos[:, 6] == 1.0)


def test_initial_positions_y_nonnegative(proposal):
    pos = proposal.initial_positions()
    assert np.all(pos[:, 1] >= 0.0)


# ── _perturb_rotation ────────────────────────────────────────────────────────

def test_perturb_rotation_shape():
    q_in = np.array([0.0, 0.0, 0.0, 1.0])
    q_out = SixDOFProposal._perturb_rotation(q_in, rot_std=0.1)
    assert q_out.shape == (4,)


def test_perturb_rotation_unit_norm():
    q_in = np.array([0.0, 0.0, 0.0, 1.0])
    for _ in range(20):
        q_out = SixDOFProposal._perturb_rotation(q_in, rot_std=0.5)
        assert norm(q_out) == pytest.approx(1.0, abs=1e-6)


def test_perturb_rotation_zero_std_identity():
    # Zero perturbation should return the same rotation
    q_in = np.array([0.0, 0.0, 0.0, 1.0])
    q_out = SixDOFProposal._perturb_rotation(q_in, rot_std=0.0)
    assert np.allclose(np.abs(q_out), np.abs(q_in), atol=1e-6)


# ── propose_general ──────────────────────────────────────────────────────────

def test_propose_general_shape(proposal):
    pos = proposal.initial_positions()
    out = proposal.propose_general(pos.copy(), epoch_num=0, count=False)
    assert out.shape == (NUM_WORLDS, 7)


def test_propose_general_preserves_index0_position(proposal):
    pos = proposal.initial_positions()
    pos[0, :3] = np.array([0.5, 0.3, -0.2])
    pos[0, 3:] = np.array([0.0, 0.0, 0.0, 1.0])
    original_xyz = pos[0, :3].copy()

    out = proposal.propose_general(pos, epoch_num=0, count=False)

    # X and Z may be clipped but should not have noise added
    assert out[0, 1] == pytest.approx(original_xyz[1])  # Y never clipped


def test_propose_general_clips_xz(proposal):
    pos = proposal.initial_positions()
    pos[:, 0] = 999.0   # way outside x_max
    pos[:, 2] = -999.0  # way outside z_min

    out = proposal.propose_general(pos, epoch_num=0, count=False)

    assert np.all(out[:, 0] <= PRIORS.x_max)
    assert np.all(out[:, 2] >= PRIORS.z_min)


def test_propose_general_quaternions_unit_norm(proposal):
    pos = proposal.initial_positions()
    out = proposal.propose_general(pos, epoch_num=0, count=False)
    norms = norm(out[:, 3:], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_propose_general_increments_cur_iters(proposal):
    pos = proposal.initial_positions()
    before = proposal.cur_iters
    proposal.propose_general(pos, epoch_num=0, count=False)
    assert proposal.cur_iters == before + 1


# ── get_std ──────────────────────────────────────────────────────────────────

def test_get_std_respects_priors():
    priors = Priors(pos_std=0.2, rot_std=0.05, total_iter=100)
    p = SixDOFProposal(priors, NUM_WORLDS, seed=0, schedule="no_decay")
    pos_std, rot_std = p.get_std()
    assert pos_std == pytest.approx(0.2)
    assert rot_std == pytest.approx(0.05)


def test_get_std_decreases_with_linear_schedule():
    priors = Priors(pos_std=1.0, rot_std=1.0, total_iter=100)
    p = SixDOFProposal(priors, NUM_WORLDS, seed=0, schedule="linear")
    pos = p.initial_positions()

    stds_early = p.get_std()[0]
    for _ in range(80):
        p.propose_general(pos, epoch_num=0, count=False)
    stds_late = p.get_std()[0]

    assert stds_late < stds_early


# ── epoch tracking ───────────────────────────────────────────────────────────

def test_epoch_tracking_with_count(proposal):
    pos = proposal.initial_positions()
    proposal.propose_general(pos, epoch_num=3, count=True)
    assert 3 in proposal.epoch_num
    assert len(proposal.save_pos_std) == 1
    assert len(proposal.save_rot_std) == 1


def test_epoch_tracking_without_count(proposal):
    pos = proposal.initial_positions()
    proposal.propose_general(pos, epoch_num=3, count=False)
    assert len(proposal.epoch_num) == 0


# ── Priors ───────────────────────────────────────────────────────────────────

def test_priors_defaults():
    p = Priors()
    assert p.init_mean == 0.0
    assert p.pos_std == 0.1
    assert p.rot_std == 0.1
    assert p.x_min == -1.0 and p.x_max == 1.0
    assert p.z_min == -1.0 and p.z_max == 1.0


def test_priors_custom():
    p = Priors(pos_std=0.5, rot_std=0.2, x_min=-2.0, x_max=2.0)
    assert p.pos_std == 0.5
    assert p.x_min == -2.0
