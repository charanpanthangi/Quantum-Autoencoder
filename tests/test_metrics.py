"""Tests for metric helpers used by the QAE."""

from __future__ import annotations

import numpy as np

from app.metrics import reconstruction_loss, state_fidelity


def test_state_fidelity_bounds():
    fid = state_fidelity(np.array([0.5, 0.5, 0, 0]), np.array([0.5, 0.5, 0, 0]))
    assert 0.0 <= fid <= 1.0
    assert np.isclose(fid, 1.0)


def test_reconstruction_loss_simple_case():
    loss = reconstruction_loss(np.array([1, 0, 0, 0]), np.array([0.9, 0.1, 0, 0]))
    assert loss > 0
    assert loss < 0.02
