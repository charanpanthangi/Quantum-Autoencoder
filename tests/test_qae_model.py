"""Tests for the PennyLane quantum autoencoder circuit."""

from __future__ import annotations

from pennylane import numpy as np

from app.qae_model import initialize_parameters, qae_circuit


def test_qae_forward_output_shape():
    params = initialize_parameters(seed=1)
    probs = qae_circuit(params.encoder, params.decoder, (0.1, 0.2))
    assert probs.shape == (4,)
    assert np.isclose(np.sum(probs), 1.0)


def test_qae_probabilities_range():
    params = initialize_parameters(seed=2)
    probs = qae_circuit(params.encoder, params.decoder, (0.3, 0.4))
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
