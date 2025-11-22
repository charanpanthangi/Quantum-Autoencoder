"""Tests for the training loop to ensure losses decrease."""

from __future__ import annotations

import numpy as np

from app.dataset import generate_training_states
from app.trainer import train_qae


def test_training_loss_decreases():
    data = generate_training_states(4)
    results = train_qae(data, n_epochs=3, lr=0.3, seed=1)
    history = results["loss_history"]
    assert len(history) == 3
    # Loss should decrease at least once across epochs.
    assert history[-1] <= max(history[0], history[1])
