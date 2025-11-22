"""Tests for dataset generation and state preparation helpers."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

from app.dataset import generate_training_states, prepare_state


def test_generate_training_states_count():
    states = generate_training_states(5)
    assert len(states) == 5
    for pair in states:
        assert len(pair) == 2


def test_prepare_state_runs():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def _circuit():
        prepare_state((0.1, 0.2), wires=(0, 1))
        return qml.probs(wires=(0, 1))

    probs = _circuit()
    assert np.isclose(np.sum(probs), 1.0)
    assert probs.shape == (4,)
