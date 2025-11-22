"""Dataset utilities for the simple Quantum Autoencoder example.

This module generates a tiny family of two-qubit states.  Each state is
produced by encoding a two-dimensional classical point into rotations on the
qubits.  Keeping the data small makes the training loop quick and beginner
friendly.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


def generate_training_states(n_states: int = 8) -> list[tuple[float, float]]:
    """Create a list of rotation angles describing training states.

    The training set is a grid of points inside the square ``[0, 1] x [0, 1]``.
    Each point is mapped to a pair of angles that will drive ``RY`` rotations on
    the two input qubits.  Using a grid gives enough variety for the
    autoencoder to learn while still being tiny.

    Args:
        n_states: Number of states to generate. The function builds the smallest
            square grid that contains at least ``n_states`` points and then
            truncates to exactly ``n_states``.

    Returns:
        A list of ``(angle_a, angle_b)`` pairs in radians.
    """

    # Determine a square grid size close to the requested number of states.
    side = int(np.ceil(np.sqrt(max(1, n_states))))
    grid = np.linspace(0.0, np.pi / 2, side)

    # Build combinations of angles. We stop after ``n_states`` so the function
    # respects the requested dataset size when a perfect square is not used.
    states: list[tuple[float, float]] = []
    for a in grid:
        for b in grid:
            states.append((float(a), float(b)))
            if len(states) >= n_states:
                return states
    return states


def prepare_state(angle_pair: tuple[float, float], wires: tuple[int, int]) -> None:
    """Encode a classical 2D point into a two-qubit quantum state.

    Each component of ``angle_pair`` drives a simple ``RY`` rotation.  We add a
    CNOT gate to sprinkle a little entanglement, which makes the compression
    task more interesting but still lightweight.

    Args:
        angle_pair: Two rotation angles (in radians) for the qubits.
        wires: The pair of wire indices used by the QNode.
    """

    a, b = angle_pair

    # First qubit holds the first feature; second qubit holds the second one.
    qml.RY(a, wires=wires[0])
    qml.RZ(a / 2, wires=wires[0])
    qml.RY(b, wires=wires[1])
    qml.RZ(b / 2, wires=wires[1])

    # Light entanglement encourages the encoder to learn correlations.
    qml.CNOT(wires=wires)
