"""Evaluation helpers for the Quantum Autoencoder.

We use simple probability-based metrics to keep the ideas approachable.  The
functions below avoid heavy linear algebra and stick to intuitive concepts like
"how close are two probability distributions?".
"""

from __future__ import annotations

import numpy as np


def state_fidelity(target_probs: np.ndarray, reconstructed_probs: np.ndarray) -> float:
    """Compute a lightweight fidelity estimate between two probability vectors.

    A true quantum fidelity would require full density matrices.  For this
    beginner-friendly project we approximate similarity by taking the square
    root of the pairwise product of probabilities (the classical Bhattacharyya
    coefficient).

    Args:
        target_probs: Probabilities for the reference state.
        reconstructed_probs: Probabilities output by the autoencoder.

    Returns:
        A value between 0 and 1 where 1 means perfect agreement.
    """

    target_probs = np.asarray(target_probs)
    reconstructed_probs = np.asarray(reconstructed_probs)
    overlap = np.sum(np.sqrt(target_probs * reconstructed_probs))
    return float(np.clip(overlap, 0.0, 1.0))


def reconstruction_loss(target_probs: np.ndarray, reconstructed_probs: np.ndarray) -> float:
    """Mean squared error between target and reconstructed probabilities.

    The loss is classical but differentiable, which makes it a friendly choice
    for variational training with PennyLane.

    Args:
        target_probs: Probabilities for the reference state.
        reconstructed_probs: Probabilities output by the autoencoder.

    Returns:
        Scalar loss value where smaller is better.
    """

    target_probs = np.asarray(target_probs)
    reconstructed_probs = np.asarray(reconstructed_probs)
    return float(np.mean((target_probs - reconstructed_probs) ** 2))
