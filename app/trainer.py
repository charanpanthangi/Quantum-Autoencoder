"""Training utilities for the Quantum Autoencoder.

The routines here implement a simple hybrid optimization loop: a quantum circuit
runs inside PennyLane while a classical optimizer nudges the gate angles to
reduce reconstruction loss.  Everything is intentionally lightweight so it can
run quickly on a CPU.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pennylane as qml
from pennylane import numpy as np

from .dataset import prepare_state
from .metrics import reconstruction_loss, state_fidelity
from .qae_model import initialize_parameters, latent_probability, qae_circuit

WIRES = (0, 1)


@qml.qnode(qml.device("default.qubit", wires=WIRES))
def _target_probs(input_angles: Tuple[float, float]) -> List[float]:
    """Reference probabilities for the raw (uncompressed) input state."""

    prepare_state(input_angles, wires=WIRES)
    return qml.probs(wires=WIRES)


def train_qae(training_data: List[Tuple[float, float]], n_epochs: int = 50, lr: float = 0.2, seed: int | None = 0) -> Dict[str, object]:
    """Train the quantum autoencoder on a small dataset.

    Args:
        training_data: List of angle pairs describing input states.
        n_epochs: Number of gradient-descent epochs.
        lr: Learning rate for the optimizer.
        seed: Optional random seed for repeatable results.

    Returns:
        Dictionary with learned parameters and training history.
    """

    params = initialize_parameters(seed=seed)
    encoder_params = np.array(params.encoder, requires_grad=True)
    decoder_params = np.array(params.decoder, requires_grad=True)

    opt = qml.GradientDescentOptimizer(stepsize=lr)
    loss_history: List[float] = []

    def cost_fn(enc, dec, angles):
        # Use qml.math operations to keep the expression differentiable.
        output_probs = qae_circuit(enc, dec, angles)
        target = _target_probs(angles)
        return qml.math.mean((output_probs - target) ** 2)

    for _ in range(n_epochs):
        for angles in training_data:
            encoder_params, decoder_params, _ = opt.step(cost_fn, encoder_params, decoder_params, angles)

        # Track the average loss after each epoch without updating parameters.
        epoch_losses = []
        for angles in training_data:
            output_probs = qae_circuit(encoder_params, decoder_params, angles)
            target = _target_probs(angles)
            epoch_losses.append(reconstruction_loss(target, output_probs))
        loss_history.append(float(np.mean(epoch_losses)))

    return {
        "encoder": encoder_params,
        "decoder": decoder_params,
        "loss_history": loss_history,
    }


def evaluate_qae(training_data: List[Tuple[float, float]], encoder_params, decoder_params) -> Dict[str, float]:
    """Compute reconstruction fidelity and latent probabilities on the dataset."""

    fidelities = []
    latent_probs = []
    for idx, angles in enumerate(training_data):
        output_probs = qae_circuit(encoder_params, decoder_params, angles)
        target = _target_probs(angles)
        fidelities.append(state_fidelity(target, output_probs))
        latent_probs.append(latent_probability(encoder_params, angles))

    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "latent_probs": latent_probs,
    }
