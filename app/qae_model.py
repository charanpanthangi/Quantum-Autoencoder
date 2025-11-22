"""Quantum autoencoder model built with PennyLane.

The goal of this small circuit is to concentrate information from two input
qubits into a single "latent" qubit, then reconstruct the original state.
We keep the design intentionally simple so beginners can follow every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pennylane as qml
from pennylane import numpy as np

from .dataset import prepare_state

# Two input qubits are sufficient for the demo.
WIRES: Tuple[int, int] = (0, 1)


@dataclass
class ModelParams:
    """Container for encoder and decoder parameters.

    Attributes:
        encoder: Rotation angles for the encoder layers with shape
            ``(n_layers, 2, 3)`` representing ``Rot`` parameters for each wire.
        decoder: Rotation angles for the decoder layers. Mirrors the encoder
            layout so both halves have comparable expressive power.
    """

    encoder: np.ndarray
    decoder: np.ndarray


def initialize_parameters(n_layers: int = 2, seed: int | None = None) -> ModelParams:
    """Create small random parameters for the encoder and decoder.

    Args:
        n_layers: Number of rotation-entanglement layers to use.
        seed: Optional random seed for reproducibility.

    Returns:
        ``ModelParams`` holding encoder and decoder tensors.
    """

    rng = np.random.default_rng(seed)
    # Shape: (layers, wires, parameters_per_rot)
    encoder = rng.normal(scale=0.2, size=(n_layers, 2, 3))
    decoder = rng.normal(scale=0.2, size=(n_layers, 2, 3))
    return ModelParams(encoder=encoder, decoder=decoder)


def _rot_layer(weights: np.ndarray, entangle_reverse: bool = False) -> None:
    """Apply ``Rot`` gates for one layer.

    Args:
        weights: Array with shape ``(2, 3)`` containing angles for both qubits.
        entangle_reverse: If ``True`` use a reversed CNOT to vary the structure.
    """

    for wire, angles in zip(WIRES, weights):
        qml.Rot(*angles, wires=wire)

    # A CNOT encourages correlations to flow between qubits.
    if entangle_reverse:
        qml.CNOT(wires=WIRES[::-1])
    else:
        qml.CNOT(wires=WIRES)


def _encoder(params: np.ndarray) -> None:
    """Run the encoder part of the circuit.

    The encoder aims to push as much information as possible into wire ``0``.
    """

    for layer_weights in params:
        _rot_layer(layer_weights, entangle_reverse=False)


def _decoder(params: np.ndarray) -> None:
    """Run the decoder part of the circuit.

    After compression we imagine discarding wire ``1``. In practice we keep the
    wire but encourage it to stay close to ``|0>`` so that the useful
    information must live on the latent wire ``0``. The decoder then tries to
    spread that information back across both wires to rebuild the input state.
    """

    for layer_weights in params:
        _rot_layer(layer_weights, entangle_reverse=True)


@qml.qnode(qml.device("default.qubit", wires=WIRES))
def qae_circuit(encoder_params: np.ndarray, decoder_params: np.ndarray, input_angles: Tuple[float, float]) -> List[float]:
    """Full autoencoder circuit returning output probabilities.

    Steps:
        1. Prepare the input state from classical angles.
        2. Apply the encoder layers.
        3. Conceptually compress by encouraging wire ``1`` to ``|0>``.
        4. Apply decoder layers to reconstruct the two-qubit state.
        5. Measure computational basis probabilities as a simple description of
           the reconstructed state.

    Args:
        encoder_params: Trainable encoder rotation weights.
        decoder_params: Trainable decoder rotation weights.
        input_angles: Angles used to prepare the input state.

    Returns:
        A list of probabilities for the four computational basis states.
    """

    # 1. Create the starting state.
    prepare_state(input_angles, wires=WIRES)

    # 2. Encode.
    _encoder(encoder_params)

    # 3. Gentle compression: rotate the trash wire toward |0> so information is
    # forced onto the latent wire. This is a soft form of discarding.
    qml.RX(-np.pi / 2, wires=WIRES[1])

    # 4. Decode.
    _decoder(decoder_params)

    # 5. Probabilities give a friendly, real-valued output for loss functions.
    return qml.probs(wires=WIRES)


def latent_probability(encoder_params: np.ndarray, input_angles: Tuple[float, float]) -> float:
    """Compute how often the trash qubit is near ``|0>`` after encoding.

    High probability means the encoder successfully moved most information onto
    the latent qubit.

    Args:
        encoder_params: Trainable parameters for the encoder only.
        input_angles: Angles describing the input state.

    Returns:
        Probability of measuring ``|0>`` on the second qubit immediately after
        the encoder.
    """

    @qml.qnode(qml.device("default.qubit", wires=WIRES))
    def _encoder_probe() -> float:
        prepare_state(input_angles, wires=WIRES)
        _encoder(encoder_params)
        return qml.probs(wires=[WIRES[1]])[0]

    return float(_encoder_probe())
