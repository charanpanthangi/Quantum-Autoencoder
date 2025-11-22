"""Plotting helpers that only produce SVG outputs.

SVG images stay text-based, so GitHub can render and diff them without showing
"binary file" warnings.  Every plot here calls ``plt.savefig(..., format='svg')``
to enforce that rule.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction_loss(history: list[float], output_path: str) -> None:
    """Plot the reconstruction loss over epochs and save as SVG.

    Args:
        history: List of average losses per epoch.
        output_path: Path to save the SVG file.
    """

    epochs = np.arange(1, len(history) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history, marker="o", color="purple", label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Quantum Autoencoder reconstruction loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_latent_space(latent_probs: list[float], labels: list[str], output_path: str) -> None:
    """Visualize latent probabilities on a line to show clustering.

    Args:
        latent_probs: Probability of the trash qubit being ``|0>`` for each
            training sample. Higher means better compression.
        labels: Friendly labels for each state.
        output_path: Path to save the SVG file.
    """

    plt.figure(figsize=(7, 2.5))
    plt.scatter(latent_probs, np.zeros_like(latent_probs), c="teal", s=60)
    for x, label in zip(latent_probs, labels):
        plt.text(x, 0.02, label, rotation=45, ha="center", va="bottom", fontsize=8)

    plt.yticks([])
    plt.xlabel("P(trash qubit = |0>) after encoding")
    plt.title("Latent space probabilities (close to 1.0 means strong compression)")
    plt.xlim(0, 1)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
