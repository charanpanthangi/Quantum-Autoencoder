"""CLI entry point for training and visualizing the Quantum Autoencoder.

Run with:

    python app/main.py --epochs 50 --lr 0.2

The script generates a tiny dataset, trains the QAE, reports reconstruction
fidelity, runs a PCA baseline, and saves SVG plots in the ``examples/`` folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .classical_baseline import run_pca_baseline
from .dataset import generate_training_states
from .plots import plot_latent_space, plot_reconstruction_loss
from .trainer import evaluate_qae, train_qae


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a lightweight training run."""

    parser = argparse.ArgumentParser(description="Train a simple quantum autoencoder")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for gradient descent")
    parser.add_argument("--states", type=int, default=8, help="Number of training states to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    """Main orchestration function executed by the CLI."""

    args = parse_args()
    training_states = generate_training_states(args.states)

    print("Generating training data...")
    for idx, angles in enumerate(training_states):
        print(f"  State {idx}: angles={angles}")

    print("\nStarting QAE training...\n")
    results = train_qae(training_states, n_epochs=args.epochs, lr=args.lr, seed=args.seed)
    encoder_params = results["encoder"]
    decoder_params = results["decoder"]
    loss_history = results["loss_history"]

    print("Training complete. Evaluating reconstruction fidelity...\n")
    metrics = evaluate_qae(training_states, encoder_params, decoder_params)

    mean_fidelity = metrics["mean_fidelity"]
    latent_probs = metrics["latent_probs"]
    print(f"Average fidelity on training set: {mean_fidelity:.3f}")

    # PCA baseline on the raw classical angles (scaled to [0,1]).
    classical_points = np.array(training_states) / (np.pi / 2)
    pca_error = run_pca_baseline(classical_points)
    print(f"Classical PCA reconstruction MSE (1D latent): {pca_error:.4f}\n")

    # Save SVG plots to the examples folder.
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    loss_svg = examples_dir / "qae_reconstruction_loss.svg"
    latent_svg = examples_dir / "qae_latent_space.svg"

    plot_reconstruction_loss(loss_history, str(loss_svg))
    labels = [f"state {i}" for i in range(len(training_states))]
    plot_latent_space(latent_probs, labels, str(latent_svg))

    print("SVG plots saved:")
    print(f"  Loss curve: {loss_svg}")
    print(f"  Latent space: {latent_svg}")


if __name__ == "__main__":
    main()
