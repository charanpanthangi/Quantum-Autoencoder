# Quantum Autoencoder (QAE) – Simple Compression Demo

## What This Project Does
- Creates a tiny family of two-qubit states from simple rotation angles.
- Trains a quantum autoencoder (QAE) to squeeze those states into a smaller
  latent space (one qubit) and then reconstruct them.
- Shows the training loss and latent-space behavior with SVG plots so you can
  see the compression in action.

## Why Quantum Autoencoders Are Interesting
- They reduce the number of qubits needed to represent certain families of
  states.
- They mirror classical autoencoders but operate on quantum data.
- They can help with quantum data compression and even error mitigation in
  larger workflows.

## Why We Use SVG Instead of PNG
> GitHub’s CODEX interface cannot preview binary image files like PNG or JPG and
> often shows “Binary files are not supported” in pull request views. To avoid
> this, all visualizations in this repository are saved as lightweight SVG
> (vector) images. SVGs are text-based, easy to diff, and render cleanly inside
> GitHub and CODEX.

## How the QAE Works (Simple Words)
- The encoder applies parameterized quantum gates that try to store the useful
  information on one "latent" qubit.
- The decoder uses that latent qubit plus an extra clean qubit to rebuild the
  original two-qubit state.
- Training adjusts the gate angles to minimize reconstruction error using
  gradient descent.

## Repository Structure
- `app/` – dataset generator, PennyLane QAE model, training loop, plotting, and CLI.
- `notebooks/` – Jupyter tutorial showing an end-to-end run.
- `examples/` – SVG plots produced by the CLI or notebook.
- `tests/` – quick pytest checks to ensure the pieces run without errors.

## How to Run
```bash
pip install -r requirements.txt
python app/main.py
```

To explore the notebook tutorial:
```bash
jupyter notebook notebooks/qae_demo.ipynb
```

## What You Should See
- Reconstruction loss decreasing over epochs.
- Printed reconstruction fidelity after training.
- Two SVG plots in `examples/` showing the loss curve and latent space.

## Future Work
- Try more qubits or deeper circuits.
- Experiment with different state families or encodings.
- Run the model on quantum hardware when available.
- Stack multiple encoder/decoder layers for richer compression.
