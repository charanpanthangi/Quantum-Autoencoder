"""Simple classical compression baseline for comparison.

This module uses Principal Component Analysis (PCA) to compress two-dimensional
points down to a one-dimensional latent space and then reconstruct them.  The
resulting reconstruction error acts as a rough classical baseline against which
we can compare the quantum autoencoder.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


def run_pca_baseline(points: np.ndarray) -> float:
    """Run a 1D PCA compression and return the reconstruction error.

    Args:
        points: Array of shape ``(n_samples, 2)`` holding the classical data
            encoded by the quantum dataset.

    Returns:
        Mean squared reconstruction error (smaller is better).
    """

    # Fit a PCA model that keeps just one principal component.
    pca = PCA(n_components=1)
    latent = pca.fit_transform(points)
    reconstructed = pca.inverse_transform(latent)
    return float(mean_squared_error(points, reconstructed))
