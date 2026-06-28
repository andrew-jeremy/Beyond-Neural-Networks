from __future__ import annotations

import numpy as np


def make_smooth_graph_signals(laplacian: np.ndarray, num_signals: int = 180, noise_sigma: float = 0.30, heat_scale: float = 4.0, seed: int = 7):
    """Generate smooth graph signals and noisy observations.

    Clean signals are obtained by heat diffusion of white noise:
        f = U exp(-heat_scale Lambda) U^T z.
    """
    rng = np.random.default_rng(seed)
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    heat = np.exp(-heat_scale * np.clip(eigvals, 0, None))
    H = (eigvecs * heat) @ eigvecs.T
    z = rng.normal(size=(num_signals, laplacian.shape[0]))
    clean = z @ H.T
    # Normalize each signal to comparable energy.
    clean = clean / (clean.std(axis=1, keepdims=True) + 1e-12)
    noisy = clean + rng.normal(scale=noise_sigma, size=clean.shape)
    return clean, noisy
