from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

KernelName = Literal["mexican_hat", "heat", "meyer_like"]


def softmax(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    z = a - np.max(a)
    e = np.exp(z)
    return e / np.sum(e)


def soft_shrink(x: np.ndarray, lam: float | np.ndarray) -> np.ndarray:
    """Wavelet soft-thresholding S_lambda(x)=sign(x) max(|x|-lambda,0)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


@dataclass
class GLWTFilterBank:
    """Exact spectral Graph Laplacian Wavelet Transform filter bank.

    This class uses eigendecomposition for clarity and reproducibility on small/
    medium benchmark graphs. The repo also includes a Chebyshev utility in
    chebyshev.py for large-graph approximation without eigendecomposition.
    """

    laplacian: np.ndarray
    scales: Iterable[float] = (0.1, 0.3, 1.0, 3.0, 10.0)
    kernel: KernelName = "mexican_hat"
    eps: float = 1e-12

    def __post_init__(self):
        self.L = np.asarray(self.laplacian, dtype=float)
        # eigh gives ascending eigenvalues for symmetric matrices.
        eigvals, eigvecs = np.linalg.eigh(self.L)
        self.eigvals = np.clip(eigvals, 0.0, None)
        self.eigvecs = eigvecs
        self.scales = np.asarray(list(self.scales), dtype=float)
        self.filters = [self._make_filter_matrix(float(s)) for s in self.scales]

    def spectral_response(self, scale: float) -> np.ndarray:
        lam = self.eigvals
        x = scale * lam
        if self.kernel == "mexican_hat":
            # Bandpass graph wavelet, zero at DC and decays at high frequencies.
            g = x * np.exp(-x)
        elif self.kernel == "heat":
            # Low-pass heat diffusion response.
            g = np.exp(-x)
        elif self.kernel == "meyer_like":
            # Smooth compact-ish bump around x=1 in log-frequency.
            g = x * np.exp(-0.5 * (x - 1.0) ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        return g

    def _make_filter_matrix(self, scale: float) -> np.ndarray:
        g = self.spectral_response(scale)
        U = self.eigvecs
        return (U * g) @ U.T

    def transform(self, X: np.ndarray) -> list[np.ndarray]:
        """Return GLWT coefficients for node signals X with shape (n,d) or (n,)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return [H @ X for H in self.filters]

    def inverse_like(self, coeffs: list[np.ndarray]) -> list[np.ndarray]:
        """Apply the corresponding synthesis filter to coefficient arrays."""
        return [H @ C for H, C in zip(self.filters, coeffs)]

    def modulate(
        self,
        coeffs: list[np.ndarray],
        lambdas: np.ndarray,
        gammas: np.ndarray | None = None,
        phases: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        lambdas = np.asarray(lambdas, dtype=float)
        K = len(coeffs)
        gammas = np.ones(K) if gammas is None else np.asarray(gammas, dtype=float)
        phases = np.zeros(K) if phases is None else np.asarray(phases, dtype=float)
        out = []
        for k, C in enumerate(coeffs):
            out.append(gammas[k] * soft_shrink(C, lambdas[k]) * np.cos(phases[k]))
        return out

    def forward(
        self,
        X: np.ndarray,
        lambdas: np.ndarray,
        gammas: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
        coeffs = self.transform(X)
        mod = self.modulate(coeffs, lambdas=lambdas, gammas=gammas, phases=phases)
        recon_parts = self.inverse_like(mod)
        K = len(recon_parts)
        weights = np.ones(K) / K if alpha is None else softmax(np.asarray(alpha, dtype=float))
        recon = np.zeros_like(recon_parts[0])
        for w, R in zip(weights, recon_parts):
            recon += w * R
        return recon, coeffs, recon_parts, weights

    def features(self, X: np.ndarray, lambdas: np.ndarray | None = None, include_coeffs: bool = True) -> np.ndarray:
        """Create per-node deterministic GLWT features for rule induction.

        For an input matrix X=(n,d), each node receives feature summaries across
        scales: local coefficient energy, signed response mean, and reconstructed
        energy. These are symbolic/spectral features, not learned embeddings.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        K = len(self.scales)
        if lambdas is None:
            lambdas = np.zeros(K)
        coeffs = self.transform(X)
        mod = self.modulate(coeffs, lambdas=np.asarray(lambdas))
        recon = self.inverse_like(mod)
        feats = []
        for C, M, R in zip(coeffs, mod, recon):
            feats.extend([
                np.mean(C, axis=1),
                np.std(C, axis=1),
                np.mean(np.abs(C), axis=1),
                np.mean(M, axis=1),
                np.mean(np.abs(M), axis=1),
                np.mean(np.abs(R), axis=1),
            ])
        return np.stack(feats, axis=1)
