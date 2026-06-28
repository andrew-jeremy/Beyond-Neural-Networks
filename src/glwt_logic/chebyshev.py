from __future__ import annotations

import numpy as np


def chebyshev_apply(L: np.ndarray, x: np.ndarray, coeffs: np.ndarray, lambda_max: float | None = None) -> np.ndarray:
    """Apply sum_k coeffs[k] T_k(L_tilde) x without eigendecomposition.

    This is included for large-graph GLWT approximation. It is not used by the
    default small benchmark script, which uses exact eigendecomposition so that
    metrics are deterministic and easy to inspect.
    """
    L = np.asarray(L, dtype=float)
    x = np.asarray(x, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    if lambda_max is None:
        lambda_max = float(np.linalg.eigvalsh(L).max())
    L_tilde = (2.0 / lambda_max) * L - np.eye(L.shape[0])
    T0 = x
    out = coeffs[0] * T0
    if len(coeffs) == 1:
        return out
    T1 = L_tilde @ x
    out = out + coeffs[1] * T1
    for k in range(2, len(coeffs)):
        T2 = 2.0 * (L_tilde @ T1) - T0
        out = out + coeffs[k] * T2
        T0, T1 = T1, T2
    return out
