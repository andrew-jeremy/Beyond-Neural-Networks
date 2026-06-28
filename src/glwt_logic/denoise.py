from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from scipy.optimize import minimize

from .glwt import GLWTFilterBank, soft_shrink


def _snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    num = np.sum(clean ** 2)
    den = np.sum((clean - estimate) ** 2) + 1e-12
    return 10.0 * np.log10(num / den)


@dataclass
class GLWTDenoiser:
    """Trainable non-neural GLWT denoiser.

    It learns a global threshold factor and convex scale weights by minimizing
    reconstruction MSE. This follows the paper's spectral shrinkage objective
    while avoiding neural layers.
    """
    bank: GLWTFilterBank
    beta_entropy: float = 0.0
    lambda_factor_grid: tuple[float, ...] = tuple(np.linspace(0.0, 1.5, 16))

    def fit(self, noisy_train: np.ndarray, clean_train: np.ndarray, noisy_val: np.ndarray, clean_val: np.ndarray, noise_sigma: float):
        self.history_ = []
        K = len(self.bank.scales)
        best = None
        for lf in self.lambda_factor_grid:
            lambdas = np.ones(K) * float(lf) * noise_sigma
            R_train = self._reconstruction_parts(noisy_train, lambdas)  # (m,n,K)
            weights = self._fit_weights(R_train, clean_train)
            train_pred = np.tensordot(R_train, weights, axes=([2], [0]))
            R_val = self._reconstruction_parts(noisy_val, lambdas)
            val_pred = np.tensordot(R_val, weights, axes=([2], [0]))
            record = {
                "lambda_factor": float(lf),
                "lambdas": lambdas.tolist(),
                "weights": weights.tolist(),
                "train_mse": float(np.mean((train_pred - clean_train) ** 2)),
                "val_mse": float(np.mean((val_pred - clean_val) ** 2)),
                "train_snr_db": float(_snr_db(clean_train, train_pred)),
                "val_snr_db": float(_snr_db(clean_val, val_pred)),
            }
            self.history_.append(record)
            if best is None or record["val_mse"] < best["val_mse"]:
                best = record
        self.lambda_factor_ = best["lambda_factor"]
        self.lambdas_ = np.array(best["lambdas"], dtype=float)
        self.weights_ = np.array(best["weights"], dtype=float)
        return self

    def _reconstruction_parts(self, noisy: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
        # Input signals are (num_signals, n). Bank expects (n,d), so operate by transpose.
        X = noisy.T
        coeffs = self.bank.transform(X)
        parts = []
        for H, C, lam in zip(self.bank.filters, coeffs, lambdas):
            M = soft_shrink(C, lam)
            R = H @ M
            parts.append(R.T)
        return np.stack(parts, axis=2)

    def _fit_weights(self, R: np.ndarray, clean: np.ndarray) -> np.ndarray:
        K = R.shape[2]
        def obj(w):
            pred = np.tensordot(R, w, axes=([2], [0]))
            mse = np.mean((pred - clean) ** 2)
            if self.beta_entropy != 0:
                ww = np.clip(w, 1e-12, 1.0)
                mse += self.beta_entropy * np.sum(ww * np.log(ww))
            return mse
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * K
        res = minimize(obj, np.ones(K) / K, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
        if not res.success:
            return np.ones(K) / K
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()

    def predict(self, noisy: np.ndarray) -> np.ndarray:
        R = self._reconstruction_parts(noisy, self.lambdas_)
        return np.tensordot(R, self.weights_, axes=([2], [0]))

    def metrics(self, noisy: np.ndarray, clean: np.ndarray) -> dict[str, float]:
        pred = self.predict(noisy)
        baseline_mse = float(np.mean((noisy - clean) ** 2))
        mse = float(np.mean((pred - clean) ** 2))
        return {
            "mse_noisy": baseline_mse,
            "mse_glwt": mse,
            "mse_improvement_pct": 100.0 * (baseline_mse - mse) / max(baseline_mse, 1e-12),
            "snr_noisy_db": float(_snr_db(clean, noisy)),
            "snr_glwt_db": float(_snr_db(clean, pred)),
            "snr_gain_db": float(_snr_db(clean, pred) - _snr_db(clean, noisy)),
        }

    def save(self, path: str):
        payload = {
            "scales": self.bank.scales.tolist(),
            "kernel": self.bank.kernel,
            "lambda_factor": float(self.lambda_factor_),
            "lambdas": self.lambdas_.tolist(),
            "weights": self.weights_.tolist(),
            "history": self.history_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
