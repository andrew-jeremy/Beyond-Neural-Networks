from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text


@dataclass
class GLWTRuleClassifier:
    """Symbolic rule classifier over deterministic GLWT features.

    Uses a shallow decision tree as a rule induction engine. This is not a neural
    network: it learns threshold predicates over wavelet features and exports a
    human-readable rule trace.
    """
    max_depth: int = 3
    min_samples_leaf: int = 2
    random_state: int = 7

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        self.feature_names_ = feature_names or [f"x{j}" for j in range(X.shape[1])]
        self.tree_ = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.tree_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.tree_.predict(X)

    def metrics(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        }

    def rules_text(self) -> str:
        return export_text(self.tree_, feature_names=self.feature_names_)

    def save_rules(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.rules_text())

    def save_metadata(self, path: str):
        payload = {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_names": self.feature_names_,
            "rules": self.rules_text(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
