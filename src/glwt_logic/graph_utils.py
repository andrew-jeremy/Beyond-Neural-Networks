from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split


def karate_graph():
    """Return Zachary Karate Club graph, labels, and adjacency.

    Labels are mapped from the built-in NetworkX 'club' attribute:
    Mr. Hi -> 0, Officer -> 1.
    """
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    labels = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)], dtype=int)
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=float)
    return G, A, labels


def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    A = np.asarray(A, dtype=float)
    deg = A.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_inv_sqrt = np.diag(inv_sqrt)
    return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt


def combinatorial_laplacian(A: np.ndarray) -> np.ndarray:
    """Combinatorial graph Laplacian L = D - A."""
    return np.diag(np.asarray(A).sum(axis=1)) - np.asarray(A, dtype=float)


def stratified_masks(labels: np.ndarray, train_size: float = 0.5, val_size: float = 0.25, seed: int = 7):
    """Create stratified train/validation/test index splits."""
    idx = np.arange(len(labels))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, labels, train_size=train_size, random_state=seed, stratify=labels
    )
    # val_size is fraction of full dataset; convert relative to temp split.
    rel_val = val_size / (1.0 - train_size)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, train_size=rel_val, random_state=seed + 1, stratify=y_temp
    )
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def structural_node_signals(A: np.ndarray) -> np.ndarray:
    """Build non-neural node input signals from graph structure.

    We combine identity impulses with simple scalar graph descriptors. The identity
    matrix lets GLWT produce scale-localized structural fingerprints for every node.
    Extra columns are deterministic descriptors, not learned embeddings.
    """
    G = nx.from_numpy_array(A)
    n = A.shape[0]
    deg = A.sum(axis=1)
    deg_norm = deg / max(float(deg.max()), 1.0)
    clustering = np.array([nx.clustering(G, i) for i in range(n)], dtype=float)
    triangles = np.array([nx.triangles(G, i) for i in range(n)], dtype=float)
    if triangles.max() > 0:
        triangles = triangles / triangles.max()
    descriptors = np.stack([deg_norm, clustering, triangles], axis=1)
    return np.concatenate([np.eye(n), descriptors], axis=1)
