from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from glwt_logic.graph_utils import karate_graph, normalized_laplacian, structural_node_signals, stratified_masks
from glwt_logic.glwt import GLWTFilterBank
from glwt_logic.denoise import GLWTDenoiser
from glwt_logic.rules import GLWTRuleClassifier
from glwt_logic.synthetic import make_smooth_graph_signals


def feature_names(scales):
    names = []
    for s in scales:
        tag = str(s).replace('.', 'p')
        names.extend([
            f"mean_c_s{tag}", f"std_c_s{tag}", f"mean_abs_c_s{tag}",
            f"mean_phi_s{tag}", f"mean_abs_phi_s{tag}", f"mean_abs_recon_s{tag}",
        ])
    return names


def run(out_dir: Path, seed: int = 7):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "rules").mkdir(exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    # Standard small graph benchmark: Zachary's Karate Club.
    G, A, y = karate_graph()
    L = normalized_laplacian(A)
    wavelet_bank = GLWTFilterBank(L, scales=(0.1, 0.3, 1.0, 3.0, 10.0), kernel="mexican_hat")
    denoise_bank = GLWTFilterBank(L, scales=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0), kernel="heat")

    # ---------------- Denoising ----------------
    clean, noisy = make_smooth_graph_signals(L, num_signals=240, noise_sigma=0.30, heat_scale=4.0, seed=seed)
    idx = np.arange(clean.shape[0])
    train_idx, temp_idx = train_test_split(idx, train_size=0.60, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.50, random_state=seed + 1)
    denoiser = GLWTDenoiser(denoise_bank, beta_entropy=0.0)
    denoiser.fit(noisy[train_idx], clean[train_idx], noisy[val_idx], clean[val_idx], noise_sigma=0.30)
    den_rows = []
    for split, split_idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        m = denoiser.metrics(noisy[split_idx], clean[split_idx])
        m["split"] = split
        m["num_signals"] = int(len(split_idx))
        den_rows.append(m)
    den_df = pd.DataFrame(den_rows)[["split", "num_signals", "mse_noisy", "mse_glwt", "mse_improvement_pct", "snr_noisy_db", "snr_glwt_db", "snr_gain_db"]]
    den_df.to_csv(out_dir / "metrics" / "denoising_metrics.csv", index=False)
    denoiser.save(out_dir / "models" / "glwt_denoiser_params.json")

    # ---------------- Classification ----------------
    X_signal = structural_node_signals(A)
    lambdas_for_features = np.ones(len(wavelet_bank.scales)) * denoiser.lambda_factor_ * 0.30
    X = wavelet_bank.features(X_signal, lambdas=lambdas_for_features[:len(wavelet_bank.scales)])
    names = feature_names(wavelet_bank.scales)
    train_nodes, val_nodes, test_nodes = stratified_masks(y, train_size=0.50, val_size=0.25, seed=seed)

    # Fit a shallow rule classifier and select max_depth using validation accuracy.
    candidates = []
    for depth in [1, 2, 3, 4]:
        clf = GLWTRuleClassifier(max_depth=depth, min_samples_leaf=1, random_state=seed).fit(X[train_nodes], y[train_nodes], names)
        val_metrics = clf.metrics(X[val_nodes], y[val_nodes])
        candidates.append((val_metrics["accuracy"], depth, clf))
    candidates.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    clf = candidates[0][2]

    class_rows = []
    for split, split_idx in [("train", train_nodes), ("validation", val_nodes), ("test", test_nodes)]:
        m = clf.metrics(X[split_idx], y[split_idx])
        m["split"] = split
        m["num_nodes"] = int(len(split_idx))
        class_rows.append(m)
    cls_df = pd.DataFrame(class_rows)[["split", "num_nodes", "accuracy", "balanced_accuracy", "macro_f1"]]
    cls_df.to_csv(out_dir / "metrics" / "classification_metrics.csv", index=False)
    clf.save_rules(out_dir / "rules" / "karate_rules.txt")
    clf.save_metadata(out_dir / "models" / "karate_rule_classifier.json")

    # Confusion matrix and activation heatmap.
    y_pred_test = clf.predict(X[test_nodes])
    cm = confusion_matrix(y[test_nodes], y_pred_test, labels=[0, 1])
    pd.DataFrame(cm, index=["true_MrHi", "true_Officer"], columns=["pred_MrHi", "pred_Officer"]).to_csv(out_dir / "metrics" / "classification_confusion_matrix_test.csv")

    coeffs = wavelet_bank.transform(X_signal)
    # Activation matrix: node x scale = mean absolute modulated coefficient.
    activations = []
    for k, C in enumerate(coeffs):
        M = np.abs(C)
        activations.append(M.mean(axis=1))
    activation_matrix = np.stack(activations, axis=1)
    activation_matrix = activation_matrix / (activation_matrix.max() + 1e-12)
    pd.DataFrame(activation_matrix, columns=[f"scale_{s}" for s in wavelet_bank.scales]).to_csv(out_dir / "metrics" / "activation_matrix.csv", index_label="node")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(activation_matrix, aspect="auto", interpolation="nearest")
    plt.title("GLWT Activation Heatmap: Karate Club")
    plt.xlabel("Scale index")
    plt.ylabel("Node index")
    plt.xticks(range(len(wavelet_bank.scales)), [str(s) for s in wavelet_bank.scales])
    plt.colorbar(im, label="normalized |coefficient| energy")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "activation_heatmap.png", dpi=200)
    plt.close()

    summary = {
        "dataset": "Zachary Karate Club (NetworkX)",
        "algorithm": "Non-neural Graph Laplacian Wavelet Transform with soft shrinkage and symbolic rule induction",
        "denoising": den_rows,
        "classification": class_rows,
        "selected_rule_depth": int(clf.max_depth),
        "selected_denoising_lambda_factor": float(denoiser.lambda_factor_),
        "selected_scale_weights": denoiser.weights_.tolist(),
        "denoising_scales": denoise_bank.scales.tolist(),
        "denoising_kernel": denoise_bank.kernel,
        "classification_scales": wavelet_bank.scales.tolist(),
        "classification_kernel": wavelet_bank.kernel,
    }
    with open(out_dir / "metrics" / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.out, seed=args.seed)
