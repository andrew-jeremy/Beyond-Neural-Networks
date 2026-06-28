# GLWT Logic Machines
## Andrew Kiruluta, UC Berkeley, CA. Jan 2026.
This repository reproduces the neural-network-free algorithm proposed in the attached paper, **Beyond Neural Networks: Symbolic Reasoning over Wavelet Logic Graph Signals**. It implements Graph Laplacian Wavelet Transform (GLWT) spectral filtering, soft-threshold modulation, convex scale aggregation, and symbolic rule induction over wavelet activations.

## What is implemented

The core pipeline is:

```text
Graph -> Laplacian -> GLWT filter bank -> spectral coefficients
      -> soft shrinkage / gain / phase modulation
      -> scale reconstructions -> convex scale weighting
      -> symbolic rule classifier over GLWT activations
```

No neural-network layers are used. There are no convolutions, attention blocks, MLPs, or learned embeddings. The classifier is a shallow rule-induction tree over deterministic GLWT features.

## Repository structure

```text
src/glwt_logic/
  glwt.py            # GLWT filter bank, shrinkage, spectral features
  denoise.py         # non-neural GLWT denoiser and training loop
  rules.py           # symbolic rule classifier over GLWT activations
  graph_utils.py     # graph loading and Laplacian utilities
  synthetic.py       # smooth graph signal generator
  chebyshev.py       # Chebyshev GLWT approximation utility
experiments/
  run_benchmarks.py  # reproduces benchmark metrics
outputs/
  metrics/           # generated train/validation/test metrics
  figures/           # activation heatmap
  rules/             # exported symbolic rules
outputs/models/      # learned thresholds/weights/rule metadata
```

## Benchmark dataset

The included benchmark uses **Zachary's Karate Club**, a standard graph benchmark available through NetworkX. The experiment reports train/validation/test metrics for:

1. **Synthetic graph signal denoising** on the Karate graph.
2. **Node classification** using symbolic rules over GLWT features.

The script is deterministic by default with seed `7`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproduce metrics

```bash
python experiments/run_benchmarks.py --out outputs --seed 7
```

Generated files:

```text
outputs/metrics/denoising_metrics.csv
outputs/metrics/classification_metrics.csv
outputs/metrics/metrics_summary.json
outputs/metrics/activation_matrix.csv
outputs/metrics/classification_confusion_matrix_test.csv
outputs/figures/activation_heatmap.png
outputs/rules/karate_rules.txt
outputs/models/glwt_denoiser_params.json
outputs/models/karate_rule_classifier.json
```

## Current generated metrics

The repository already includes generated metrics from the benchmark run. See:

```bash
cat outputs/metrics/metrics_summary.json
```

## Notes on larger graph datasets

For Cora/Citeseer, use the same `GLWTFilterBank` API with a loaded adjacency matrix and feature matrix. For large graphs, replace exact eigendecomposition with the Chebyshev approximation in `src/glwt_logic/chebyshev.py`.

## Tests

```bash
pytest -q
```

## Metrics from the included run

Classification on Zachary Karate Club using symbolic GLWT rule induction:

| Split | Nodes | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Train | 17 | 1.000 | 1.000 | 1.000 |
| Validation | 8 | 0.875 | 0.875 | 0.873 |
| Test | 9 | 1.000 | 1.000 | 1.000 |

Synthetic smooth graph-signal denoising on the Karate graph:

| Split | Signals | Noisy MSE | GLWT MSE | MSE improvement | SNR gain |
|---|---:|---:|---:|---:|---:|
| Train | 144 | 0.0922 | 0.0494 | 46.41% | 2.71 dB |
| Validation | 48 | 0.0879 | 0.0491 | 44.11% | 2.53 dB |
| Test | 48 | 0.0888 | 0.0509 | 42.72% | 2.42 dB |

The learned symbolic rule in this run is:

```text
|--- mean_phi_s10p0 <= -0.00
|   |--- class: 1
|--- mean_phi_s10p0 >  -0.00
|   |--- class: 0
```

Because Karate Club is a small graph, these classification values should be treated as a reproducibility smoke test rather than a claim of broad state-of-the-art performance.

This repo is based on the following citation. Please cite this reference when using this repo in your work: https://www.researchgate.net/publication/399493876_Integrating_Quantum_Circuit_Reasoning_into_Large-Scale_Neurosymbolic_AI_Architectures?channel=doi&linkId=695d10c09aa6b4649dc89c42&showFulltext=true