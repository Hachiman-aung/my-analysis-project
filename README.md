# Uncertainty Quantification for Out-of-Distribution Detection

Master's thesis project (Trustworthy / Robust ML) comparing uncertainty
quantification (UQ) methods for out-of-distribution (OOD) detection and
reliable confidence estimation, with a focus on **conformal prediction**.

## Research Question

How well do different UQ methods detect out-of-distribution inputs and
produce reliable (calibrated) confidence estimates, and how do their
guarantees hold up under distribution shift?

## Methods Compared

| Method | Description | Relative compute cost |
|---|---|---|
| Softmax confidence | Naive baseline: max softmax probability | 1x (free) |
| MC-Dropout | Multiple stochastic forward passes with dropout active at inference | ~30x inference |
| Deep Ensembles | 3–5 independently trained small models | ~5x training + inference |
| Split Conformal Prediction (APS) | Distribution-free prediction sets with coverage guarantees | ~1x (near-free on top of any classifier) |

## Datasets

- **In-distribution (ID): CIFAR-10** — 10-class, 32x32 RGB natural images.
  Dataset page: https://www.cs.toronto.edu/~kriz/cifar.html
- **Out-of-distribution (OOD): SVHN** (Street View House Numbers) — 32x32 RGB
  digit images, semantically disjoint from CIFAR-10.
  Dataset page: http://ufldl.stanford.edu/housenumbers/
- Optional, for the stress-test section: **CIFAR-10-C** (corrupted CIFAR-10)
  can be substituted for the built-in simulated Gaussian noise corruption.
  Dataset page: https://zenodo.org/record/2535967

Both CIFAR-10 and SVHN are downloaded automatically via `torchvision.datasets`
the first time the notebook is run — no manual download required.

## Project Structure

```
.
├── main.ipynb        # Full experimental pipeline (see notebook sections below)
├── requirements.txt  # Python dependencies
├── README.md          # This file
└── data/               # Auto-created on first run; holds downloaded datasets (gitignored)
```

## Notebook Sections (`main.ipynb`)

1. **Setup & Environment** — imports, seeding, device selection, central config block
2. **Data** — load CIFAR-10 (ID) and SVHN (OOD), train/calibration/eval split, quick EDA
3. **Base Model** — small CNN classifier, trained once and reused by all UQ methods
4. **Uncertainty Quantification Methods**
   - 4a. Softmax confidence
   - 4b. MC-Dropout
   - 4c. Deep Ensembles
   - 4d. Split Conformal Prediction (Adaptive Prediction Sets)
5. **Evaluation Metrics** — calibration (ECE, reliability diagrams), OOD detection (AUROC/AUPR), conformal coverage & set size
6. **Comparative Analysis** — summary table and plots across all methods
7. **Stress Tests / Robustness** — performance under increasing simulated distribution shift
8. **Discussion / Error Analysis** — qualitative look at failure cases
9. **Conclusion Scaffold** — bullet points to convert into the thesis conclusion chapter

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook main.ipynb
```

A GPU is not required — the base CNN is deliberately small and the full
pipeline (base model + 5 ensemble members + MC-Dropout + conformal
calibration) is designed to run on CPU within a reasonable time, in line with
limited-compute, public-dataset-only constraints.

## Notes for Thesis Writing

- Everything downstream of Section 3 (the base model) reuses the same
  trained weights, so all UQ comparisons are apples-to-apples.
- The `CONFIG` dictionary in Section 1 centralizes all hyperparameters —
  rerun experiments by editing one cell rather than hunting through the notebook.
- Section 7 uses simulated Gaussian noise for the stress test to keep the
  notebook self-contained. Swap in real CIFAR-10-C tensors for a stronger,
  more citable robustness result if compute/storage allows.

## References / Further Reading

- Angelopoulos & Bates, *"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"* (2021)
- Lakshminarayanan et al., *"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"* (NeurIPS 2017)
- Gal & Ghahramani, *"Dropout as a Bayesian Approximation"* (ICML 2016)
- Guo et al., *"On Calibration of Modern Neural Networks"* (ICML 2017)
