# Fine-Art Leakage Benchmark

Reproducibility repository for:

**Near-Duplicate Leakage in Fine-Art Classification: A Leakage-Controlled Evaluation Protocol and Multi-Seed Benchmark with Perceptual Hash Deduplication**

Haitham Qutaiba Ghadhban  
Computer Engineering Department, University of Diyala, Iraq  
haithamqutaiba@uodiyala.edu.iq

---

## Overview

This repository contains all code, data files, and saved predictions required to fully reproduce the results reported in the paper. Saved per-image predictions (`.npz`) for every backbone, baseline and seed are included in `predictions/`; `regenerate_tables.py` recomputes every table and statistical test from them.

The paper makes two core contributions:

1. A pHash-based near-duplicate detection protocol with a documented precision/recall audit
2. A leakage-controlled five-fold OOF stacking framework that explicitly separates early stopping from meta-classifier training

A controlled fixed-test-set leakage isolation experiment (`notebooks/08_leakage_fixed_testset.py`) holds the deduplicated test set constant and varies only whether training contains duplicates, isolating a paired per-seed accuracy change of −0.14pp (95% CI [−1.58, +1.30]pp).

---

## Repository Structure

```
fine-art-leakage-benchmark/
│
├── data/
│   ├── retained_images.txt              # 3,793 filenames retained after deduplication
│   ├── retained_images_v2.txt           # 3,775 filenames after corrected audit (future work)
│   ├── duplicate_groups.csv             # 547 duplicate groups with Hamming distances
│   ├── phash_values.csv                 # 64-bit pHash for every image
│   ├── split_assignments.csv            # Train/val/test assignments for all 5 seeds
│   ├── fold_assignments.csv             # OOF fold and inner-val membership for all 5 seeds
│   ├── dedup_audit_10bit.csv            # All pairs at exactly 10-bit boundary with verdicts
│   ├── fn_audit_verdicts.csv            # False-negative audit verdicts (338 pairs)
│   ├── fp_pairs_to_removed_images.csv   # False-positive pair to removed image mapping
│   ├── residual_pairs_cross_split.csv   # 35 residual pairs crossing train-test boundary
│   ├── group_aware_split_assignments.csv # Split assignments for group-aware policy
│   ├── group_aware_results.csv          # Per-seed results for group-aware split
│   ├── leakage_fixed_testset.csv        # Per-seed fixed-test-set leakage results
│   └── mcnemar_family_holm.csv          # Holm-adjusted McNemar p-values
│
├── notebooks/
│   ├── 01_train_main.py                 # Main training: 4 backbones + OOF stacking
│   ├── 02_evaluate.py                   # Evaluation: McNemar, ablation, per-class
│   ├── 03_baseline_mobilenet.py         # MobileNetV3Large baseline
│   ├── 04_baseline_swintiny.py          # Swin-Tiny baseline
│   ├── 05_baseline_convnext.py          # ConvNeXtTiny baseline
│   ├── 06_baseline_ensartnet.py         # EnsArtNet reproduction
│   ├── 07_leakage_isolation.py          # Xception leakage isolation (OOF protocol)
│   ├── 08_leakage_fixed_testset.py      # Fixed-test-set leakage experiment
│   ├── 09_group_aware_split.py          # Group-aware split experiment
│   └── 10_export_fold_indices.py        # Export OOF fold and inner-val indices
│
├── predictions/                         # Per-image softmax predictions for all models and seeds
│
├── regenerated/                         # Tables regenerated from saved predictions
│
├── regenerate_tables.py                 # Regenerate all tables and CIs from predictions
├── localize.py                          # Cross-platform filename normalisation
├── mcnemar_evaluation.py                # Standalone McNemar test script
├── requirements.txt                     # Python environment specification
└── README.md                            # This file
```

---

## Dataset

The experiments use the **Best Artworks of All Time** dataset:

- Source: https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time
- Original images: 4,388
- After pHash deduplication (Hamming ≤ 10): **3,793**
- Classes: 10 artists

| Artist | Images (after dedup) |
|---|---|
| Vincent van Gogh | 800 |
| Edgar Degas | 574 |
| Pablo Picasso | 415 |
| Pierre-Auguste Renoir | 333 |
| Albrecht Dürer | 324 |
| Paul Gauguin | 310 |
| Francisco Goya | 286 |
| Alfred Sisley | 257 |
| Rembrandt | 248 |
| Titian | 246 |

---

## Deduplication Protocol

- Algorithm: Perceptual hashing (pHash) with 64-bit hash via DCT
- Threshold: Hamming distance ≤ 10
- Groups detected: 547 (530 same-artist, 17 cross-split false positives)
- Images removed: 595
- Retention rule: First member alphabetically by filepath is retained
- Audit: estimated precision 0.97, recall ~0.95; 35 residual pairs remain above threshold
- All audit verdicts: `data/dedup_audit_10bit.csv`, `data/fn_audit_verdicts.csv`

---

## Key Results

| Model | Mean Acc | Std |
|---|---|---|
| ResNet50 | 86.33% | ±2.15% |
| EfficientNetV2B0 | 76.52% | ±0.76% |
| InceptionV3 | 83.20% | ±1.34% |
| Xception | 86.57% | ±1.48% |
| EnsArtNet reproduced | 62.53% | ±2.81% |
| MobileNetV3Large | 79.47% | ±1.48% |
| ConvNeXtTiny | 90.05% | ±0.51% |
| **Swin-Tiny** | **93.74%** | **±0.66%** |
| Soft voting (4 CNN) | 88.65% | ±1.58% |
| Two-backbone (R50+Xcp) | 89.84% | ±1.76% |
| **Stacked ensemble** | **89.63%** | **±1.85%** |

---

## Reproducing the Results

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run main training
```bash
# On Kaggle with GPU — approximately 8-10 hours
python notebooks/01_train_main.py
```

### Step 3 — Regenerate all tables and tests
```bash
python regenerate_tables.py
```

---

## Leakage Isolation Experiments

### Fixed-test-set experiment (Table 13)
```bash
python notebooks/08_leakage_fixed_testset.py
```
Result: adding duplicates to training only changes accuracy by −0.14pp (95% CI [−1.58, +1.30]pp) — near-duplicates in training do not measurably raise accuracy.

### Group-aware split experiment (Table 14)
```bash
python notebooks/09_group_aware_split.py
```
Result: deletion policy vs group-aware split = +0.42pp (95% CI [−0.42, +1.26]pp) — no measurable difference.

---

## OOF Protocol

Within each fold:

```
fold_train
├── fold_train_inner  →  gradient updates (80% of fold_train)
├── fold_inner_val    →  early stopping ONLY (15% of fold_train)
│
fold_oof              →  OOF predictions ONLY — never seen during training
```

Fold assignments for all seeds: `data/fold_assignments.csv`

---

## Statistical Validation

All McNemar tests use Holm step-down adjustment within three prespecified families.  
Full results: `data/mcnemar_family_holm.csv`

95% CI for stacking gain over soft voting: **[0.53, 1.44]pp**

---

## Environment

- GPU: NVIDIA Tesla T4 (16GB VRAM) — Kaggle
- Python: 3.12
- TensorFlow: 2.19
- PyTorch: 2.10
- scikit-learn: 1.6.1
- timm: 1.0.26
- imagehash: 4.3.1

---

## Citation

```
Ghadhban, H.Q. (2026). Near-Duplicate Leakage in Fine-Art Classification:
A Leakage-Controlled Evaluation Protocol and Multi-Seed Benchmark with
Perceptual Hash Deduplication. International Journal of Intelligent
Engineering and Systems.
```

---

## Contact

Haitham Qutaiba Ghadhban  
haithamqutaiba@uodiyala.edu.iq  
Computer Engineering Department, University of Diyala, Iraq
