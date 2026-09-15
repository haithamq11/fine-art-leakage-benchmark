# Fine-Art Leakage Benchmark

Reproducibility repository for:

**Reproducibility code for Near-Duplicate Leakage in Fine-Art Classification: A Reproducible Stacked CNN Ensemble with Perceptual Hash Deduplication**

Haitham Qutaiba Ghadhban  
Computer Engineering Department, University of Diyala, Iraq  
haithamqutaiba@uodiyala.edu.iq

---

## Overview

This repository contains all code, data files, and saved predictions required to fully reproduce the results reported in the paper. The paper makes two core contributions:

1. A pHash-based near-duplicate detection protocol that removes 595 leaking images before any data split is applied
2. A leakage-free five-fold OOF stacking framework that explicitly separates early stopping from meta-classifier training

A controlled leakage isolation experiment demonstrates a **3.60pp accuracy inflation** attributable to near-duplicate leakage when holding all other variables constant.

---

## Repository Structure

```
fine-art-leakage-benchmark/
│
├── data/
│   ├── retained_images.txt          # 3,793 filenames retained after deduplication
│   ├── duplicate_groups.csv         # 547 duplicate groups with Hamming distances
│   ├── phash_values.csv             # 64-bit pHash for every image
│   └── split_assignments.csv        # Train/val/test assignments for all 5 seeds
│
├── notebooks/
│   ├── 01_train_main.py             # Main training: 4 backbones + OOF stacking
│   ├── 02_evaluate.py               # Evaluation: McNemar, ablation, per-class
│   ├── 03_baseline_mobilenet.py     # MobileNetV3Large baseline [20,21]
│   ├── 04_baseline_swintiny.py      # Swin-Tiny baseline [22]
│   ├── 05_baseline_convnext.py      # ConvNeXtTiny baseline [28]
│   ├── 06_baseline_ensartnet.py     # EnsArtNet reproduction [18]
│   └── 07_leakage_isolation.py      # Xception without deduplication
│
├── mcnemar_evaluation.py            # Standalone McNemar test script
├── requirements.txt                 # Python environment specification
└── README.md                        # This file
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
- Groups detected: 547 (543 same-artist, 4 cross-artist)
- Images removed: 595
- Retention rule: First member alphabetically by filepath is retained
- Manual audit: 50 stratified pairs confirmed — zero false positives

---

## Reproducing the Splits

```python
from sklearn.model_selection import train_test_split
import pandas as pd

retained = pd.read_csv('data/retained_images.txt',
                       comment='#', header=None,
                       names=['filename', 'artist'])

for seed in [42, 0, 1, 7, 123]:
    train_val, test = train_test_split(
        retained, test_size=0.15,
        stratify=retained['artist'], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.15/0.85,
        stratify=train_val['artist'], random_state=seed)
    print(f"Seed {seed}: train={len(train)}, val={len(val)}, test={len(test)}")
```

Expected output:
```
Seed 42:  train=2655, val=569, test=569
Seed 0:   train=2655, val=569, test=569
Seed 1:   train=2655, val=569, test=569
Seed 7:   train=2655, val=569, test=569
Seed 123: train=2655, val=569, test=569
```

---

## Reproducing the Main Results

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run main training (4 backbones + OOF stacking)
```bash
# On Kaggle with GPU — approximately 8-10 hours
python notebooks/01_train_main.py
```

### Step 3 — Run evaluation
```bash
python notebooks/02_evaluate.py
```

Expected main results:

| Model | Mean Acc | Std |
|---|---|---|
| ResNet50 | 86.33% | ±2.15% |
| EfficientNetV2B0 | 76.52% | ±0.76% |
| InceptionV3 | 83.20% | ±1.34% |
| Xception | 86.57% | ±1.48% |
| Soft voting | 88.65% | ±1.58% |
| **Stacked ensemble** | **89.63%** | **±1.85%** |

---

## Reproducing the Baselines

```bash
# MobileNetV3Large [20,21] — ~3 hours on T4 GPU
python notebooks/03_baseline_mobilenet.py

# Swin-Tiny [22] — ~4 hours on T4 GPU
# Requires: swin_tiny_patch4_window7_224.pth weights
python notebooks/04_baseline_swintiny.py

# ConvNeXtTiny [28] — ~3 hours on T4 GPU
python notebooks/05_baseline_convnext.py

# EnsArtNet reproduction [18] — ~4 hours on T4 GPU
python notebooks/06_baseline_ensartnet.py
```

Expected baseline results:

| Method | Mean Acc | Std |
|---|---|---|
| Swin-Tiny | 71.25% | ±3.02% |
| MobileNetV3Large | 79.47% | ±1.48% |
| ConvNeXtTiny | 79.82% | ±1.97% |
| EnsArtNet reproduced | 68.58% | ±10.15% |

---

## Reproducing the Leakage Isolation Experiment

```bash
# Xception WITHOUT deduplication — ~2 hours on T4 GPU
python notebooks/07_leakage_isolation.py
```

Expected result:

| Protocol | Mean Acc | Std |
|---|---|---|
| Xception WITH duplicates | 90.17% | ±1.04% |
| Xception WITHOUT duplicates | 86.57% | ±1.48% |
| **Isolated leakage effect** | **3.60pp** | |

---

## OOF Protocol — Leakage-Free Guarantee

The key contribution is the leakage-free OOF protocol. Within each fold:

```
fold_train
├── fold_train_inner  →  gradient updates (80%)
├── fold_inner_val    →  early stopping ONLY (15% of fold_train)
│
fold_oof              →  OOF predictions ONLY — never seen during training
```

This triple partition ensures meta-classifier training data is free from any validation-set influence. The `fold_oof` fold is **never** used for early stopping.

---

## Environment

All experiments run on Kaggle GPU environment:

- GPU: NVIDIA Tesla T4 (16GB VRAM)
- Python: 3.12
- TensorFlow: 2.19
- PyTorch: 2.10 (for Swin-Tiny and ConvNeXt-Tiny)
- scikit-learn: 1.6.1
- timm: 1.0.26
- imagehash: 4.3.1

---

## Statistical Validation

McNemar's test results (ensemble vs ResNet50):

| Seed | p-value | Significant |
|---|---|---|
| 42 | 0.003378 | Yes |
| 0 | 0.002602 | Yes |
| 1 | 0.064909 | No |
| 7 | 0.002459 | Yes |
| 123 | 0.015347 | Yes |

95% CI for stacking gain over soft voting: **[0.53, 1.44]pp**

---

## Citation

If you use this code or data in your research, please cite:

```
Ghadhban, H.Q. (2026). Near-Duplicate Leakage in Fine-Art Classification:
A Reproducible Stacked CNN Ensemble with Perceptual Hash Deduplication.
```

---

## Contact

Haitham Qutaiba Ghadhban  
haithamqutaiba@uodiyala.edu.iq  
Computer Engineering Department, University of Diyala, Iraq
