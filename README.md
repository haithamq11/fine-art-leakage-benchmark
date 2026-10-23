# Fine-Art Leakage Benchmark

Reproducibility repository for:

**Reproducibility code for Near-Duplicate Leakage in Fine-Art Classification: A Reproducible Stacked CNN Ensemble with Perceptual Hash Deduplication**

Haitham Qutaiba Ghadhban  
Computer Engineering Department, University of Diyala, Iraq  
haithamqutaiba@uodiyala.edu.iq

---

## Overview

This repository contains all code and data files required to reproduce the results reported in the paper. Saved per-image predictions (`.npz`) for every backbone, baseline and seed are included in `predictions/` (round 2); `regenerate_tables.py` recomputes every table and test from them. The paper makes two core contributions:

1. A pHash-based near-duplicate detection protocol that removes 595 leaking images before any data split is applied
2. A leakage-free five-fold OOF stacking framework that explicitly separates early stopping from meta-classifier training

A controlled leakage isolation experiment (`notebooks/07_leakage_isolation.py`) trains Xception with duplicates retained under **exactly** the same five-fold OOF protocol as the deduplicated run, so the paired per-seed difference isolates the effect of near-duplicate leakage.

---

## Repository Structure

```
fine-art-leakage-benchmark/
│
├── data/
│   ├── retained_images.txt          # 3,793 filenames retained after deduplication
│   ├── duplicate_groups.csv         # 547 duplicate groups with Hamming distances
│   ├── phash_values.csv             # 64-bit pHash for every image
│   ├── split_assignments.csv        # Train/val/test assignments for all 5 seeds
│   ├── dedup_audit_10bit.csv        # Visual verdict for all 60 pairs at exactly 10 bits (17 false positives)
│   ├── leakage_isolation_5seeds.csv # Leakage isolation run (Xception with duplicates, 5 seeds)
│   ├── leakage_isolation_5seeds_summary.txt
│   └── ensartnet_results.csv        # EnsArtNet reproduction, 5 seeds (Kaggle)
│
├── notebooks/
│   ├── 01_train_main.py             # Main training: 4 backbones + OOF stacking
│   ├── 02_evaluate.py               # Evaluation: McNemar, ablation, per-class
│   ├── 03_baseline_mobilenet.py     # MobileNetV3Large baseline [20,21]
│   ├── 04_baseline_swintiny.py      # Swin-Tiny baseline [22]
│   ├── 05_baseline_convnext.py      # ConvNeXtTiny baseline [30]
│   ├── 06_baseline_ensartnet.py     # EnsArtNet reproduction [18]
│   ├── 07_leakage_isolation.py      # Xception without deduplication (same 5-fold OOF protocol)
│   ├── 08_leakage_fixed_testset.py  # fixed-test-set leakage design with size control (round 2)
│   ├── 09_group_aware_split.py      # duplicate-group-aware split baseline (round 2)
│   └── 10_export_fold_indices.py    # exports data/fold_assignments.csv (round 2)
│
├── predictions/                     # per-image OOF/test probabilities, all backbones, baselines and seeds
├── regenerate_tables.py             # recomputes all tables, CIs and McNemar/Holm tests from predictions/
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
- Threshold: Hamming distance ≤ 10 between any two images
- Grouping: transitive closure (union-find). Two images that are each within 10 bits of a
  third image are placed in the same group even if they are more than 10 bits apart from
  each other. As a result, 8 removed images lie at distance 12–18 from their group
  representative (see `hamming_distance` in `data/duplicate_groups.csv`).
- Groups detected: 547 (543 same-artist, 4 cross-artist)
- Images removed: 595
- Retention rule: First member alphabetically by filepath is retained
- Manual audit: 50 stratified pairs in the 0–10 range confirmed — zero false positives

---

## Deduplication audit at the 10-bit boundary

The pHash used here has a fixed number of set bits, so every pairwise Hamming distance is even and there are no
9-bit pairs. Exactly 60 image pairs lie at the 10-bit threshold. All of them were inspected visually
(`data/dedup_audit_10bit.csv`): 43 are genuine digitisation variants of the same painting and 17 are visually
distinct paintings, i.e. false positives at the boundary. The 17 false-positive links (which include all four
cross-artist links: Sisley/Degas, Goya/Rembrandt, Titian/Degas, Picasso/Rembrandt) cause 17 distinct paintings
(0.4% of the 4,388 images) to be removed that would otherwise have been retained. Removing a distinct painting
cannot create leakage, so the retained set of 3,793 images is conservative; the threshold and the retained set
were kept unchanged so that every reported result stays reproducible. Of the 8 removed images that lie 12-18 bits
from their group representative, 2 are genuine variants of the representative, 3 are genuine variants of another
group member, and 3 are distinct paintings joined through a false-positive 10-bit link.


## Round-2 additions (2026-10-11)

New scripts (all read the released split and group files; none changes any reported result):

| Script | Purpose | Runtime |
|---|---|---|
| `notebooks/08_leakage_fixed_testset.py` | Leakage estimate with a FIXED deduplicated test set: arm A clean training, arm B training + matched duplicates of training images only, arm C training + the same number of class-matched extra images (size control). Arm B − arm C isolates duplicates in training with test identity, size and class composition held constant. | ≈12 h per seed (T4 ×2); one seed per Kaggle version, resumable |
| `notebooks/09_group_aware_split.py` | Duplicate-GROUP-aware split that keeps all 4,388 images and assigns every group as a unit to train/val/test and to the inner folds; Xception under the identical OOF protocol. Benchmarks the deletion policy against a leakage-controlled split that retains valid variants. | ≈4 h per seed |
| `notebooks/10_export_fold_indices.py` | Regenerates `data/fold_assignments.csv`: OOF fold and inner-validation membership of every training image for every seed, from the same seeded procedure as `01_train_main.py`. | seconds |
| `regenerate_tables.py` | Recomputes Tables 6–9 and 11, every paired CI and every McNemar test with Holm adjustment from `predictions/`. | seconds |

New data files: `fold_assignments.csv`, `fn_audit_verdicts.csv` (338 inspected pairs above the threshold: 35 residual near-duplicates,
7 repetitions, 167 distinct), `residual_pairs_cross_split.csv` (per seed: residual pairs across train/test, train/val, …),
`fp_pairs_to_removed_images.csv` (how the 17 false-positive 10-bit edges map to removed images: 14 singletons, 6 separated from
their representative), `mcnemar_family_holm.csv` (25 tests, Holm within family and across all), `retained_images_v2.txt`
(corrected list, 3,775 images; NOT used for any reported result).

`predictions/` holds the per-image OOF and test softmax outputs of every backbone, MobileNetV3Large, ConvNeXtTiny, Swin-Tiny
and the duplicate-retained Xception run for all five seeds (row order = `data/split_assignments.csv` / `fold_assignments.csv`).
The EnsArtNet re-implementation was trained on Kaggle and only its per-seed accuracies were kept (`data/ensartnet_results.csv`).

Deduplication error rates (round-2 audit): pairwise precision ≈ 0.97 (623 genuine of 640 accepted pairs at ≤ 10 bits), recall
≈ 0.95 (623 of 658 identified genuine pairs); 35 residual pairs remain among the 3,793 retained images, of which 2–7 per seed
straddle train and test (0.4–1.2 % of the 569 test images). The protocol is therefore described as leakage-controlled.

## Reproducibility note: splits are taken from `data/split_assignments.csv`

`os.listdir()` returns files in a platform-dependent order, and `train_test_split` depends on the row order of the
dataframe it receives. Recomputing the 70/15/15 split on another machine therefore silently produces **different**
train/validation/test images with the same class counts (on Windows only about 12-16% of each test set coincided
with the paper's). All scripts now look for `data/split_assignments.csv` (searched in `data/`, `../data/`,
`/kaggle/working/` and any attached Kaggle dataset) and take the exact per-seed membership and order from it; if the
file is not found they fall back to a filename-sorted dataframe and print a warning. The non-deduplicated experiment
(`07_leakage_isolation.py`) has no shipped split file and always uses the filename-sorted order.

Windows users: extracting the Kaggle archive on Windows re-encodes the two spellings of the umlaut in the
`Albrecht_Dürer_*.jpg` filenames, so the names in `split_assignments.csv` may not match the local filenames literally.
The paired files are byte-identical images; map the names before matching (NFD `ü` -> `u╠ê`, and
`u╠ê` -> `uΓòá├¬`) or run on Linux/Kaggle.

## Reproducing the Splits

Do **not** recompute the split from a directory listing (see the note above). Read the shipped assignments:

```python
import pandas as pd
split = pd.read_csv('data/split_assignments.csv')     # columns: filename, artist, seed, split
for seed in [42, 0, 1, 7, 123]:
    s = split[split.seed == seed]
    train, val, test = (s[s.split == p].reset_index(drop=True) for p in ('train', 'val', 'test'))
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

The row order inside each split is the order in which the saved prediction files (`*_oof.npz`, `*_test.npz`) are stored.
For reference, the splits were originally generated with `train_test_split(..., stratify=artist, random_state=seed)`
(70/15/15) applied to the retained images **in the order of `data/phash_values.csv`** (the Kaggle directory order);
applying the same call to an alphabetically ordered list gives different images.

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
# Downloads the public timm ImageNet-1k checkpoint; falls back to a local
# swin_tiny_patch4_window7_224.pth only when offline
python notebooks/04_baseline_swintiny.py

# ConvNeXtTiny [30] — ~3 hours on T4 GPU
python notebooks/05_baseline_convnext.py

# EnsArtNet reproduction [18] — ~4-5 hours on T4 GPU
# Uses the same augmentation policy and (with FINE_TUNE = True) the same
# unfreezing policy as the proposed framework; see the header of the script
python notebooks/06_baseline_ensartnet.py
```

Expected baseline results (five seeds, exact splits from `data/split_assignments.csv`):

| Method | Mean Acc | Std | Paired gain vs stacked ensemble (89.63%) |
|---|---|---|---|
| MobileNetV3Large | 79.47% | ±1.48% | −10.16pp |
| ConvNeXtTiny | 90.05% | ±0.51% | +0.42pp, 95% CI [−1.26, +2.10], not significant |
| Swin-Tiny | 93.74% | ±0.66% | +4.11pp, 95% CI [+2.09, +6.14], McNemar p < 0.05 in all 5 seeds |
| EnsArtNet reproduced (published configuration, `FINE_TUNE = True`) | 62.53% | ±2.81% | −27.10pp; per-seed 61.86 / 65.91 / 60.11 / 59.75 / 65.03 (`data/ensartnet_results.csv`) |

ConvNeXtTiny and Swin-Tiny were measured with the corrected augmentation policy on an RTX 5070 (PyTorch 2.7,
timm 1.0.29, scikit-learn 1.8.0); the EnsArtNet reproduction was run on Kaggle (GPU T4 x2, TensorFlow 2.x) with the
published Stage 1 learning rate of 1e-5, which is one order of magnitude below the 1e-4 used by `01_train_main.py`
and is the most plausible reason for its low accuracy and large seed-to-seed spread. The previously reported values (Swin-Tiny 71.25% ±3.02%, ConvNeXtTiny 79.82% ±1.97%,
EnsArtNet 68.58% ±10.15%) were produced by earlier versions of these scripts that did not run under identical
conditions to the proposed framework and must not be reported.

---

## Reproducing the Leakage Isolation Experiment

```bash
# Xception WITHOUT deduplication, same 5-fold OOF protocol — ~4 hours on T4 GPU
# Run AFTER 01_train_main.py so the per-seed deduplicated Xception accuracies are
# read from /kaggle/working/predictions_v3 (otherwise Table 5 values are used as fallback)
python notebooks/07_leakage_isolation.py
```

The script prints, per seed, the accuracy with duplicates retained, the accuracy with
duplicates removed, their paired difference, and a 95% t-interval for the mean difference.

| Protocol | Mean Acc | Std |
|---|---|---|
| Xception WITH duplicates (same 5-fold OOF protocol, 4,388 images) | 89.04% | ±0.65% |
| Xception WITHOUT duplicates (deduplicated, 3,793 images) | 86.57% | ±1.48% |
| **Isolated leakage effect (paired, 5 seeds)** | **+2.47pp** | 95% CI [+0.28, +4.66]pp |

Measured on Kaggle (GPU T4 x2, TensorFlow 2.19) in notebook `run-07-leakage`, Version 5. Per-seed effects: +3.67, +1.47,
+0.55, +4.89, +1.78pp. The interval excludes zero. The previously reported 90.17% ±1.04% (3.60pp effect) came from a
single model trained on the full 70% split with no fold averaging, which confounded duplicate retention with training-set
size, early-stopping set and fold-ensembling, and must not be reported.

---|---|---|
| Xception WITH duplicates | to be re-measured with the corrected script | |
| Xception WITHOUT duplicates | 86.57% | ±1.48% |
| **Isolated leakage effect** | to be re-measured | |

The previously reported 90.17% ±1.04% (3.60pp effect) came from a single model trained on
the full 70% split with early stopping on the 15% validation split and no fold averaging.
That protocol differed from the deduplicated run in training-set size, early-stopping set
and fold-ensembling, so the 3.60pp figure was not an isolated leakage effect and must not
be reported.

---

## OOF Protocol — Leakage-Free Guarantee

The key contribution is the leakage-free OOF protocol. Within each fold:

```
train split (70% of the deduplicated data, 2,655 images)
├── fold_train (80% of train, 4 of 5 folds)
│   ├── fold_train_inner  →  gradient updates ONLY (85% of fold_train ≈ 68% of train)
│   └── fold_inner_val    →  early stopping ONLY  (15% of fold_train ≈ 12% of train)
└── fold_oof (20% of train, the held-out fold)
                          →  OOF predictions ONLY — never seen during training or early stopping
```

This triple partition ensures meta-classifier training data is free from any validation-set influence. The `fold_oof` fold is **never** used for early stopping.

The 15% validation split produced by the 70/15/15 partition (569 images per seed) is held out
and **not used** by the main pipeline; early stopping relies exclusively on `fold_inner_val`.

---

## Environment

All experiments run on Kaggle GPU environment:

- GPU: NVIDIA Tesla T4 (16GB VRAM)
- Python: 3.12
- TensorFlow: 2.19 (Keras 3)
- PyTorch: 2.10 / torchvision 0.25 (for Swin-Tiny and ConvNeXt-Tiny)
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

### Update 2026-10-22
The round-2 experiments have been run for all five seeds: `data/leakage_fixed_testset.csv` + `predictions/fixed_testset_seed{42,0,1,7,123}.npz` (script 08) and `data/group_aware_results.csv`, `data/group_aware_split_assignments.csv` + `predictions/group_aware_seed{42,0,1,7,123}_test.npz` (script 09). `regenerate_tables.py` now also writes `table13_fixed_testset.csv` and `table14_group_aware.csv` (Tables 13-14 of the manuscript).
