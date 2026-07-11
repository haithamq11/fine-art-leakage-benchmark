# Fine-Art Leakage Benchmark

Reproducibility repository for:

> **Near-Duplicate Leakage in Fine-Art Classification: A Reproducible Stacked CNN Baseline with Perceptual Hash Deduplication**
> Haitham Qutaiba Ghadhban, Omar Hatem Zaidan
> University of Diyala, Iraq

## Repository Contents

| File | Description |
|------|-------------|
| `retained_images.txt` | List of 3,793 image filenames retained after pHash deduplication |
| `duplicate_groups.csv` | All 547 duplicate groups with filenames, Hamming distances, and retention decisions |
| `phash_values.csv` | 64-bit pHash value for every image in the original dataset |
| `split_assignments.csv` | Train/val/test assignments for all 3,793 images across 5 seeds (42, 0, 1, 7, 123) |
| `mcnemar_evaluation.py` | Script to reproduce McNemar statistical tests from saved predictions |
| `requirements.txt` | Python environment specification |

## Dataset

The experiments use the **Best Artworks of All Time** dataset:
- Source: https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time
- Original images: 4,388
- After pHash deduplication (Hamming ≤ 10): 3,793
- Classes: 10 artists (Vincent van Gogh, Edgar Degas, Pablo Picasso, Pierre-Auguste Renoir, Albrecht Dürer, Paul Gauguin, Francisco Goya, Alfred Sisley, Rembrandt, Titian)

## Deduplication Protocol

- Algorithm: Perceptual hashing (pHash) with 64-bit hash
- Threshold: Hamming distance ≤ 10
- Groups detected: 547 (543 same-artist, 4 cross-artist)
- Images removed: 595
- Retention rule: First member alphabetically by filepath is retained

## Reproducing the Splits

```python
from sklearn.model_selection import train_test_split
import pandas as pd

retained = pd.read_csv('retained_images.txt', comment='#',
                       header=None, names=['filename','artist'])

for seed in [42, 0, 1, 7, 123]:
    train_val, test = train_test_split(
        retained, test_size=0.15, stratify=retained['artist'], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.15/0.85,
        stratify=train_val['artist'], random_state=seed)
    print(f"Seed {seed}: train={len(train)}, val={len(val)}, test={len(test)}")
```

## Contact

Haitham Qutaiba Ghadhban
haithamqutaiba@uodiyala.edu.iq
Computer Engineering Department, University of Diyala, Iraq
