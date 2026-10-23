# 10_export_fold_indices.py
# Round-2 reproducibility item (reviewer 1, point 5): regenerate and export, for every seed, the outer split,
# the 5 OOF folds and the inner-validation subset of every fold, exactly as 01_train_main.py derives them
# (StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) on the training split, then a stratified 15%
# inner validation subset with random_state = seed + fold). No training is performed; runs in seconds anywhere.
# Output: data/fold_assignments.csv with columns filename, artist, seed, split, fold, role
#   role = "test" | "val" (outer validation, unused) | "fold_oof" | "fold_train_inner" | "fold_inner_val"
import glob, os, unicodedata as ud
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
def find_file(name):
    for pat in [f"data/{name}", f"../data/{name}", f"/kaggle/working/{name}", f"/kaggle/input/*/{name}", f"/kaggle/input/*/*/{name}", f"/kaggle/input/*/data/{name}"]:
        hits = sorted(glob.glob(pat))
        if hits: return hits[0]
    raise FileNotFoundError(name)
split = pd.read_csv(find_file("split_assignments.csv")); rows = []
for seed in sorted(split.seed.unique()):
    sp = split[split.seed == seed]
    for part in ("val", "test"):
        for r in sp[sp.split == part].itertuples(): rows.append([r.filename, r.artist, seed, part, -1, part])
    train = sp[sp.split == "train"].reset_index(drop=True)          # row order = split_assignments.csv order (as in 01_train_main.py)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    for f, (tr_idx, oof_idx) in enumerate(skf.split(train, train["artist"])):
        fold_train = train.iloc[tr_idx]
        inner_tr, inner_val = train_test_split(fold_train, test_size=0.15, stratify=fold_train["artist"], random_state=int(seed) + f)
        for r in train.iloc[oof_idx].itertuples(): rows.append([r.filename, r.artist, seed, "train", f, "fold_oof"])
        for r in inner_tr.itertuples(): rows.append([r.filename, r.artist, seed, "train", f, "fold_train_inner"])
        for r in inner_val.itertuples(): rows.append([r.filename, r.artist, seed, "train", f, "fold_inner_val"])
out = pd.DataFrame(rows, columns=["filename", "artist", "seed", "split", "fold", "role"])
dst = os.path.join(os.path.dirname(find_file("split_assignments.csv")), "fold_assignments.csv"); out.to_csv(dst, index=False)
print(out.groupby(["seed", "role"]).size().unstack(fill_value=0)); print("written", dst)
