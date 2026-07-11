"""
McNemar evaluation script for:
Near-Duplicate Leakage in Fine-Art Classification:
A Reproducible Stacked CNN Baseline with Perceptual Hash Deduplication

Usage: python mcnemar_evaluation.py --predictions_dir /path/to/predictions_v2
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.contingency_tables import mcnemar
import argparse, os

SEEDS = [42, 0, 1, 7, 123]
BACKBONES = ["resnet50", "efficientnetv2b0", "inceptionv3", "xception"]

def duplicate_aware_split(df, seed, test_size=0.15, val_size=0.15):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["artist"], random_state=seed)
    val_adj = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_adj,
        stratify=train_val["artist"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def run_mcnemar(predictions_dir, retained_images_path, split_assignments_path):
    retained = pd.read_csv(retained_images_path, comment="#",
                           header=None, names=["filename","artist"])
    le = LabelEncoder()
    le.fit(sorted(retained["artist"].unique()))

    results = []
    for seed in SEEDS:
        splits = pd.read_csv(split_assignments_path)
        test_files = splits[(splits["seed"]==seed) & (splits["split"]=="test")]["filename"].tolist()
        train_files = splits[(splits["seed"]==seed) & (splits["split"]=="train")]["filename"].tolist()

        test_labels = le.transform(
            splits[(splits["seed"]==seed) & (splits["split"]=="test")]["artist"].values)
        train_labels = le.transform(
            splits[(splits["seed"]==seed) & (splits["split"]=="train")]["artist"].values)

        oof_preds = {}
        test_preds = {}
        for b in BACKBONES:
            oof_data  = np.load(f"{predictions_dir}/{b}_seed{seed}_oof.npz")
            test_data = np.load(f"{predictions_dir}/{b}_seed{seed}_test.npz")
            oof_preds[b]  = oof_data["oof"]
            test_preds[b] = test_data["test"]

        X_train = np.concatenate([oof_preds[b] for b in BACKBONES], axis=1)
        X_test  = np.concatenate([test_preds[b] for b in BACKBONES], axis=1)
        meta = LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
        meta.fit(X_train, train_labels)
        ens_preds = meta.predict(X_test)

        for baseline_name in ["resnet50", "xception"]:
            base_preds = np.argmax(test_preds[baseline_name], axis=1)
            ens_correct  = (ens_preds == test_labels)
            base_correct = (base_preds == test_labels)
            b_val = int(np.sum(ens_correct & ~base_correct))
            c_val = int(np.sum(~ens_correct & base_correct))
            res = mcnemar([[0, b_val],[c_val, 0]], exact=True)
            results.append({
                "seed": seed, "baseline": baseline_name,
                "b": b_val, "c": c_val,
                "pvalue": res.pvalue,
                "significant_p05": res.pvalue < 0.05
            })
            print(f"Seed {seed} vs {baseline_name}: b={b_val}, c={c_val}, p={res.pvalue:.6f}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--retained_images", default="retained_images.txt")
    parser.add_argument("--split_assignments", default="split_assignments.csv")
    args = parser.parse_args()
    results = run_mcnemar(args.predictions_dir, args.retained_images, args.split_assignments)
    results.to_csv("mcnemar_results.csv", index=False)
    print("\nSaved: mcnemar_results.csv")
