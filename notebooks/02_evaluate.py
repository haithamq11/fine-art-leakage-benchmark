import os, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
from PIL import Image
import imagehash
from collections import defaultdict

# ── Setup ─────────────────────────────────────────────────────────────────────
SAVE_DIR = "/kaggle/working/predictions_v3"
os.makedirs(SAVE_DIR, exist_ok=True)

pred_input = "/kaggle/input/datasets/haithamqutaiba/predictions-vv3"
if os.path.exists(pred_input):
    for f in os.listdir(pred_input):
        if f.endswith(".npz"):
            shutil.copy(os.path.join(pred_input, f), os.path.join(SAVE_DIR, f))
    print(f"Restored: {len(os.listdir(SAVE_DIR))} files")

SEEDS     = [42, 0, 1, 7, 123]
BACKBONES = ['resnet50', 'efficientnetv2b0', 'inceptionv3', 'xception']

img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir):
    img_dir = "/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"

ARTIST_MATCHERS = {
    "Vincent_van_Gogh":      lambda f: f.startswith("Vincent_van_Gogh_"),
    "Edgar_Degas":           lambda f: f.startswith("Edgar_Degas_"),
    "Pablo_Picasso":         lambda f: f.startswith("Pablo_Picasso_"),
    "Pierre-Auguste_Renoir": lambda f: f.startswith("Pierre-Auguste_Renoir_"),
    "Albrecht_Dürer":        lambda f: f.startswith("Albrecht_D") and "rer_" in f,
    "Paul_Gauguin":          lambda f: f.startswith("Paul_Gauguin_"),
    "Francisco_Goya":        lambda f: f.startswith("Francisco_Goya_"),
    "Rembrandt":             lambda f: f.startswith("Rembrandt_"),
    "Alfred_Sisley":         lambda f: f.startswith("Alfred_Sisley_"),
    "Titian":                lambda f: f.startswith("Titian_"),
}

records = []
for fname in os.listdir(img_dir):
    for artist, matcher in ARTIST_MATCHERS.items():
        if matcher(fname):
            records.append({"filename": fname, "filepath": os.path.join(img_dir, fname), "artist": artist})
            break
df = pd.DataFrame(records)

print("Computing pHash...")
hashes = {}
for _, row in df.iterrows():
    try: hashes[row['filepath']] = imagehash.phash(Image.open(row['filepath']).convert("RGB"))
    except: pass

parent = {p: p for p in hashes}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y): parent[find(x)] = find(y)
paths = list(hashes.keys())
for i in range(len(paths)):
    for j in range(i+1, len(paths)):
        if hashes[paths[i]] - hashes[paths[j]] <= 10: union(paths[i], paths[j])
groups = defaultdict(list)
for p in paths: groups[find(p)].append(p)
paths_to_remove = set()
for rep, members in groups.items():
    if len(members) > 1:
        for m in members:
            if m != rep: paths_to_remove.add(m)
df_clean = df[~df['filepath'].isin(paths_to_remove)].reset_index(drop=True)
assert len(df_clean) == 3793
print(f"Clean dataset: {len(df_clean)} images")

le = LabelEncoder()
le.fit(sorted(df_clean['artist'].unique()))

def duplicate_aware_split(df, seed, test_size=0.15, val_size=0.15):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df['artist'], random_state=seed)
    val_adj = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_adj, stratify=train_val['artist'], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# ── EVALUATION 1: Main results ────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION 1: MAIN RESULTS")
print("="*60)

all_results = []
all_ensemble_preds = {}
all_test_labels = {}
all_soft_preds = {}
all_backbone_preds = {}
all_meta_coefs = []

for seed in SEEDS:
    train, val, test = duplicate_aware_split(df_clean, seed=seed)
    train_labels = le.transform(train['artist'].values)
    test_labels  = le.transform(test['artist'].values)

    oof_preds  = {b: np.load(f"{SAVE_DIR}/{b}_seed{seed}_oof.npz")['oof']  for b in BACKBONES}
    test_preds = {b: np.load(f"{SAVE_DIR}/{b}_seed{seed}_test.npz")['test'] for b in BACKBONES}

    X_train = np.concatenate([oof_preds[b]  for b in BACKBONES], axis=1)
    X_test  = np.concatenate([test_preds[b] for b in BACKBONES], axis=1)

    meta = LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
    meta.fit(X_train, train_labels)

    ens_preds = meta.predict(X_test)
    acc_ens   = accuracy_score(test_labels, ens_preds) * 100
    f1_mac    = f1_score(test_labels, ens_preds, average='macro') * 100
    f1_wt     = f1_score(test_labels, ens_preds, average='weighted') * 100
    avg_probs = np.mean([test_preds[b] for b in BACKBONES], axis=0)
    acc_soft  = accuracy_score(test_labels, np.argmax(avg_probs, axis=1)) * 100
    indiv     = {b: accuracy_score(test_labels, np.argmax(test_preds[b], axis=1)) * 100 for b in BACKBONES}

    coef = meta.coef_
    backbone_weights = {b: float(np.mean(np.abs(coef[:, i*10:(i+1)*10]))) for i, b in enumerate(BACKBONES)}
    all_meta_coefs.append({'seed': seed, **backbone_weights})

    all_results.append({'seed': seed, **indiv, 'soft_voting': acc_soft,
                        'ensemble_acc': acc_ens, 'f1_macro': f1_mac, 'f1_weighted': f1_wt})
    all_ensemble_preds[seed] = ens_preds
    all_test_labels[seed]    = test_labels
    all_soft_preds[seed]     = np.argmax(avg_probs, axis=1)
    all_backbone_preds[seed] = test_preds
    print(f"Seed {seed}: ensemble={acc_ens:.2f}% soft={acc_soft:.2f}% f1={f1_mac:.2f}%")

results_df = pd.DataFrame(all_results)
print(f"\nFinal results:")
print(results_df.to_string(index=False))
for col in ['resnet50','efficientnetv2b0','inceptionv3','xception','soft_voting','ensemble_acc','f1_macro']:
    vals = results_df[col].values
    print(f"  {col}: {np.mean(vals):.2f}% +/-{np.std(vals, ddof=1):.2f}%")
results_df.to_csv("/kaggle/working/all_seeds_results.csv", index=False)

# ── EVALUATION 2: McNemar vs ResNet50 and Xception ───────────────────────────
print("\n" + "="*60)
print("EVALUATION 2: McNEMAR vs ResNet50 AND Xception")
print("="*60)
mcnemar_results = []
for seed in SEEDS:
    ens  = all_ensemble_preds[seed]
    labs = all_test_labels[seed]
    r50  = np.argmax(all_backbone_preds[seed]['resnet50'], axis=1)
    xcp  = np.argmax(all_backbone_preds[seed]['xception'], axis=1)
    b_r  = int(np.sum((ens == labs) & (r50 != labs)))
    c_r  = int(np.sum((ens != labs) & (r50 == labs)))
    p_r  = mcnemar([[0, b_r],[c_r, 0]], exact=True).pvalue
    b_x  = int(np.sum((ens == labs) & (xcp != labs)))
    c_x  = int(np.sum((ens != labs) & (xcp == labs)))
    p_x  = mcnemar([[0, b_x],[c_x, 0]], exact=True).pvalue
    print(f"Seed {seed}: b_R50={b_r} c_R50={c_r} p_R50={p_r:.6f} {'Yes*' if p_r<0.05 else 'No'} | b_Xcp={b_x} c_Xcp={c_x} p_Xcp={p_x:.6f} {'Yes*' if p_x<0.05 else 'No'}")
    mcnemar_results.append({'seed': seed, 'b_vs_resnet50': b_r, 'c_vs_resnet50': c_r,
                            'p_vs_resnet50': p_r, 'b_vs_xception': b_x,
                            'c_vs_xception': c_x, 'p_vs_xception': p_x})
pd.DataFrame(mcnemar_results).to_csv("/kaggle/working/mcnemar_results.csv", index=False)

# ── EVALUATION 3: McNemar vs Soft Voting ─────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION 3: McNEMAR vs SOFT VOTING + 95% CI")
print("="*60)
gains = []
for seed in SEEDS:
    ens  = all_ensemble_preds[seed]
    soft = all_soft_preds[seed]
    labs = all_test_labels[seed]
    b    = int(np.sum((ens == labs) & (soft != labs)))
    c    = int(np.sum((ens != labs) & (soft == labs)))
    p    = mcnemar([[0, b],[c, 0]], exact=True).pvalue
    ens_acc  = accuracy_score(labs, ens)  * 100
    soft_acc = accuracy_score(labs, soft) * 100
    gains.append(ens_acc - soft_acc)
    print(f"Seed {seed}: b={b} c={c} p={p:.6f} {'Yes*' if p<0.05 else 'No'}")

gains  = np.array(gains)
n      = len(gains)
se     = np.std(gains, ddof=1) / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_low, ci_high = np.mean(gains) - t_crit * se, np.mean(gains) + t_crit * se
print(f"\nMean gain: {np.mean(gains):.2f}pp | 95% CI: [{ci_low:.2f}, {ci_high:.2f}]pp")

# ── EVALUATION 4: Ablation ────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION 4: BACKBONE ABLATION")
print("="*60)
ablation_configs = {
    'Full ensemble (4 backbones)':  BACKBONES,
    'Without EfficientNetV2B0':     ['resnet50','inceptionv3','xception'],
    'Without ResNet50':             ['efficientnetv2b0','inceptionv3','xception'],
    'Without InceptionV3':          ['resnet50','efficientnetv2b0','xception'],
    'Without Xception':             ['resnet50','efficientnetv2b0','inceptionv3'],
    'Soft voting (4 backbones)':    BACKBONES,
}
ablation_results = {}
for config_name, backbone_list in ablation_configs.items():
    accs = []
    for seed in SEEDS:
        train, val, test = duplicate_aware_split(df_clean, seed=seed)
        train_labels = le.transform(train['artist'].values)
        test_labels  = le.transform(test['artist'].values)
        oof_p  = {b: np.load(f"{SAVE_DIR}/{b}_seed{seed}_oof.npz")['oof']  for b in backbone_list}
        test_p = {b: np.load(f"{SAVE_DIR}/{b}_seed{seed}_test.npz")['test'] for b in backbone_list}
        if config_name == 'Soft voting (4 backbones)':
            acc = accuracy_score(test_labels, np.argmax(np.mean([test_p[b] for b in backbone_list], axis=0), axis=1)) * 100
        else:
            X_tr = np.concatenate([oof_p[b]  for b in backbone_list], axis=1)
            X_te = np.concatenate([test_p[b] for b in backbone_list], axis=1)
            meta = LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
            meta.fit(X_tr, train_labels)
            acc = accuracy_score(test_labels, meta.predict(X_te)) * 100
        accs.append(acc)
    ablation_results[config_name] = accs
    print(f"  {config_name:<35}: {np.mean(accs):.2f}% +/-{np.std(accs, ddof=1):.2f}%")

full_mean = np.mean(ablation_results['Full ensemble (4 backbones)'])
print(f"\nVs full ensemble:")
for config_name, accs in ablation_results.items():
    print(f"  {config_name:<35}: {np.mean(accs)-full_mean:+.2f}pp")

# ── EVALUATION 5: Per-class report ───────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION 5: PER-CLASS REPORT (representative seed)")
print("="*60)
ens_accs = [r['ensemble_acc'] for r in all_results]
rep_seed = SEEDS[np.argmin([abs(a - np.mean(ens_accs)) for a in ens_accs])]
print(f"Representative seed: {rep_seed}")
train_r, val_r, test_r = duplicate_aware_split(df_clean, seed=rep_seed)
test_labels_r  = le.transform(test_r['artist'].values)
train_labels_r = le.transform(train_r['artist'].values)
oof_r  = {b: np.load(f"{SAVE_DIR}/{b}_seed{rep_seed}_oof.npz")['oof']  for b in BACKBONES}
test_r2= {b: np.load(f"{SAVE_DIR}/{b}_seed{rep_seed}_test.npz")['test'] for b in BACKBONES}
X_tr   = np.concatenate([oof_r[b]  for b in BACKBONES], axis=1)
X_te   = np.concatenate([test_r2[b] for b in BACKBONES], axis=1)
meta_r = LogisticRegression(max_iter=5000, C=1.0, random_state=rep_seed)
meta_r.fit(X_tr, train_labels_r)
ens_preds_r = meta_r.predict(X_te)
print(classification_report(test_labels_r, ens_preds_r,
      target_names=sorted(df_clean['artist'].unique())))
print("\nALL EVALUATIONS COMPLETE")
