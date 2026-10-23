# 08_leakage_fixed_testset.py
# Round-2 experiment for reviewer 1, point 1: a leakage estimate with a FIXED evaluation set.
#
# Design (per seed, identical to 01_train_main.py / 07_leakage_isolation.py otherwise):
#   * The deduplicated 70/15/15 split of data/split_assignments.csv is kept. The TEST set (569 images, deduplicated)
#     is therefore identical in both arms, so test identity, test size and class composition are held constant.
#   * Arm A ("clean")      : train on the 2,655 deduplicated training images exactly as in 01_train_main.py.
#   * Arm B ("duplicates") : train on the same 2,655 images PLUS every removed near-duplicate whose retained
#     representative lies in the TRAINING split (so only training receives duplicates; no duplicate of a test or
#     validation image is added). This isolates "duplicates present in training" from everything else.
#   * Arm C ("size control"): train on the 2,655 images PLUS the same NUMBER of randomly chosen extra images drawn
#     from the removed images whose representative is NOT in the test split, matching class counts of arm B.
#     Arm C controls for training-set size and class composition; arm B minus arm C is the duplicate effect.
#     (Images whose representative is in the test split are never used, so arm C adds no test leakage either.)
#   * Backbone: Xception, 5-fold OOF with inner validation subset, two-stage fine-tuning, same optimiser,
#     augmentation and class weighting; test prediction = mean over the 5 fold models.
# Output: /kaggle/working/leakage_fixed_testset.csv (seed, acc_clean, acc_dup, acc_size, n_extra, dup_effect_pp,
#         size_effect_pp) and per-seed npz with per-image test probabilities for all three arms.
# Runtime: ~3 x the Xception run of 07 (about 12 h per seed on T4 x2) -> run ONE SEED PER NOTEBOOK VERSION and
#          resume: existing npz files are reused.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=False)
import os, gc, glob, random, unicodedata as ud, math
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,0,1,7,123").split(",")]
BATCH_SIZE = 32; IMG_SIZE = (299, 299); EPOCHS_STAGE1 = 20; EPOCHS_STAGE2 = 20; PATIENCE = 10; N_FOLDS = 5; UNFREEZE = 30
SAVE_DIR = "/kaggle/working/predictions_fixed_testset"; os.makedirs(SAVE_DIR, exist_ok=True)
img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir): img_dir = "/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"
def find_file(name):
    for pat in [f"data/{name}", f"../data/{name}", f"/kaggle/working/{name}", f"/kaggle/input/*/{name}", f"/kaggle/input/*/*/{name}", f"/kaggle/input/*/data/{name}"]:
        hits = sorted(glob.glob(pat))
        if hits: return hits[0]
    raise FileNotFoundError(name)
NFC = lambda s: ud.normalize("NFC", s)
split = pd.read_csv(find_file("split_assignments.csv")); split["key"] = split["filename"].map(NFC)
groups = pd.read_csv(find_file("duplicate_groups.csv"))
for c in ("group_representative", "member"): groups[c] = groups[c].map(NFC)
ARTIST_MATCHERS = {"Vincent_van_Gogh": lambda f: f.startswith("Vincent_van_Gogh_"), "Edgar_Degas": lambda f: f.startswith("Edgar_Degas_"), "Pablo_Picasso": lambda f: f.startswith("Pablo_Picasso_"), "Pierre-Auguste_Renoir": lambda f: f.startswith("Pierre-Auguste_Renoir_"), "Albrecht_Dürer": lambda f: f.startswith("Albrecht_D") and "rer_" in f, "Paul_Gauguin": lambda f: f.startswith("Paul_Gauguin_"), "Francisco_Goya": lambda f: f.startswith("Francisco_Goya_"), "Rembrandt": lambda f: f.startswith("Rembrandt_"), "Alfred_Sisley": lambda f: f.startswith("Alfred_Sisley_"), "Titian": lambda f: f.startswith("Titian_")}
records = []
for fname in os.listdir(img_dir):
    for artist, m in ARTIST_MATCHERS.items():
        if m(fname): records.append({"filename": fname, "key": NFC(fname), "filepath": os.path.join(img_dir, fname), "artist": artist}); break
df_all = pd.DataFrame(records).set_index("key"); assert len(df_all) == 4388
classes = np.array(sorted(df_all["artist"].unique())); le = LabelEncoder().fit(classes)

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
def build_model():
    base = Xception(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)); base.trainable = False
    x = GlobalAveragePooling2D()(base.output); x = Dense(256, activation="relu")(x); x = Dropout(0.5)(x)
    return Model(base.input, Dense(10, activation="softmax")(x)), base
def gen(df, augment, shuffle):
    kw = dict(rotation_range=15, width_shift_range=0.10, height_shift_range=0.10, shear_range=0.10, zoom_range=0.10, horizontal_flip=True, fill_mode="nearest") if augment else {}
    return ImageDataGenerator(preprocessing_function=preprocess_input, **kw).flow_from_dataframe(df, x_col="filepath", y_col="artist", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=shuffle, classes=list(classes))
def train_and_predict(train_df, test_df, seed, tag):
    """5-fold OOF on train_df (inner validation carved from the 4 training folds); returns mean test probabilities."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed); test_probs = []
    for f, (tr_idx, oof_idx) in enumerate(skf.split(train_df, train_df["artist"])):
        fold_train = train_df.iloc[tr_idx]
        inner_tr, inner_val = train_test_split(fold_train, test_size=0.15, stratify=fold_train["artist"], random_state=seed + f)
        cw = dict(enumerate(compute_class_weight("balanced", classes=classes, y=inner_tr["artist"].values)))
        model, base = build_model()
        cb = [EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True), ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7)]
        model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(gen(inner_tr, True, True), validation_data=gen(inner_val, False, False), epochs=EPOCHS_STAGE1, class_weight=cw, callbacks=cb, verbose=0)
        base.trainable = True
        for layer in base.layers[:-UNFREEZE]: layer.trainable = False
        model.compile(optimizer=Adam(5e-5), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(gen(inner_tr, True, True), validation_data=gen(inner_val, False, False), epochs=EPOCHS_STAGE2, class_weight=cw, callbacks=cb, verbose=0)
        test_probs.append(model.predict(gen(test_df, False, False), verbose=0)); print(f"  [{tag}] fold {f} done", flush=True)
        del model, base; gc.collect(); tf.keras.backend.clear_session()
    return np.mean(test_probs, axis=0)

results = []
for seed in SEEDS:
    out = f"{SAVE_DIR}/fixed_testset_seed{seed}.npz"
    sp = split[split.seed == seed]
    train_keys = sp[sp.split == "train"]["key"].tolist(); test_keys = sp[sp.split == "test"]["key"].tolist(); val_keys = set(sp[sp.split == "val"]["key"])
    train_df = df_all.loc[train_keys].reset_index(); test_df = df_all.loc[test_keys].reset_index()
    y_test = le.transform(test_df["artist"].values)
    # removed duplicates whose representative is in the TRAINING split -> arm B extras
    rem = groups[groups.action == "remove"]
    extras_B = rem[rem.group_representative.isin(train_keys)]["member"].unique().tolist()
    extras_B = [k for k in extras_B if k in df_all.index]
    # size control: same number of removed images whose representative is in train or val (never test), class-matched
    pool = rem[~rem.group_representative.isin(test_keys)]["member"].unique().tolist(); pool = [k for k in pool if k in df_all.index and k not in extras_B]
    rng = random.Random(seed); extras_C = []
    for artist, cnt in df_all.loc[extras_B]["artist"].value_counts().items():
        cand = [k for k in pool if df_all.loc[k, "artist"] == artist]; rng.shuffle(cand); extras_C += cand[:cnt]
    print(f"seed {seed}: train {len(train_df)}, test {len(test_df)}, arm-B extras {len(extras_B)}, arm-C extras {len(extras_C)}", flush=True)
    if os.path.exists(out):
        d = np.load(out); pA, pB, pC = d["clean"], d["dup"], d["size"]; print("  restored", flush=True)
    else:
        set_seed(seed); pA = train_and_predict(train_df, test_df, seed, "clean")
        set_seed(seed); pB = train_and_predict(pd.concat([train_df, df_all.loc[extras_B].reset_index()]).reset_index(drop=True), test_df, seed, "dup")
        set_seed(seed); pC = train_and_predict(pd.concat([train_df, df_all.loc[extras_C].reset_index()]).reset_index(drop=True), test_df, seed, "size")
        np.savez(out, clean=pA, dup=pB, size=pC, y=y_test, test_files=np.array(test_df["filename"]), extras_B=np.array(extras_B), extras_C=np.array(extras_C))
    acc = {k: accuracy_score(y_test, np.argmax(p, 1)) * 100 for k, p in (("clean", pA), ("dup", pB), ("size", pC))}
    results.append({"seed": seed, "acc_clean": acc["clean"], "acc_dup": acc["dup"], "acc_size": acc["size"], "n_extra": len(extras_B),
                    "dup_effect_pp": acc["dup"] - acc["clean"], "size_effect_pp": acc["size"] - acc["clean"], "dup_minus_size_pp": acc["dup"] - acc["size"]})
    print(results[-1], flush=True)
res = pd.DataFrame(results); res.to_csv("/kaggle/working/leakage_fixed_testset.csv", index=False); print(res.to_string())
if len(res) > 1:
    from scipy import stats
    for col in ("dup_effect_pp", "size_effect_pp", "dup_minus_size_pp"):
        d = res[col].values; h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        print(f"{col}: {d.mean():+.2f} pp, 95% CI [{d.mean()-h:+.2f}, {d.mean()+h:+.2f}] (paired on the identical test set, n={len(d)})")
