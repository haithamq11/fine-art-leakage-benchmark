# 09_group_aware_split.py
# Round-2 experiment for reviewer 2, point 1: deletion policy vs. a duplicate-GROUP-AWARE split.
#
# Group-aware split: all 4,388 images are kept. Every pHash duplicate group (data/duplicate_groups.csv) is treated
# as one unit, so a group is assigned entirely to train, validation or test; no near-duplicate pair can straddle
# the split. Stratified 70/15/15 by artist at the group level (seeded as in the paper). Only training groups
# contribute variants to training, i.e. the model sees digitisation variants as augmentation but never a variant
# of a test image.
# Arms compared on Xception with the identical 5-fold OOF protocol of 01_train_main.py:
#   * "dedup"       : the paper's deduplicated split (data/split_assignments.csv)      -> already measured (Table 6)
#   * "group_aware" : this script (all images retained, group-level split)
# Because the two arms have different test sets (569 deduplicated vs. ~659 group-aware), the comparison is reported
# per seed as a benchmark-policy comparison, not as a paired leakage estimate.
# Output: /kaggle/working/group_aware_results.csv and /kaggle/working/group_aware_split_assignments.csv
# Runtime: ~4 h per seed on T4 x2 (same as one Xception run of 07).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=False)
import os, gc, glob, random, unicodedata as ud
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
SAVE_DIR = "/kaggle/working/predictions_group_aware"; os.makedirs(SAVE_DIR, exist_ok=True)
img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir): img_dir = "/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"
def find_file(name):
    for pat in [f"data/{name}", f"../data/{name}", f"/kaggle/working/{name}", f"/kaggle/input/*/{name}", f"/kaggle/input/*/*/{name}", f"/kaggle/input/*/data/{name}"]:
        hits = sorted(glob.glob(pat))
        if hits: return hits[0]
    raise FileNotFoundError(name)
NFC = lambda s: ud.normalize("NFC", s)
groups = pd.read_csv(find_file("duplicate_groups.csv"))
for c in ("group_representative", "member"): groups[c] = groups[c].map(NFC)
ARTIST_MATCHERS = {"Vincent_van_Gogh": lambda f: f.startswith("Vincent_van_Gogh_"), "Edgar_Degas": lambda f: f.startswith("Edgar_Degas_"), "Pablo_Picasso": lambda f: f.startswith("Pablo_Picasso_"), "Pierre-Auguste_Renoir": lambda f: f.startswith("Pierre-Auguste_Renoir_"), "Albrecht_Dürer": lambda f: f.startswith("Albrecht_D") and "rer_" in f, "Paul_Gauguin": lambda f: f.startswith("Paul_Gauguin_"), "Francisco_Goya": lambda f: f.startswith("Francisco_Goya_"), "Rembrandt": lambda f: f.startswith("Rembrandt_"), "Alfred_Sisley": lambda f: f.startswith("Alfred_Sisley_"), "Titian": lambda f: f.startswith("Titian_")}
records = []
for fname in sorted(os.listdir(img_dir), key=NFC):
    for artist, m in ARTIST_MATCHERS.items():
        if m(fname): records.append({"filename": fname, "key": NFC(fname), "filepath": os.path.join(img_dir, fname), "artist": artist}); break
df_all = pd.DataFrame(records); assert len(df_all) == 4388
rep_of = dict(zip(groups["member"], groups["group_representative"]))
df_all["group"] = df_all["key"].map(lambda k: rep_of.get(k, k))          # singletons are their own group
classes = np.array(sorted(df_all["artist"].unique())); le = LabelEncoder().fit(classes)

def group_aware_split(df, seed, test_size=0.15, val_size=0.15):
    g = df.groupby("group").agg(artist=("artist", "first"), n=("key", "size")).reset_index().sort_values("group").reset_index(drop=True)
    tv, te = train_test_split(g, test_size=test_size, stratify=g["artist"], random_state=seed)
    tr, va = train_test_split(tv, test_size=val_size / (1 - test_size), stratify=tv["artist"], random_state=seed)
    lab = {**{x: "train" for x in tr["group"]}, **{x: "val" for x in va["group"]}, **{x: "test" for x in te["group"]}}
    return df.assign(split=df["group"].map(lab))
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
def build_model():
    base = Xception(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)); base.trainable = False
    x = GlobalAveragePooling2D()(base.output); x = Dense(256, activation="relu")(x); x = Dropout(0.5)(x)
    return Model(base.input, Dense(10, activation="softmax")(x)), base
def gen(df, augment, shuffle):
    kw = dict(rotation_range=15, width_shift_range=0.10, height_shift_range=0.10, shear_range=0.10, zoom_range=0.10, horizontal_flip=True, fill_mode="nearest") if augment else {}
    return ImageDataGenerator(preprocessing_function=preprocess_input, **kw).flow_from_dataframe(df, x_col="filepath", y_col="artist", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=shuffle, classes=list(classes))

all_assign = []; results = []
for seed in SEEDS:
    d = group_aware_split(df_all, seed); all_assign.append(d[["filename", "artist", "group", "split"]].assign(seed=seed))
    train = d[d.split == "train"].reset_index(drop=True); test = d[d.split == "test"].reset_index(drop=True)
    assert not set(train["group"]) & set(test["group"])
    y_test = le.transform(test["artist"].values); out = f"{SAVE_DIR}/group_aware_seed{seed}_test.npz"
    print(f"seed {seed}: train {len(train)} val {(d.split=='val').sum()} test {len(test)} images; groups straddling train/test: 0", flush=True)
    if os.path.exists(out): probs = np.load(out)["test"]; print("  restored")
    else:
        set_seed(seed); skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed); tp = []
        # folds are also group-aware: fold assignment by group so that no variant pair straddles the inner folds
        g = train.groupby("group").agg(artist=("artist", "first")).reset_index()
        for f, (tr_g, oof_g) in enumerate(skf.split(g, g["artist"])):
            fold_train = train[train.group.isin(g.iloc[tr_g]["group"])]
            gi = fold_train.groupby("group").agg(artist=("artist", "first")).reset_index()
            itr_g, ival_g = train_test_split(gi, test_size=0.15, stratify=gi["artist"], random_state=seed + f)
            inner_tr = fold_train[fold_train.group.isin(itr_g["group"])]; inner_val = fold_train[fold_train.group.isin(ival_g["group"])]
            cw = dict(enumerate(compute_class_weight("balanced", classes=classes, y=inner_tr["artist"].values)))
            model, base = build_model()
            cb = [EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True), ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7)]
            model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
            model.fit(gen(inner_tr, True, True), validation_data=gen(inner_val, False, False), epochs=EPOCHS_STAGE1, class_weight=cw, callbacks=cb, verbose=0)
            base.trainable = True
            for layer in base.layers[:-UNFREEZE]: layer.trainable = False
            model.compile(optimizer=Adam(5e-5), loss="categorical_crossentropy", metrics=["accuracy"])
            model.fit(gen(inner_tr, True, True), validation_data=gen(inner_val, False, False), epochs=EPOCHS_STAGE2, class_weight=cw, callbacks=cb, verbose=0)
            tp.append(model.predict(gen(test, False, False), verbose=0)); print(f"  fold {f} done", flush=True)
            del model, base; gc.collect(); tf.keras.backend.clear_session()
        probs = np.mean(tp, axis=0); np.savez(out, test=probs, y=y_test, test_files=np.array(test["filename"]))
    pred = np.argmax(probs, 1)
    # accuracy on all group-aware test images, and on the subset that are group representatives (deduplicated view)
    rep_mask = test["key"].values == test["group"].values
    results.append({"seed": seed, "n_test": len(test), "acc_all": accuracy_score(y_test, pred) * 100, "f1_all": f1_score(y_test, pred, average="macro") * 100,
                    "n_test_representatives": int(rep_mask.sum()), "acc_representatives": accuracy_score(y_test[rep_mask], pred[rep_mask]) * 100})
    print(results[-1], flush=True)
pd.concat(all_assign).to_csv("/kaggle/working/group_aware_split_assignments.csv", index=False)
res = pd.DataFrame(results); res.to_csv("/kaggle/working/group_aware_results.csv", index=False); print(res.to_string())
