# Leakage Isolation Experiment (corrected protocol)
#
# Xception trained on the NON-deduplicated dataset (4,388 images, 10 classes)
# using EXACTLY the same protocol as notebooks/01_train_main.py:
#   * same 70/15/15 stratified split per seed (seeds 42, 0, 1, 7, 123)
#   * same 5-fold StratifiedKFold on the 70% training split
#   * same inner triple partition per fold
#       fold_train_inner -> gradient updates only
#       fold_inner_val   -> early stopping only (15% of fold_train, random_state = seed + fold)
#       fold_oof         -> out-of-fold predictions only
#   * same two-stage fine-tuning (head: Adam 1e-4, 20 ep; last 30 layers: Adam 5e-5, 20 ep)
#   * same callbacks, augmentation, class weighting, batch size
#   * same test-time protocol: test prediction = mean of the 5 fold models
#
# The ONLY difference from the deduplicated Xception run is that pHash
# near-duplicates are retained. Everything else is held constant, so the
# per-seed accuracy difference isolates the effect of near-duplicate leakage.
#
# The previous version of this script trained ONE model on the full 70% split
# with early stopping on the 15% validation split and no fold averaging, which
# confounded duplicate retention with training-set size, early-stopping set and
# fold-ensembling. Results produced by that version must not be reported.

import subprocess
subprocess.run(["pip", "install", "imagehash", "-q"], check=True)

import os, gc, random, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
warnings.filterwarnings('ignore')

# ── Constants (identical to 01_train_main.py for the Xception backbone) ───────
SEEDS         = [42, 0, 1, 7, 123]
BATCH_SIZE    = 32
IMG_SIZE      = (299, 299)
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20
PATIENCE      = 10
N_FOLDS       = 5
UNFREEZE      = 30
NUM_CLASSES   = 10
RAW_IMAGE_COUNT = 4388

SAVE_DIR       = "/kaggle/working/predictions_xception_nodup"
DEDUP_PRED_DIR = "/kaggle/working/predictions_v3"   # output of 01_train_main.py
os.makedirs(SAVE_DIR, exist_ok=True)

# Fallback per-seed deduplicated Xception accuracies (Table 5 of the paper),
# used only when the .npz files from 01_train_main.py are not available.
DEDUP_XCEPTION_FALLBACK = {42: 86.47, 0: 86.99, 1: 88.22, 7: 84.18, 123: 86.99}

img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir):
    img_dir = "/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir):
    raise FileNotFoundError("Dataset not found.")
print(f"Dataset path: {img_dir}")

# ── Helper functions (copied from 01_train_main.py) ───────────────────────────
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def duplicate_aware_split(df, seed, test_size=0.15, val_size=0.15):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df['artist'], random_state=seed)
    val_adj = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_adj,
        stratify=train_val['artist'], random_state=seed)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))

# ── Deterministic row order (platform-independent) ────────────────────────────
# os.listdir() order differs between operating systems and train_test_split depends on row order.
# The non-deduplicated set has no shipped split file, so rows are sorted by (NFC-normalised) filename
# before splitting; this makes the split identical on every machine.
import unicodedata as _ud
_orig_duplicate_aware_split = duplicate_aware_split
def duplicate_aware_split(df, seed, test_size=0.15, val_size=0.15):
    order = sorted(range(len(df)), key=lambda i: _ud.normalize('NFC', df['filename'].iat[i]))
    return _orig_duplicate_aware_split(df.iloc[order].reset_index(drop=True), seed, test_size, val_size)
# ──────────────────────────────────────────────────────────────────────────────

def build_generators(train_df, val_df, img_size, preprocess_fn, classes):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15, width_shift_range=0.10,
        height_shift_range=0.10, shear_range=0.10,
        zoom_range=0.10, horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    kw = dict(x_col='filepath', y_col='artist', target_size=img_size,
              batch_size=BATCH_SIZE, class_mode='categorical', classes=classes)
    train_gen = train_datagen.flow_from_dataframe(train_df, shuffle=True, **kw)
    val_gen   = val_datagen.flow_from_dataframe(val_df, shuffle=False, **kw)
    return train_gen, val_gen

def plain_generator(df, img_size, preprocess_fn, classes):
    return ImageDataGenerator(preprocessing_function=preprocess_fn).flow_from_dataframe(
        df, x_col='filepath', y_col='artist', target_size=img_size,
        batch_size=BATCH_SIZE, class_mode='categorical', classes=classes, shuffle=False)

def build_backbone(img_size):
    base = Xception(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output), base

def train_backbone(img_size, train_gen, inner_val_gen, class_weights, seed):
    """
    Leakage-free OOF protocol (identical to 01_train_main.py):
    - train_gen:     fold_train_inner only
    - inner_val_gen: carved from fold_train only, used for early stopping
    - fold_oof:      NEVER passed here, used only for OOF predictions after training
    """
    tf.random.set_seed(seed)
    model, base = build_backbone(img_size)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    # Stage 1: train head only (lr = 1e-4)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=inner_val_gen, epochs=EPOCHS_STAGE1,
              class_weight=class_weights, callbacks=callbacks, verbose=0)
    # Stage 2: unfreeze last N layers (lr = 5e-5)
    base.trainable = True
    for layer in base.layers[:-UNFREEZE]:
        layer.trainable = False
    model.compile(optimizer=Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    train_gen.reset()
    inner_val_gen.reset()
    model.fit(train_gen, validation_data=inner_val_gen, epochs=EPOCHS_STAGE2,
              class_weight=class_weights, callbacks=callbacks, verbose=0)
    return model

# ── Build the NON-deduplicated dataset (no pHash removal) ─────────────────────
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
            records.append({"filename": fname,
                            "filepath": os.path.join(img_dir, fname),
                            "artist": artist})
            break
df = pd.DataFrame(records)
print(f"NON-deduplicated dataset: {len(df)} images (duplicates retained)")
if len(df) != RAW_IMAGE_COUNT:
    print(f"WARNING: expected {RAW_IMAGE_COUNT} raw images, found {len(df)}. "
          f"Check the dataset version before reporting results.")

classes = sorted(df['artist'].unique())
le = LabelEncoder(); le.fit(classes)
classes_arr = np.array(classes)

# ── Main loop: identical structure to 01_train_main.py ────────────────────────
all_results = []
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed} - Xception WITHOUT deduplication (5-fold OOF)\n{'='*60}")
    oof_save  = f"{SAVE_DIR}/xception_nodup_seed{seed}_oof.npz"
    test_save = f"{SAVE_DIR}/xception_nodup_seed{seed}_test.npz"
    if os.path.exists(oof_save) and os.path.exists(test_save):
        data = np.load(test_save)
        all_results.append({'seed': seed, 'acc_nodup': float(data['acc'])*100,
                            'f1_nodup': float(data['f1'])*100})
        print(f"Loaded: acc={float(data['acc'])*100:.2f}%")
        continue

    set_seed(seed)
    train, val, test = duplicate_aware_split(df, seed=seed)   # val unused, exactly as in 01_train_main.py
    train_labels = le.transform(train['artist'].values)
    test_labels  = le.transform(test['artist'].values)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(train), NUM_CLASSES))
    fold_test_preds = []

    for fold, (tr_idx, oof_idx) in enumerate(skf.split(train, train_labels)):
        print(f"    Fold {fold+1}/{N_FOLDS}...", end=' ', flush=True)
        fold_train = train.iloc[tr_idx].reset_index(drop=True)
        fold_oof   = train.iloc[oof_idx].reset_index(drop=True)

        # Inner val carved from fold_train only (same random_state rule as 01_train_main.py)
        fold_train_inner, fold_inner_val = train_test_split(
            fold_train, test_size=0.15,
            stratify=fold_train['artist'], random_state=seed + fold)

        fold_cw = dict(enumerate(compute_class_weight(
            'balanced', classes=classes_arr, y=fold_train_inner['artist'].values)))

        train_gen, inner_val_gen = build_generators(
            fold_train_inner, fold_inner_val, IMG_SIZE, xception_preprocess, classes)
        oof_gen  = plain_generator(fold_oof, IMG_SIZE, xception_preprocess, classes)
        test_gen = plain_generator(test,     IMG_SIZE, xception_preprocess, classes)

        model = train_backbone(IMG_SIZE, train_gen, inner_val_gen, fold_cw, seed + fold)

        oof_gen.reset()
        oof_preds[oof_idx] = model.predict(oof_gen, verbose=0)
        test_gen.reset()
        fold_test_preds.append(model.predict(test_gen, verbose=0))

        fold_acc = accuracy_score(test_labels, np.argmax(fold_test_preds[-1], axis=1))
        print(f"fold acc={fold_acc:.4f}", flush=True)
        del model; gc.collect(); tf.keras.backend.clear_session()

    test_p = np.mean(fold_test_preds, axis=0)          # fold-averaged, as in 01_train_main.py
    acc = accuracy_score(test_labels, np.argmax(test_p, axis=1))
    f1  = f1_score(test_labels, np.argmax(test_p, axis=1), average='macro')
    print(f"Seed {seed}: acc={acc*100:.2f}% f1={f1*100:.2f}%")
    np.savez(oof_save,  oof=oof_preds)
    np.savez(test_save, test=test_p, acc=acc, f1=f1)
    all_results.append({'seed': seed, 'acc_nodup': acc*100, 'f1_nodup': f1*100})

# ── Paired comparison against the deduplicated Xception run ──────────────────
results_df = pd.DataFrame(all_results)
dedup_accs, dedup_source = [], []
for seed in results_df['seed']:
    p = f"{DEDUP_PRED_DIR}/xception_seed{seed}_test.npz"
    if os.path.exists(p):
        dedup_accs.append(float(np.load(p)['acc'])*100); dedup_source.append('npz')
    else:
        dedup_accs.append(DEDUP_XCEPTION_FALLBACK[seed]); dedup_source.append('table5')
results_df['acc_dedup'] = dedup_accs
results_df['dedup_source'] = dedup_source
results_df['leakage_effect_pp'] = results_df['acc_nodup'] - results_df['acc_dedup']

d = results_df['leakage_effect_pp'].values
n = len(d)
se = d.std(ddof=1) / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_lo, ci_hi = d.mean() - t_crit*se, d.mean() + t_crit*se

print(f"\n{'='*60}\nLEAKAGE ISOLATION (same 5-fold OOF protocol, duplicates retained vs removed)\n{'='*60}")
print(results_df.to_string(index=False))
print(f"\nXception WITH duplicates    : {results_df['acc_nodup'].mean():.2f}% +/-{results_df['acc_nodup'].std(ddof=1):.2f}%")
print(f"Xception WITHOUT duplicates : {results_df['acc_dedup'].mean():.2f}% +/-{results_df['acc_dedup'].std(ddof=1):.2f}%")
print(f"Isolated leakage effect     : {d.mean():.2f}pp  (paired 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]pp, n={n} seeds)")
if 'table5' in dedup_source:
    print("NOTE: some deduplicated accuracies came from Table 5 fallback values, "
          "not from 01_train_main.py outputs.")
results_df.to_csv("/kaggle/working/xception_nodup_results.csv", index=False)
