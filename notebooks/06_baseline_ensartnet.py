# EnsArtNet Reproduction [18] — Mezina & Burget 2025
# Architecture: 4 backbone branches + feature concatenation + focal loss
# Run on Kaggle GPU — approximately 4-5 hours for 5 seeds
#
# Fixes relative to the previous version of this script:
#   1. Adam(lr=...) -> Adam(learning_rate=...). Keras 3 (TensorFlow >= 2.16)
#      rejects the `lr` keyword, so the old script crashed before training.
#   2. Each branch now receives its OWN correctly preprocessed input:
#        ResNet50          -> resnet50.preprocess_input       (224x224)
#        EfficientNetV2B0  -> raw 0..255 pixels                (224x224; rescaling is inside the model)
#        InceptionV3       -> inception_v3.preprocess_input   (299x299)
#        Xception          -> xception.preprocess_input        (299x299, identical to inception's)
#      Previously EfficientNetV2B0 was fed ResNet-preprocessed (BGR, mean-subtracted) pixels.
#   3. Training-time augmentation is now applied, using the SAME policy as the
#      proposed framework (rotation 15, shift 10%, shear 10%, zoom 10%, h-flip, nearest fill).
#      The paper states this policy was used; the old script had no augmentation at all.
#   4. Optional Stage 2 fine-tuning (FINE_TUNE = True) unfreezes the last N layers of
#      each backbone with the same N and learning rate as the proposed framework, so the
#      reproduction is trained under the same freezing policy as the four proposed
#      backbones. Set FINE_TUNE = False to keep all backbones frozen (the behaviour of the
#      old script). Whichever setting is used MUST be stated in Section 4.3 of the paper.
#   5. steps_per_epoch uses ceil() so the last partial batch is not silently dropped.

import subprocess
subprocess.run(["pip", "install", "imagehash", "-q"], check=True)

import os, gc, math, random, shutil, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50, EfficientNetV2B0, InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout,
                                     BatchNormalization, Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import imagehash
from collections import defaultdict
warnings.filterwarnings('ignore')

SEEDS         = [42, 0, 1, 7, 123]
BATCH_SIZE    = 32
IMG_SIZE_STD  = (224, 224)
IMG_SIZE_INC  = (299, 299)
NUM_CLASSES   = 10
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20
PATIENCE      = 10
LR_STAGE1     = 1e-5          # as reported in the paper (Adam, lr = 0.00001)
LR_STAGE2     = 5e-5          # same as the proposed framework's Stage 2
FOCAL_GAMMA   = 2.0
FOCAL_ALPHA   = 0.25
FINE_TUNE     = True          # False = keep all four backbones frozen (old behaviour)
UNFREEZE      = {"r50": 30, "eff": 20, "inc": 30, "xcp": 30}   # same as 01_train_main.py
SAVE_DIR      = "/kaggle/working/predictions_ensartnet"
os.makedirs(SAVE_DIR, exist_ok=True)

pred_input = "/kaggle/input/datasets/haithamq11/predictions-ensartnet/"
if os.path.exists(pred_input):
    for f in os.listdir(pred_input):
        if f.endswith(".npz"): shutil.copy(os.path.join(pred_input, f), os.path.join(SAVE_DIR, f))
    print(f"Restored: {len(os.listdir(SAVE_DIR))} files")

img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir): img_dir = "/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def duplicate_aware_split(df, seed, test_size=0.15, val_size=0.15):
    tv, test = train_test_split(df, test_size=test_size, stratify=df['artist'], random_state=seed)
    train, val = train_test_split(tv, test_size=val_size/(1-test_size), stratify=tv['artist'], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# ── Reproducible splits (platform-independent) ────────────────────────────────
# os.listdir() order differs between operating systems, and train_test_split depends on row order,
# so recomputing the split on another machine silently produces DIFFERENT train/val/test images.
# The exact splits used in the paper are shipped in data/split_assignments.csv and are used when found.
import glob as _glob, unicodedata as _ud
def _find_split_file():
    for pat in ['data/split_assignments.csv', '../data/split_assignments.csv', '/kaggle/working/split_assignments.csv',
                '/kaggle/input/*/split_assignments.csv', '/kaggle/input/*/*/split_assignments.csv', '/kaggle/input/*/data/split_assignments.csv']:
        hits = sorted(_glob.glob(pat))
        if hits: return hits[0]
    return None
_SPLIT_FILE = _find_split_file()
_orig_duplicate_aware_split = duplicate_aware_split
def duplicate_aware_split(df, seed, test_size=0.15, val_size=0.15):
    if _SPLIT_FILE is None:
        print("WARNING: split_assignments.csv not found; splitting a filename-sorted dataframe. "
              "This is deterministic but may NOT reproduce the paper's exact splits.")
        order = sorted(range(len(df)), key=lambda i: _ud.normalize('NFC', df['filename'].iat[i]))
        return _orig_duplicate_aware_split(df.iloc[order].reset_index(drop=True), seed, test_size, val_size)
    sp = pd.read_csv(_SPLIT_FILE)
    m = df.set_index(df['filename'].map(lambda s: _ud.normalize('NFC', s)).rename('_k'))
    out = []
    for part in ('train', 'val', 'test'):
        rows = sp[(sp['seed'] == seed) & (sp['split'] == part)]
        keys = rows['filename'].map(lambda s: _ud.normalize('NFC', s)).tolist()
        missing = [k for k in keys if k not in m.index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} filenames from {_SPLIT_FILE} are not in the dataset, e.g. {missing[:3]}. "
                                    "On Windows the Duerer files may have been re-encoded by the archive extractor; see README.")
        out.append(m.loc[keys].reset_index(drop=True)[list(df.columns)])
    print(f"Splits for seed {seed} taken from {_SPLIT_FILE}")
    return tuple(out)
# ──────────────────────────────────────────────────────────────────────────────

# ── Model ─────────────────────────────────────────────────────────────────────
def build_ensartnet(num_classes=NUM_CLASSES):
    """
    EnsArtNet following Mezina & Burget [18]:
    - 4 branches: ResNet50, EfficientNetV2B0, InceptionV3, Xception
    - Each: GAP -> Dense(256) -> BatchNorm -> Dropout(0.5)
    - Concatenate -> Dense(256) -> Dense(64) -> Dropout(0.3) -> Dense(10, softmax)
    - Loss: Focal (gamma=2.0, alpha=0.25)
    Returns the model and the dict of backbone objects (needed for Stage 2 unfreezing).
    """
    in_r50 = Input(shape=(*IMG_SIZE_STD, 3), name='in_r50')
    in_eff = Input(shape=(*IMG_SIZE_STD, 3), name='in_eff')
    in_inc = Input(shape=(*IMG_SIZE_INC, 3), name='in_inc')
    in_xcp = Input(shape=(*IMG_SIZE_INC, 3), name='in_xcp')

    bases = {
        'r50': ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE_STD, 3)),
        'eff': EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE_STD, 3)),
        'inc': InceptionV3(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE_INC, 3)),
        'xcp': Xception(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE_INC, 3)),
    }
    def branch(name, inp):
        base = bases[name]
        base._name = f'base_{name}'
        base.trainable = False
        x = base(inp, training=False)          # BN layers stay in inference mode during fine-tuning
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', name=f'd_{name}')(x)
        x = BatchNormalization(name=f'bn_{name}')(x)
        x = Dropout(0.5, name=f'dr_{name}')(x)
        return x
    m = Concatenate()([branch('r50', in_r50), branch('eff', in_eff),
                       branch('inc', in_inc), branch('xcp', in_xcp)])
    x = Dense(256, activation='relu')(m)
    x = Dense(64,  activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=[in_r50, in_eff, in_inc, in_xcp], outputs=out), bases

def compile_model(model, lr):
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=CategoricalFocalCrossentropy(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
                  metrics=['accuracy'])

# ── Dataset (deduplicated, identical to 01_train_main.py) ─────────────────────
ARTIST_MATCHERS = {"Vincent_van_Gogh": lambda f: f.startswith("Vincent_van_Gogh_"), "Edgar_Degas": lambda f: f.startswith("Edgar_Degas_"), "Pablo_Picasso": lambda f: f.startswith("Pablo_Picasso_"), "Pierre-Auguste_Renoir": lambda f: f.startswith("Pierre-Auguste_Renoir_"), "Albrecht_Dürer": lambda f: f.startswith("Albrecht_D") and "rer_" in f, "Paul_Gauguin": lambda f: f.startswith("Paul_Gauguin_"), "Francisco_Goya": lambda f: f.startswith("Francisco_Goya_"), "Rembrandt": lambda f: f.startswith("Rembrandt_"), "Alfred_Sisley": lambda f: f.startswith("Alfred_Sisley_"), "Titian": lambda f: f.startswith("Titian_")}
records = []
for fname in os.listdir(img_dir):
    for artist, matcher in ARTIST_MATCHERS.items():
        if matcher(fname): records.append({"filename": fname, "filepath": os.path.join(img_dir, fname), "artist": artist}); break
df = pd.DataFrame(records)
hashes = {}
for _, row in df.iterrows():
    try: hashes[row['filepath']] = imagehash.phash(Image.open(row['filepath']).convert("RGB"))
    except: pass
parent = {p: p for p in hashes}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y): parent[find(x)] = find(y)
paths = list(hashes.keys())
for i in range(len(paths)):
    for j in range(i+1, len(paths)):
        if hashes[paths[i]] - hashes[paths[j]] <= 10: union(paths[i], paths[j])
groups = defaultdict(list)
for p in paths: groups[find(p)].append(p)
pts_rem = set()
for rep, members in groups.items():
    if len(members) > 1:
        for m in members:
            if m != rep: pts_rem.add(m)
df_clean = df[~df['filepath'].isin(pts_rem)].reset_index(drop=True)
assert len(df_clean) == 3793
le = LabelEncoder(); le.fit(sorted(df_clean['artist'].unique()))
classes = np.array(sorted(df_clean['artist'].unique()))

# ── Data loading with per-branch preprocessing and shared augmentation ───────
AUG = ImageDataGenerator(rotation_range=15, width_shift_range=0.10, height_shift_range=0.10,
                         shear_range=0.10, zoom_range=0.10, horizontal_flip=True, fill_mode='nearest')

def load_batch(df_batch, le, augment=False):
    n = len(df_batch)
    x_r50 = np.zeros((n, *IMG_SIZE_STD, 3), dtype=np.float32)
    x_eff = np.zeros((n, *IMG_SIZE_STD, 3), dtype=np.float32)
    x_inc = np.zeros((n, *IMG_SIZE_INC, 3), dtype=np.float32)
    x_xcp = np.zeros((n, *IMG_SIZE_INC, 3), dtype=np.float32)
    labels = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for i, (_, row) in enumerate(df_batch.iterrows()):
        img = Image.open(row['filepath']).convert('RGB')
        big = np.array(img.resize(IMG_SIZE_INC)).astype(np.float32)        # 299x299, 0..255
        if augment:
            big = AUG.random_transform(big)                                 # ONE transform shared by all branches
        small = np.array(Image.fromarray(big.astype(np.uint8)).resize(IMG_SIZE_STD)).astype(np.float32)
        x_r50[i] = resnet_preprocess(small.copy())
        x_eff[i] = effnet_preprocess(small.copy())     # identity: EfficientNetV2 rescales internally
        x_inc[i] = inception_preprocess(big.copy())
        x_xcp[i] = xception_preprocess(big.copy())
        labels[i, le.transform([row['artist']])[0]] = 1.0
    return [x_r50, x_eff, x_inc, x_xcp], labels

def data_generator(df, le, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    while True:
        idx = np.arange(len(df))
        if shuffle: np.random.shuffle(idx)
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[idx[start:start+batch_size]].reset_index(drop=True)
            yield load_batch(batch_df, le, augment=augment)

def predict_all(model, df, le, batch_size=BATCH_SIZE):
    preds = []
    for start in range(0, len(df), batch_size):
        X, _ = load_batch(df.iloc[start:start+batch_size].reset_index(drop=True), le)
        preds.append(model.predict(X, verbose=0))
    return np.vstack(preds)

# ── Training loop ─────────────────────────────────────────────────────────────
all_results = []
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed} - EnsArtNet Reproduction (FINE_TUNE={FINE_TUNE})\n{'='*60}")
    test_save = f"{SAVE_DIR}/ensartnet_seed{seed}_test.npz"
    if os.path.exists(test_save):
        data = np.load(test_save); all_results.append({'seed': seed, 'acc': float(data['acc'])*100, 'f1': float(data['f1'])*100})
        print(f"Loaded: acc={float(data['acc'])*100:.2f}%"); continue
    set_seed(seed)
    train, val, test = duplicate_aware_split(df_clean, seed=seed)
    test_labels = le.transform(test['artist'].values)
    cw_dict = dict(enumerate(compute_class_weight('balanced', classes=classes, y=train['artist'].values)))
    steps_tr = math.ceil(len(train) / BATCH_SIZE)
    steps_va = math.ceil(len(val) / BATCH_SIZE)

    model, bases = build_ensartnet()
    cb = [EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
          ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)]

    # Stage 1: backbones frozen, train fusion head (lr = 1e-5 as in the paper)
    compile_model(model, LR_STAGE1)
    model.fit(data_generator(train, le, BATCH_SIZE, shuffle=True, augment=True), steps_per_epoch=steps_tr,
              validation_data=data_generator(val, le, BATCH_SIZE, shuffle=False), validation_steps=steps_va,
              epochs=EPOCHS_STAGE1, class_weight=cw_dict, callbacks=cb, verbose=1)

    # Stage 2 (optional): unfreeze last N layers of each backbone, same N and lr as the proposed framework
    if FINE_TUNE:
        for name, base in bases.items():
            base.trainable = True
            for layer in base.layers[:-UNFREEZE[name]]:
                layer.trainable = False
        compile_model(model, LR_STAGE2)
        model.fit(data_generator(train, le, BATCH_SIZE, shuffle=True, augment=True), steps_per_epoch=steps_tr,
                  validation_data=data_generator(val, le, BATCH_SIZE, shuffle=False), validation_steps=steps_va,
                  epochs=EPOCHS_STAGE2, class_weight=cw_dict, callbacks=cb, verbose=1)

    preds = predict_all(model, test, le)
    acc = accuracy_score(test_labels, np.argmax(preds, axis=1)); f1 = f1_score(test_labels, np.argmax(preds, axis=1), average='macro')
    print(f"Seed {seed}: acc={acc*100:.2f}% f1={f1*100:.2f}%")
    np.savez(test_save, test=preds, acc=acc, f1=f1)
    all_results.append({'seed': seed, 'acc': acc*100, 'f1': f1*100})
    del model, bases; gc.collect(); tf.keras.backend.clear_session()

results_df = pd.DataFrame(all_results)
print(f"\nEnsArtNet (FINE_TUNE={FINE_TUNE}): {results_df['acc'].mean():.2f}% +/-{results_df['acc'].std(ddof=1):.2f}%")
results_df.to_csv("/kaggle/working/ensartnet_results.csv", index=False)
