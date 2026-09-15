import subprocess
subprocess.run(["pip", "install", "imagehash", "-q"], check=True)

import os, gc, random, shutil, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetV2B0, InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
import imagehash
from collections import defaultdict
warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
SEEDS         = [42, 0, 1, 7, 123]
BATCH_SIZE    = 32
IMG_SIZE_STD  = (224, 224)
IMG_SIZE_INC  = (299, 299)
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20
PATIENCE      = 10
N_FOLDS       = 5
SAVE_DIR      = "/kaggle/working/predictions_v3"
os.makedirs(SAVE_DIR, exist_ok=True)

UNFREEZE = {
    "resnet50":         30,
    "efficientnetv2b0": 20,
    "inceptionv3":      30,
    "xception":         30,
}

BACKBONES = [
    ("resnet50",         IMG_SIZE_STD, resnet_preprocess),
    ("efficientnetv2b0", IMG_SIZE_STD, effnet_preprocess),
    ("inceptionv3",      IMG_SIZE_INC, inception_preprocess),
    ("xception",         IMG_SIZE_INC, xception_preprocess),
]

# ── Verify dataset path ───────────────────────────────────────────────────────
img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir):
    alt = "/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"
    if os.path.exists(alt):
        img_dir = alt
    else:
        raise FileNotFoundError("Dataset not found.")
print(f"Dataset path: {img_dir}")

# ── Helper functions ──────────────────────────────────────────────────────────
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

def build_generators(train_df, val_df, test_df, img_size, preprocess_fn):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15, width_shift_range=0.10,
        height_shift_range=0.10, shear_range=0.10,
        zoom_range=0.10, horizontal_flip=True, fill_mode='nearest')
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='artist',
        target_size=img_size, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True)
    val_gen = val_test_datagen.flow_from_dataframe(
        val_df, x_col='filepath', y_col='artist',
        target_size=img_size, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False)
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df, x_col='filepath', y_col='artist',
        target_size=img_size, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False)
    return train_gen, val_gen, test_gen

def build_backbone(backbone_name, img_size):
    if backbone_name == "resnet50":
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    elif backbone_name == "efficientnetv2b0":
        base = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    elif backbone_name == "inceptionv3":
        base = InceptionV3(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    elif backbone_name == "xception":
        base = Xception(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    return model, base

def train_backbone(backbone_name, img_size, train_gen, inner_val_gen, class_weights, seed):
    """
    Leakage-free OOF protocol:
    - train_gen: fold_train_inner only
    - inner_val_gen: carved from fold_train_inner only — used for early stopping
    - fold_oof: NEVER passed here — used only for OOF predictions after training
    """
    tf.random.set_seed(seed)
    model, base = build_backbone(backbone_name, img_size)
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
    for layer in base.layers[:-UNFREEZE[backbone_name]]:
        layer.trainable = False
    model.compile(optimizer=Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    train_gen.reset()
    inner_val_gen.reset()
    model.fit(train_gen, validation_data=inner_val_gen, epochs=EPOCHS_STAGE2,
              class_weight=class_weights, callbacks=callbacks, verbose=0)
    return model

# ── Build clean dataset ───────────────────────────────────────────────────────
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
print(f"Raw images: {len(df)}")

print("Computing pHash...")
hashes = {}
for _, row in df.iterrows():
    try:
        hashes[row['filepath']] = imagehash.phash(
            Image.open(row['filepath']).convert("RGB"))
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
        if hashes[paths[i]] - hashes[paths[j]] <= 10:
            union(paths[i], paths[j])

groups = defaultdict(list)
for p in paths: groups[find(p)].append(p)
paths_to_remove = set()
for rep, members in groups.items():
    if len(members) > 1:
        for m in members:
            if m != rep: paths_to_remove.add(m)

df_clean = df[~df['filepath'].isin(paths_to_remove)].reset_index(drop=True)
assert len(df_clean) == 3793, f"Expected 3793, got {len(df_clean)}"
print(f"Clean dataset: {len(df_clean)} images confirmed.")

all_results = []

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    set_seed(seed)
    train, val, test = duplicate_aware_split(df_clean, seed=seed)
    classes    = np.array(sorted(df_clean['artist'].unique()))
    le         = LabelEncoder()
    le.fit(sorted(df_clean['artist'].unique()))
    train_labels = le.transform(train['artist'].values)
    test_labels  = le.transform(test['artist'].values)

    oof_preds  = {b[0]: np.zeros((len(train), 10)) for b in BACKBONES}
    test_preds = {b[0]: np.zeros((len(test),  10)) for b in BACKBONES}
    individual_accs = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for backbone_name, img_size, preprocess_fn in BACKBONES:
        oof_save  = f"{SAVE_DIR}/{backbone_name}_seed{seed}_oof.npz"
        test_save = f"{SAVE_DIR}/{backbone_name}_seed{seed}_test.npz"

        if os.path.exists(oof_save) and os.path.exists(test_save):
            oof_preds[backbone_name]       = np.load(oof_save)['oof']
            test_preds[backbone_name]      = np.load(test_save)['test']
            individual_accs[backbone_name] = float(np.load(test_save)['acc'])
            print(f"  [{backbone_name}] Loaded. acc={individual_accs[backbone_name]:.4f}")
            continue

        print(f"\n  [{backbone_name}] Running {N_FOLDS}-fold OOF...")
        fold_test_preds = []

        for fold, (tr_idx, oof_idx) in enumerate(skf.split(train, train_labels)):
            print(f"    Fold {fold+1}/{N_FOLDS}...", end=' ', flush=True)
            fold_train = train.iloc[tr_idx].reset_index(drop=True)
            fold_oof   = train.iloc[oof_idx].reset_index(drop=True)

            # Inner val carved from fold_train only — leakage-free
            fold_train_inner, fold_inner_val = train_test_split(
                fold_train, test_size=0.15,
                stratify=fold_train['artist'], random_state=seed + fold)

            fold_cw = dict(enumerate(compute_class_weight(
                'balanced', classes=classes,
                y=fold_train_inner['artist'].values)))

            fold_train_inner_gen, fold_inner_val_gen, _ = build_generators(
                fold_train_inner, fold_inner_val, fold_inner_val,
                img_size, preprocess_fn)

            oof_gen = ImageDataGenerator(
                preprocessing_function=preprocess_fn).flow_from_dataframe(
                fold_oof, x_col='filepath', y_col='artist',
                target_size=img_size, batch_size=BATCH_SIZE,
                class_mode='categorical', shuffle=False)

            test_gen_fold = ImageDataGenerator(
                preprocessing_function=preprocess_fn).flow_from_dataframe(
                test, x_col='filepath', y_col='artist',
                target_size=img_size, batch_size=BATCH_SIZE,
                class_mode='categorical', shuffle=False)

            # fold_oof NEVER passed to train_backbone
            model = train_backbone(backbone_name, img_size,
                                   fold_train_inner_gen, fold_inner_val_gen,
                                   fold_cw, seed + fold)

            oof_gen.reset()
            oof_preds[backbone_name][oof_idx] = model.predict(oof_gen, verbose=0)

            test_gen_fold.reset()
            fold_test_preds.append(model.predict(test_gen_fold, verbose=0))

            fold_acc = accuracy_score(test_labels, np.argmax(fold_test_preds[-1], axis=1))
            print(f"fold acc={fold_acc:.4f}", flush=True)
            del model; gc.collect(); tf.keras.backend.clear_session()

        test_preds[backbone_name] = np.mean(fold_test_preds, axis=0)
        individual_accs[backbone_name] = accuracy_score(
            test_labels, np.argmax(test_preds[backbone_name], axis=1))
        np.savez(oof_save,  oof=oof_preds[backbone_name])
        np.savez(test_save, test=test_preds[backbone_name],
                 acc=individual_accs[backbone_name])
        print(f"  [{backbone_name}] Done. acc={individual_accs[backbone_name]:.4f}")

    # Stacking
    X_train_meta = np.concatenate([oof_preds[b[0]]  for b in BACKBONES], axis=1)
    X_test_meta  = np.concatenate([test_preds[b[0]] for b in BACKBONES], axis=1)
    meta = LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
    meta.fit(X_train_meta, train_labels)

    ens_preds  = meta.predict(X_test_meta)
    acc_ens    = accuracy_score(test_labels, ens_preds)
    f1_macro   = f1_score(test_labels, ens_preds, average='macro')
    f1_weighted= f1_score(test_labels, ens_preds, average='weighted')
    avg_probs  = np.mean([test_preds[b[0]] for b in BACKBONES], axis=0)
    acc_soft   = accuracy_score(test_labels, np.argmax(avg_probs, axis=1))

    print(f"\n  Seed {seed}: ensemble={acc_ens*100:.2f}% soft={acc_soft*100:.2f}%")
    all_results.append({
        'seed': seed,
        'resnet50': individual_accs['resnet50'],
        'efficientnetv2b0': individual_accs['efficientnetv2b0'],
        'inceptionv3': individual_accs['inceptionv3'],
        'xception': individual_accs['xception'],
        'soft_voting': acc_soft,
        'ensemble_acc': acc_ens,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    })

results_df = pd.DataFrame(all_results)
print(f"\n{'='*60}\nALL SEEDS COMPLETE\n{'='*60}")
print(results_df.to_string(index=False))
results_df.to_csv("/kaggle/working/all_seeds_results.csv", index=False)
print(f"\nSaved: all_seeds_results.csv")
print(f"Download all .npz files from: {SAVE_DIR}")
