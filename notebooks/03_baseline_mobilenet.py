# MobileNetV3Large Baseline — following Ma et al. [20] and Wang & Song [21]
# Run on Kaggle GPU — approximately 3 hours for 5 seeds

import subprocess
subprocess.run(["pip", "install", "imagehash", "-q"], check=True)

import os, gc, random, shutil, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
import imagehash
from collections import defaultdict
warnings.filterwarnings('ignore')

SEEDS         = [42, 0, 1, 7, 123]
BATCH_SIZE    = 32
IMG_SIZE      = (224, 224)
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20
PATIENCE      = 10
N_FOLDS       = 5
SAVE_DIR      = "/kaggle/working/predictions_mobilenet"
UNFREEZE_LAYERS = 30
os.makedirs(SAVE_DIR, exist_ok=True)

pred_input = "/kaggle/input/datasets/haithamqutaiba/predictions-mobilenet/"
if os.path.exists(pred_input):
    for f in os.listdir(pred_input):
        if f.endswith(".npz"):
            shutil.copy(os.path.join(pred_input, f), os.path.join(SAVE_DIR, f))
    print(f"Restored: {len(os.listdir(SAVE_DIR))} files")

img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir):
    img_dir = "/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def duplicate_aware_split(df, seed, test_size=0.15, val_size=0.15):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df['artist'], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), stratify=train_val['artist'], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def build_generators(train_df, val_df, test_df):
    aug = ImageDataGenerator(preprocessing_function=mobilenet_preprocess,
        rotation_range=15, width_shift_range=0.10, height_shift_range=0.10,
        shear_range=0.10, zoom_range=0.10, horizontal_flip=True, fill_mode='nearest')
    plain = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)
    kw = dict(x_col='filepath', y_col='artist', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    return (aug.flow_from_dataframe(train_df, shuffle=True,  **kw),
            plain.flow_from_dataframe(val_df,  shuffle=False, **kw),
            plain.flow_from_dataframe(test_df, shuffle=False, **kw))

def build_model():
    base = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Model(inputs=base.input, outputs=Dense(10, activation='softmax')(x)), base

def train_model(train_gen, inner_val_gen, class_weights, seed):
    tf.random.set_seed(seed)
    model, base = build_model()
    cb = [EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
          ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)]
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=inner_val_gen, epochs=EPOCHS_STAGE1, class_weight=class_weights, callbacks=cb, verbose=0)
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAYERS]: layer.trainable = False
    model.compile(optimizer=Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    train_gen.reset(); inner_val_gen.reset()
    model.fit(train_gen, validation_data=inner_val_gen, epochs=EPOCHS_STAGE2, class_weight=class_weights, callbacks=cb, verbose=0)
    return model

ARTIST_MATCHERS = {
    "Vincent_van_Gogh": lambda f: f.startswith("Vincent_van_Gogh_"),
    "Edgar_Degas": lambda f: f.startswith("Edgar_Degas_"),
    "Pablo_Picasso": lambda f: f.startswith("Pablo_Picasso_"),
    "Pierre-Auguste_Renoir": lambda f: f.startswith("Pierre-Auguste_Renoir_"),
    "Albrecht_Dürer": lambda f: f.startswith("Albrecht_D") and "rer_" in f,
    "Paul_Gauguin": lambda f: f.startswith("Paul_Gauguin_"),
    "Francisco_Goya": lambda f: f.startswith("Francisco_Goya_"),
    "Rembrandt": lambda f: f.startswith("Rembrandt_"),
    "Alfred_Sisley": lambda f: f.startswith("Alfred_Sisley_"),
    "Titian": lambda f: f.startswith("Titian_"),
}

records = []
for fname in os.listdir(img_dir):
    for artist, matcher in ARTIST_MATCHERS.items():
        if matcher(fname):
            records.append({"filename": fname, "filepath": os.path.join(img_dir, fname), "artist": artist}); break
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
paths_to_remove = set()
for rep, members in groups.items():
    if len(members) > 1:
        for m in members:
            if m != rep: paths_to_remove.add(m)
df_clean = df[~df['filepath'].isin(paths_to_remove)].reset_index(drop=True)
assert len(df_clean) == 3793
le = LabelEncoder(); le.fit(sorted(df_clean['artist'].unique()))
classes = np.array(sorted(df_clean['artist'].unique()))

all_results = []
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed} - MobileNetV3Large\n{'='*60}")
    oof_save  = f"{SAVE_DIR}/mobilenet_seed{seed}_oof.npz"
    test_save = f"{SAVE_DIR}/mobilenet_seed{seed}_test.npz"
    if os.path.exists(oof_save) and os.path.exists(test_save):
        data = np.load(test_save)
        all_results.append({'seed': seed, 'acc': float(data['acc'])*100, 'f1': float(data['f1'])*100})
        print(f"Loaded: acc={float(data['acc'])*100:.2f}%"); continue
    set_seed(seed)
    train, val, test = duplicate_aware_split(df_clean, seed=seed)
    train_labels = le.transform(train['artist'].values)
    test_labels  = le.transform(test['artist'].values)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(train), 10)); fold_test_preds = []
    for fold, (tr_idx, oof_idx) in enumerate(skf.split(train, train_labels)):
        print(f"  Fold {fold+1}/{N_FOLDS}...", end=' ', flush=True)
        fold_train = train.iloc[tr_idx].reset_index(drop=True)
        fold_oof   = train.iloc[oof_idx].reset_index(drop=True)
        fold_train_inner, fold_inner_val = train_test_split(fold_train, test_size=0.15, stratify=fold_train['artist'], random_state=seed+fold)
        fold_cw = dict(enumerate(compute_class_weight('balanced', classes=classes, y=fold_train_inner['artist'].values)))
        ti_gen, iv_gen, _ = build_generators(fold_train_inner, fold_inner_val, fold_inner_val)
        oof_gen  = ImageDataGenerator(preprocessing_function=mobilenet_preprocess).flow_from_dataframe(fold_oof, x_col='filepath', y_col='artist', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
        test_gen = ImageDataGenerator(preprocessing_function=mobilenet_preprocess).flow_from_dataframe(test, x_col='filepath', y_col='artist', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
        model = train_model(ti_gen, iv_gen, fold_cw, seed+fold)
        oof_gen.reset(); oof_preds[oof_idx] = model.predict(oof_gen, verbose=0)
        test_gen.reset(); fold_test_preds.append(model.predict(test_gen, verbose=0))
        print(f"fold acc={accuracy_score(test_labels, np.argmax(fold_test_preds[-1], axis=1)):.4f}", flush=True)
        del model; gc.collect(); tf.keras.backend.clear_session()
    test_p = np.mean(fold_test_preds, axis=0)
    acc = accuracy_score(test_labels, np.argmax(test_p, axis=1))
    f1  = f1_score(test_labels, np.argmax(test_p, axis=1), average='macro')
    print(f"Seed {seed}: acc={acc*100:.2f}% f1={f1*100:.2f}%")
    np.savez(oof_save, oof=oof_preds); np.savez(test_save, test=test_p, acc=acc, f1=f1)
    all_results.append({'seed': seed, 'acc': acc*100, 'f1': f1*100})

results_df = pd.DataFrame(all_results)
print(f"\nMobileNetV3Large: {results_df['acc'].mean():.2f}% +/-{results_df['acc'].std(ddof=1):.2f}%")
results_df.to_csv("/kaggle/working/mobilenet_results.csv", index=False)
