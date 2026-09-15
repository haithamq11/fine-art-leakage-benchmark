# EnsArtNet Reproduction [18] — Mezina & Burget 2025
# Architecture: 4 backbone branches + feature concatenation + focal loss
# Run on Kaggle GPU — approximately 4 hours for 5 seeds

import subprocess
subprocess.run(["pip", "install", "imagehash", "-q"], check=True)

import os, gc, random, shutil, warnings
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
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout,
                                      BatchNormalization, Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from PIL import Image
import imagehash
from collections import defaultdict
warnings.filterwarnings('ignore')

SEEDS         = [42, 0, 1, 7, 123]
BATCH_SIZE    = 32
IMG_SIZE_STD  = (224, 224)
IMG_SIZE_INC  = (299, 299)
EPOCHS        = 20
PATIENCE      = 10
SAVE_DIR      = "/kaggle/working/predictions_ensartnet"
os.makedirs(SAVE_DIR, exist_ok=True)

pred_input = "/kaggle/input/datasets/haithamq11/predictions-ensartnet/"
if os.path.exists(pred_input):
    for f in os.listdir(pred_input):
        if f.endswith(".npz"): shutil.copy(os.path.join(pred_input,f),os.path.join(SAVE_DIR,f))
    print(f"Restored: {len(os.listdir(SAVE_DIR))} files")

img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir): img_dir="/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def duplicate_aware_split(df,seed,test_size=0.15,val_size=0.15):
    tv,test=train_test_split(df,test_size=test_size,stratify=df['artist'],random_state=seed)
    train,val=train_test_split(tv,test_size=val_size/(1-test_size),stratify=tv['artist'],random_state=seed)
    return train.reset_index(drop=True),val.reset_index(drop=True),test.reset_index(drop=True)

def build_ensartnet(num_classes=10):
    """
    EnsArtNet following Mezina & Burget [18]:
    - 4 branches: ResNet50, EfficientNetV2B0, InceptionV3, Xception
    - Each: GAP -> Dense(256) -> BatchNorm -> Dropout(0.5)
    - Concatenate -> Dense(256) -> Dense(64) -> Dropout(0.3) -> Dense(10, softmax)
    - Loss: Focal (gamma=2.0, alpha=0.25), Optimizer: Adam (lr=1e-5)
    """
    input_std = Input(shape=(*IMG_SIZE_STD, 3))
    input_inc = Input(shape=(*IMG_SIZE_INC, 3))
    def branch(base, inp, name):
        base.trainable = False
        x = base(inp, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', name=f'd_{name}')(x)
        x = BatchNormalization(name=f'bn_{name}')(x)
        x = Dropout(0.5, name=f'dr_{name}')(x)
        return x
    b1 = branch(ResNet50(weights='imagenet',include_top=False,input_shape=(*IMG_SIZE_STD,3)),   input_std, 'r50')
    b2 = branch(EfficientNetV2B0(weights='imagenet',include_top=False,input_shape=(*IMG_SIZE_STD,3)), input_std, 'eff')
    b3 = branch(InceptionV3(weights='imagenet',include_top=False,input_shape=(*IMG_SIZE_INC,3)),input_inc, 'inc')
    b4 = branch(Xception(weights='imagenet',include_top=False,input_shape=(*IMG_SIZE_INC,3)),   input_inc, 'xcp')
    m = Concatenate()([b1,b2,b3,b4])
    x = Dense(256, activation='relu')(m)
    x = Dense(64,  activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=[input_std,input_inc], outputs=out)

ARTIST_MATCHERS={"Vincent_van_Gogh":lambda f:f.startswith("Vincent_van_Gogh_"),"Edgar_Degas":lambda f:f.startswith("Edgar_Degas_"),"Pablo_Picasso":lambda f:f.startswith("Pablo_Picasso_"),"Pierre-Auguste_Renoir":lambda f:f.startswith("Pierre-Auguste_Renoir_"),"Albrecht_Dürer":lambda f:f.startswith("Albrecht_D") and "rer_" in f,"Paul_Gauguin":lambda f:f.startswith("Paul_Gauguin_"),"Francisco_Goya":lambda f:f.startswith("Francisco_Goya_"),"Rembrandt":lambda f:f.startswith("Rembrandt_"),"Alfred_Sisley":lambda f:f.startswith("Alfred_Sisley_"),"Titian":lambda f:f.startswith("Titian_")}
records=[]
for fname in os.listdir(img_dir):
    for artist,matcher in ARTIST_MATCHERS.items():
        if matcher(fname): records.append({"filename":fname,"filepath":os.path.join(img_dir,fname),"artist":artist}); break
df=pd.DataFrame(records)
hashes={}
for _,row in df.iterrows():
    try: hashes[row['filepath']]=imagehash.phash(Image.open(row['filepath']).convert("RGB"))
    except: pass
parent={p:p for p in hashes}
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(x,y): parent[find(x)]=find(y)
paths=list(hashes.keys())
for i in range(len(paths)):
    for j in range(i+1,len(paths)):
        if hashes[paths[i]]-hashes[paths[j]]<=10: union(paths[i],paths[j])
groups=defaultdict(list)
for p in paths: groups[find(p)].append(p)
pts_rem=set()
for rep,members in groups.items():
    if len(members)>1:
        for m in members:
            if m!=rep: pts_rem.add(m)
df_clean=df[~df['filepath'].isin(pts_rem)].reset_index(drop=True)
assert len(df_clean)==3793
le=LabelEncoder(); le.fit(sorted(df_clean['artist'].unique()))
classes=np.array(sorted(df_clean['artist'].unique()))

def load_batch(df_batch, le):
    imgs_std=np.zeros((len(df_batch),*IMG_SIZE_STD,3)); imgs_inc=np.zeros((len(df_batch),*IMG_SIZE_INC,3)); labels=np.zeros((len(df_batch),10))
    for i,(_,row) in enumerate(df_batch.iterrows()):
        img=Image.open(row['filepath']).convert('RGB')
        imgs_std[i]=resnet_preprocess(np.array(img.resize(IMG_SIZE_STD)).astype(np.float32))
        imgs_inc[i]=inception_preprocess(np.array(img.resize(IMG_SIZE_INC)).astype(np.float32))
        labels[i,le.transform([row['artist']])[0]]=1.0
    return [imgs_std,imgs_inc],labels

def data_generator(df,le,batch_size=32,shuffle=True):
    while True:
        idx=np.arange(len(df))
        if shuffle: np.random.shuffle(idx)
        for start in range(0,len(df),batch_size):
            batch_df=df.iloc[idx[start:start+batch_size]].reset_index(drop=True)
            yield load_batch(batch_df,le)

all_results=[]
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed} - EnsArtNet Reproduction\n{'='*60}")
    test_save=f"{SAVE_DIR}/ensartnet_seed{seed}_test.npz"
    if os.path.exists(test_save):
        data=np.load(test_save); all_results.append({'seed':seed,'acc':float(data['acc'])*100,'f1':float(data['f1'])*100})
        print(f"Loaded: acc={float(data['acc'])*100:.2f}%"); continue
    set_seed(seed)
    train,val,test=duplicate_aware_split(df_clean,seed=seed)
    test_labels=le.transform(test['artist'].values)
    model=build_ensartnet(num_classes=10)
    model.compile(optimizer=Adam(lr=0.00001),loss=CategoricalFocalCrossentropy(gamma=2.0,alpha=0.25),metrics=['accuracy'])
    cw_array=compute_class_weight('balanced',classes=classes,y=train['artist'].values)
    cw_dict=dict(enumerate(cw_array))
    cb=[EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True),ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-7)]
    model.fit(data_generator(train,le,BATCH_SIZE,shuffle=True),steps_per_epoch=len(train)//BATCH_SIZE,
              validation_data=data_generator(val,le,BATCH_SIZE,shuffle=False),validation_steps=len(val)//BATCH_SIZE,
              epochs=EPOCHS,class_weight=cw_dict,callbacks=cb,verbose=1)
    X_test,_=load_batch(test,le)
    preds=model.predict(X_test,verbose=0)
    acc=accuracy_score(test_labels,np.argmax(preds,axis=1)); f1=f1_score(test_labels,np.argmax(preds,axis=1),average='macro')
    print(f"Seed {seed}: acc={acc*100:.2f}% f1={f1*100:.2f}%")
    np.savez(test_save,test=preds,acc=acc,f1=f1)
    all_results.append({'seed':seed,'acc':acc*100,'f1':f1*100})
    del model; gc.collect(); tf.keras.backend.clear_session()

results_df=pd.DataFrame(all_results)
print(f"\nEnsArtNet: {results_df['acc'].mean():.2f}% +/-{results_df['acc'].std(ddof=1):.2f}%")
results_df.to_csv("/kaggle/working/ensartnet_results.csv",index=False)
