# Leakage Isolation Experiment
# Xception trained WITHOUT deduplication — same protocol, same seeds, same splits
# This isolates the contribution of duplicate removal from other protocol differences
# Result: 90.17% (+/-1.04%) WITH duplicates vs 86.57% (+/-1.48%) WITHOUT = 3.60pp isolated leakage effect

import subprocess
subprocess.run(["pip", "install", "imagehash", "-q"], check=True)

import os, gc, random, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
warnings.filterwarnings('ignore')

SEEDS         = [42, 0, 1, 7, 123]
BATCH_SIZE    = 32
IMG_SIZE      = (299, 299)
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20
PATIENCE      = 10
UNFREEZE      = 30
SAVE_DIR      = "/kaggle/working/predictions_xception_nodup"
os.makedirs(SAVE_DIR, exist_ok=True)

img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir): img_dir="/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"
print(f"Dataset path: {img_dir}")

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def duplicate_aware_split(df,seed,test_size=0.15,val_size=0.15):
    tv,test=train_test_split(df,test_size=test_size,stratify=df['artist'],random_state=seed)
    train,val=train_test_split(tv,test_size=val_size/(1-test_size),stratify=tv['artist'],random_state=seed)
    return train.reset_index(drop=True),val.reset_index(drop=True),test.reset_index(drop=True)

def build_generators(train_df,val_df,test_df,classes):
    aug=ImageDataGenerator(preprocessing_function=xception_preprocess,rotation_range=15,width_shift_range=0.10,height_shift_range=0.10,shear_range=0.10,zoom_range=0.10,horizontal_flip=True,fill_mode='nearest')
    plain=ImageDataGenerator(preprocessing_function=xception_preprocess)
    kw=dict(x_col='filepath',y_col='artist',target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode='categorical',classes=classes)
    return (aug.flow_from_dataframe(train_df,shuffle=True,**kw),
            plain.flow_from_dataframe(val_df,shuffle=False,**kw),
            plain.flow_from_dataframe(test_df,shuffle=False,**kw))

def build_model():
    base=Xception(weights='imagenet',include_top=False,input_shape=(*IMG_SIZE,3))
    base.trainable=False
    x=GlobalAveragePooling2D()(base.output)
    x=Dense(256,activation='relu')(x); x=Dropout(0.5)(x)
    return Model(inputs=base.input,outputs=Dense(10,activation='softmax')(x)),base

def train_model(train_gen,val_gen,class_weights,seed):
    tf.random.set_seed(seed)
    model,base=build_model()
    cb=[EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True),ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-7)]
    model.compile(optimizer=Adam(1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_gen,validation_data=val_gen,epochs=EPOCHS_STAGE1,class_weight=class_weights,callbacks=cb,verbose=0)
    base.trainable=True
    for layer in base.layers[:-UNFREEZE]: layer.trainable=False
    model.compile(optimizer=Adam(5e-5),loss='categorical_crossentropy',metrics=['accuracy'])
    train_gen.reset(); val_gen.reset()
    model.fit(train_gen,validation_data=val_gen,epochs=EPOCHS_STAGE2,class_weight=class_weights,callbacks=cb,verbose=0)
    return model

# NON-deduplicated dataset — NO pHash removal
ARTIST_MATCHERS={"Vincent_van_Gogh":lambda f:f.startswith("Vincent_van_Gogh_"),"Edgar_Degas":lambda f:f.startswith("Edgar_Degas_"),"Pablo_Picasso":lambda f:f.startswith("Pablo_Picasso_"),"Pierre-Auguste_Renoir":lambda f:f.startswith("Pierre-Auguste_Renoir_"),"Albrecht_Dürer":lambda f:f.startswith("Albrecht_D") and "rer_" in f,"Paul_Gauguin":lambda f:f.startswith("Paul_Gauguin_"),"Francisco_Goya":lambda f:f.startswith("Francisco_Goya_"),"Rembrandt":lambda f:f.startswith("Rembrandt_"),"Alfred_Sisley":lambda f:f.startswith("Alfred_Sisley_"),"Titian":lambda f:f.startswith("Titian_")}
records=[]
for fname in os.listdir(img_dir):
    for artist,matcher in ARTIST_MATCHERS.items():
        if matcher(fname): records.append({"filename":fname,"filepath":os.path.join(img_dir,fname),"artist":artist}); break
df=pd.DataFrame(records)
print(f"NON-deduplicated dataset: {len(df)} images (duplicates retained)")
le=LabelEncoder(); le.fit(sorted(df['artist'].unique()))
classes=sorted(df['artist'].unique())

all_results=[]
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed} - Xception WITHOUT deduplication\n{'='*60}")
    test_save=f"{SAVE_DIR}/xception_nodup_seed{seed}_test.npz"
    if os.path.exists(test_save):
        data=np.load(test_save); all_results.append({'seed':seed,'acc':float(data['acc'])*100,'f1':float(data['f1'])*100})
        print(f"Loaded: acc={float(data['acc'])*100:.2f}%"); continue
    set_seed(seed)
    train,val,test=duplicate_aware_split(df,seed=seed)
    test_labels=le.transform(test['artist'].values)
    fold_cw=dict(enumerate(compute_class_weight('balanced',classes=np.array(classes),y=train['artist'].values)))
    train_gen,val_gen,test_gen=build_generators(train,val,test,classes)
    model=train_model(train_gen,val_gen,fold_cw,seed)
    test_gen.reset(); preds=model.predict(test_gen,verbose=0)
    acc=accuracy_score(test_labels,np.argmax(preds,axis=1)); f1=f1_score(test_labels,np.argmax(preds,axis=1),average='macro')
    print(f"Seed {seed}: acc={acc*100:.2f}% f1={f1*100:.2f}%")
    np.savez(test_save,test=preds,acc=acc,f1=f1)
    all_results.append({'seed':seed,'acc':acc*100,'f1':f1*100})
    del model; gc.collect(); tf.keras.backend.clear_session()

results_df=pd.DataFrame(all_results)
print(f"\nXception WITHOUT deduplication: {results_df['acc'].mean():.2f}% +/-{results_df['acc'].std(ddof=1):.2f}%")
print(f"Xception WITH deduplication:    86.57% +/-1.48%")
print(f"Isolated leakage effect:        {results_df['acc'].mean()-86.57:.2f}pp")
results_df.to_csv("/kaggle/working/xception_nodup_results.csv",index=False)
