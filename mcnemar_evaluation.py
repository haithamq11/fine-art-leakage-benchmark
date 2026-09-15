"""
Standalone McNemar evaluation script.
Run after 01_train_main.py to reproduce all statistical tests.
Requires: predictions saved in SAVE_DIR
"""
import os, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
from PIL import Image
import imagehash
from collections import defaultdict

SAVE_DIR  = "/kaggle/working/predictions_v3"
SEEDS     = [42, 0, 1, 7, 123]
BACKBONES = ['resnet50', 'efficientnetv2b0', 'inceptionv3', 'xception']

img_dir = "/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir): img_dir="/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"

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
le=LabelEncoder(); le.fit(sorted(df_clean['artist'].unique()))

def duplicate_aware_split(df,seed,test_size=0.15,val_size=0.15):
    tv,test=train_test_split(df,test_size=test_size,stratify=df['artist'],random_state=seed)
    train,val=train_test_split(tv,test_size=val_size/(1-test_size),stratify=tv['artist'],random_state=seed)
    return train.reset_index(drop=True),val.reset_index(drop=True),test.reset_index(drop=True)

print("="*60)
print("McNEMAR TESTS — Ensemble vs ResNet50 and Xception")
print("="*60)
gains=[]
for seed in SEEDS:
    train,val,test=duplicate_aware_split(df_clean,seed=seed)
    train_labels=le.transform(train['artist'].values); test_labels=le.transform(test['artist'].values)
    oof_p={b:np.load(f"{SAVE_DIR}/{b}_seed{seed}_oof.npz")['oof'] for b in BACKBONES}
    tst_p={b:np.load(f"{SAVE_DIR}/{b}_seed{seed}_test.npz")['test'] for b in BACKBONES}
    X_tr=np.concatenate([oof_p[b] for b in BACKBONES],axis=1); X_te=np.concatenate([tst_p[b] for b in BACKBONES],axis=1)
    meta=LogisticRegression(max_iter=5000,C=1.0,random_state=seed); meta.fit(X_tr,train_labels)
    ens=meta.predict(X_te)
    r50=np.argmax(tst_p['resnet50'],axis=1); xcp=np.argmax(tst_p['xception'],axis=1)
    soft=np.argmax(np.mean([tst_p[b] for b in BACKBONES],axis=0),axis=1)
    b_r=int(np.sum((ens==test_labels)&(r50!=test_labels))); c_r=int(np.sum((ens!=test_labels)&(r50==test_labels)))
    b_x=int(np.sum((ens==test_labels)&(xcp!=test_labels))); c_x=int(np.sum((ens!=test_labels)&(xcp==test_labels)))
    b_s=int(np.sum((ens==test_labels)&(soft!=test_labels))); c_s=int(np.sum((ens!=test_labels)&(soft==test_labels)))
    p_r=mcnemar([[0,b_r],[c_r,0]],exact=True).pvalue
    p_x=mcnemar([[0,b_x],[c_x,0]],exact=True).pvalue
    p_s=mcnemar([[0,b_s],[c_s,0]],exact=True).pvalue
    ens_acc=accuracy_score(test_labels,ens)*100; soft_acc=accuracy_score(test_labels,soft)*100
    gains.append(ens_acc-soft_acc)
    print(f"Seed {seed}: vs_R50 b={b_r} c={c_r} p={p_r:.6f} {'*' if p_r<0.05 else ' '} | vs_Xcp b={b_x} c={c_x} p={p_x:.6f} {'*' if p_x<0.05 else ' '} | vs_Soft b={b_s} c={c_s} p={p_s:.6f} {'*' if p_s<0.05 else ' '}")

gains=np.array(gains); n=len(gains); se=np.std(gains,ddof=1)/np.sqrt(n)
t_crit=stats.t.ppf(0.975,df=n-1)
ci_lo,ci_hi=np.mean(gains)-t_crit*se,np.mean(gains)+t_crit*se
print(f"\n95% CI for stacking gain over soft voting: [{ci_lo:.2f}, {ci_hi:.2f}]pp")
print("Done.")
