# ConvNeXt-Tiny Baseline [28]
# Run on Kaggle GPU — approximately 3 hours for 5 seeds

import subprocess
subprocess.run(["pip", "install", "imagehash", "-q"], check=True)

import os, gc, random, shutil, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import imagehash
from collections import defaultdict
warnings.filterwarnings('ignore')

SEEDS=[ 42,0,1,7,123]; BATCH_SIZE=32; IMG_SIZE=224
EPOCHS_STAGE1=20; EPOCHS_STAGE2=20; PATIENCE=10; N_FOLDS=5
SAVE_DIR="/kaggle/working/predictions_convnext"; os.makedirs(SAVE_DIR,exist_ok=True)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

pred_input="/kaggle/input/datasets/haithamq11/predictions-convnext/"
if os.path.exists(pred_input):
    for f in os.listdir(pred_input):
        if f.endswith(".npz"): shutil.copy(os.path.join(pred_input,f),os.path.join(SAVE_DIR,f))
    print(f"Restored: {len(os.listdir(SAVE_DIR))} files")

img_dir="/kaggle/input/best-artworks-of-all-time/resized/resized"
if not os.path.exists(img_dir): img_dir="/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def duplicate_aware_split(df,seed,test_size=0.15,val_size=0.15):
    tv,test=train_test_split(df,test_size=test_size,stratify=df['artist'],random_state=seed)
    train,val=train_test_split(tv,test_size=val_size/(1-test_size),stratify=tv['artist'],random_state=seed)
    return train.reset_index(drop=True),val.reset_index(drop=True),test.reset_index(drop=True)

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
train_transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(15),transforms.ColorJitter(0.1,0.1,0.1,0.05),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
val_transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])

class ArtDataset(Dataset):
    def __init__(self,df,labels,transform): self.filepaths=df['filepath'].values; self.labels=labels; self.transform=transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self,idx): return self.transform(Image.open(self.filepaths[idx]).convert('RGB')),self.labels[idx]

def build_convnext(num_classes=10,freeze=True):
    model=timm.create_model('convnext_tiny',pretrained=True,num_classes=num_classes)
    if freeze:
        for name,param in model.named_parameters():
            if not name.startswith('head'): param.requires_grad=False
    return model.to(DEVICE)

def unfreeze_last_layers(model,n_stages=1):
    for param in model.head.parameters(): param.requires_grad=True
    for stage in list(model.stages)[-n_stages:]:
        for param in stage.parameters(): param.requires_grad=True
    return model

def train_epoch(model,loader,opt,criterion):
    model.train(); total=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE); opt.zero_grad()
        loss=criterion(model(imgs),labels); loss.backward(); opt.step(); total+=loss.item()
    return total/len(loader)

def eval_epoch(model,loader,criterion):
    model.eval(); total=0; preds,labs=[],[]
    with torch.no_grad():
        for imgs,labels in loader:
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE); out=model(imgs)
            total+=criterion(out,labels).item(); preds.append(torch.softmax(out,1).cpu().numpy()); labs.extend(labels.cpu().numpy())
    return total/len(loader),np.vstack(preds),np.array(labs)

def predict(model,loader):
    model.eval(); preds=[]
    with torch.no_grad():
        for imgs,_ in loader: preds.append(torch.softmax(model(imgs.to(DEVICE)),1).cpu().numpy())
    return np.vstack(preds)

def train_model(train_df,train_labels,val_df,val_labels,class_weights,seed):
    torch.manual_seed(seed)
    train_ds=ArtDataset(train_df,train_labels,train_transform); val_ds=ArtDataset(val_df,val_labels,val_transform)
    tl=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
    vl=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)
    cw=torch.FloatTensor([class_weights[i] for i in range(10)]).to(DEVICE)
    criterion=nn.CrossEntropyLoss(weight=cw)
    model=build_convnext(freeze=True)
    opt=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=1e-4)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=5,min_lr=1e-7)
    best=float('inf'); pc=0; bs=None
    for _ in range(EPOCHS_STAGE1):
        train_epoch(model,tl,opt,criterion); vl_,_,_=eval_epoch(model,vl,criterion); sched.step(vl_)
        if vl_<best: best=vl_; bs={k:v.clone() for k,v in model.state_dict().items()}; pc=0
        else: pc+=1;
        if pc>=PATIENCE: break
    model.load_state_dict(bs); model=unfreeze_last_layers(model,n_stages=1)
    opt2=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=5e-5)
    sched2=optim.lr_scheduler.ReduceLROnPlateau(opt2,mode='min',factor=0.5,patience=5,min_lr=1e-7)
    best2=float('inf'); pc2=0; bs2=None
    for _ in range(EPOCHS_STAGE2):
        train_epoch(model,tl,opt2,criterion); vl_,_,_=eval_epoch(model,vl,criterion); sched2.step(vl_)
        if vl_<best2: best2=vl_; bs2={k:v.clone() for k,v in model.state_dict().items()}; pc2=0
        else: pc2+=1;
        if pc2>=PATIENCE: break
    model.load_state_dict(bs2); return model

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

all_results=[]
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed} - ConvNeXt-Tiny\n{'='*60}")
    oof_save=f"{SAVE_DIR}/convnext_seed{seed}_oof.npz"; test_save=f"{SAVE_DIR}/convnext_seed{seed}_test.npz"
    if os.path.exists(oof_save) and os.path.exists(test_save):
        data=np.load(test_save); all_results.append({'seed':seed,'acc':float(data['acc'])*100,'f1':float(data['f1'])*100})
        print(f"Loaded: acc={float(data['acc'])*100:.2f}%"); continue
    set_seed(seed)
    train,val,test=duplicate_aware_split(df_clean,seed=seed)
    train_labels=le.transform(train['artist'].values); test_labels=le.transform(test['artist'].values)
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)
    oof_preds=np.zeros((len(train),10)); fold_test_preds=[]
    for fold,(tr_idx,oof_idx) in enumerate(skf.split(train,train_labels)):
        print(f"  Fold {fold+1}/{N_FOLDS}...",end=' ',flush=True)
        ft=train.iloc[tr_idx].reset_index(drop=True); fo=train.iloc[oof_idx].reset_index(drop=True)
        fti,fiv=train_test_split(ft,test_size=0.15,stratify=ft['artist'],random_state=seed+fold)
        fti_l=le.transform(fti['artist'].values); fiv_l=le.transform(fiv['artist'].values); fo_l=le.transform(fo['artist'].values)
        fcw=dict(enumerate(compute_class_weight('balanced',classes=classes,y=fti['artist'].values)))
        model=train_model(fti,fti_l,fiv,fiv_l,fcw,seed+fold)
        ood=ArtDataset(fo,fo_l,val_transform); ool=DataLoader(ood,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
        oof_preds[oof_idx]=predict(model,ool)
        tsd=ArtDataset(test,test_labels,val_transform); tsl=DataLoader(tsd,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
        fold_test_preds.append(predict(model,tsl))
        print(f"fold acc={accuracy_score(test_labels,np.argmax(fold_test_preds[-1],axis=1)):.4f}",flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    tp=np.mean(fold_test_preds,axis=0)
    acc=accuracy_score(test_labels,np.argmax(tp,axis=1)); f1=f1_score(test_labels,np.argmax(tp,axis=1),average='macro')
    print(f"Seed {seed}: acc={acc*100:.2f}% f1={f1*100:.2f}%")
    np.savez(oof_save,oof=oof_preds); np.savez(test_save,test=tp,acc=acc,f1=f1)
    all_results.append({'seed':seed,'acc':acc*100,'f1':f1*100})

results_df=pd.DataFrame(all_results)
print(f"\nConvNeXt-Tiny: {results_df['acc'].mean():.2f}% +/-{results_df['acc'].std(ddof=1):.2f}%")
results_df.to_csv("/kaggle/working/convnext_results.csv",index=False)
