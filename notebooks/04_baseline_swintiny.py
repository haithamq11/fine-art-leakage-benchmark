# Swin-Tiny Baseline — following Liu et al. [22]
# Requires: swin_tiny_patch4_window7_224.pth weights in Kaggle dataset
# Run on Kaggle GPU — approximately 4 hours for 5 seeds

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

SEEDS         = [42, 0, 1, 7, 123]
BATCH_SIZE    = 32
IMG_SIZE      = 224
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20
PATIENCE      = 10
N_FOLDS       = 5
SAVE_DIR      = "/kaggle/working/predictions_swin"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "/kaggle/input/datasets/haithamq11/modelsss/swin_tiny_patch4_window7_224.pth"
print(f"Device: {DEVICE}")

pred_input = "/kaggle/input/datasets/haithamq11/predictions-swin/"
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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def duplicate_aware_split(df, seed, test_size=0.15, val_size=0.15):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df['artist'], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), stratify=train_val['artist'], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(0.1,0.1,0.1,0.05), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
val_transform   = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])

class ArtDataset(Dataset):
    def __init__(self, df, labels, transform):
        self.filepaths = df['filepath'].values; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.filepaths[idx]).convert('RGB')), self.labels[idx]

def build_swin(num_classes=10, freeze=True):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if not k.startswith('head') and k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered); model.load_state_dict(model_dict)
    print(f"  Loaded {len(filtered)} layers")
    if freeze:
        for name, param in model.named_parameters():
            if not name.startswith('head'): param.requires_grad = False
    return model.to(DEVICE)

def unfreeze_last_layers(model, n_layers=2):
    for param in model.head.parameters(): param.requires_grad = True
    for layer in list(model.layers)[-n_layers:]:
        for param in layer.parameters(): param.requires_grad = True
    return model

def train_epoch(model, loader, optimizer, criterion):
    model.train(); total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(); loss = criterion(model(imgs), labels)
        loss.backward(); optimizer.step(); total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader, criterion):
    model.eval(); total = 0; preds, labs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs); total += criterion(out, labels).item()
            preds.append(torch.softmax(out,1).cpu().numpy()); labs.extend(labels.cpu().numpy())
    return total/len(loader), np.vstack(preds), np.array(labs)

def predict(model, loader):
    model.eval(); preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            preds.append(torch.softmax(model(imgs.to(DEVICE)),1).cpu().numpy())
    return np.vstack(preds)

def train_model(train_df, train_labels, val_df, val_labels, class_weights, seed):
    torch.manual_seed(seed)
    train_ds = ArtDataset(train_df, train_labels, train_transform)
    val_ds   = ArtDataset(val_df,   val_labels,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    cw = torch.FloatTensor([class_weights[i] for i in range(10)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    model = build_swin(freeze=True)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    best_loss = float('inf'); patience_cnt = 0; best_state = None
    for _ in range(EPOCHS_STAGE1):
        train_epoch(model, train_loader, opt, criterion)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion)
        sched.step(val_loss)
        if val_loss < best_loss: best_loss = val_loss; best_state = {k:v.clone() for k,v in model.state_dict().items()}; patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break
    model.load_state_dict(best_state)
    model = unfreeze_last_layers(model, n_layers=2)
    opt2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    best_loss2 = float('inf'); patience_cnt2 = 0; best_state2 = None
    for _ in range(EPOCHS_STAGE2):
        train_epoch(model, train_loader, opt2, criterion)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion)
        sched2.step(val_loss)
        if val_loss < best_loss2: best_loss2 = val_loss; best_state2 = {k:v.clone() for k,v in model.state_dict().items()}; patience_cnt2 = 0
        else:
            patience_cnt2 += 1
            if patience_cnt2 >= PATIENCE: break
    model.load_state_dict(best_state2)
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
        if matcher(fname): records.append({"filename":fname,"filepath":os.path.join(img_dir,fname),"artist":artist}); break
df = pd.DataFrame(records)
hashes = {}
for _, row in df.iterrows():
    try: hashes[row['filepath']] = imagehash.phash(Image.open(row['filepath']).convert("RGB"))
    except: pass
parent = {p:p for p in hashes}
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
paths_to_remove=set()
for rep,members in groups.items():
    if len(members)>1:
        for m in members:
            if m!=rep: paths_to_remove.add(m)
df_clean=df[~df['filepath'].isin(paths_to_remove)].reset_index(drop=True)
assert len(df_clean)==3793
le=LabelEncoder(); le.fit(sorted(df_clean['artist'].unique()))
classes=np.array(sorted(df_clean['artist'].unique()))

all_results=[]
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed} - Swin-Tiny\n{'='*60}")
    oof_save=f"{SAVE_DIR}/swin_seed{seed}_oof.npz"; test_save=f"{SAVE_DIR}/swin_seed{seed}_test.npz"
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
        fold_train=train.iloc[tr_idx].reset_index(drop=True); fold_oof=train.iloc[oof_idx].reset_index(drop=True)
        fold_train_inner,fold_inner_val=train_test_split(fold_train,test_size=0.15,stratify=fold_train['artist'],random_state=seed+fold)
        fold_ti_labels=le.transform(fold_train_inner['artist'].values); fold_iv_labels=le.transform(fold_inner_val['artist'].values)
        fold_oof_labels=le.transform(fold_oof['artist'].values)
        fold_cw=dict(enumerate(compute_class_weight('balanced',classes=classes,y=fold_train_inner['artist'].values)))
        model=train_model(fold_train_inner,fold_ti_labels,fold_inner_val,fold_iv_labels,fold_cw,seed+fold)
        oof_ds=ArtDataset(fold_oof,fold_oof_labels,val_transform); oof_loader=DataLoader(oof_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
        oof_preds[oof_idx]=predict(model,oof_loader)
        test_ds=ArtDataset(test,test_labels,val_transform); test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
        fold_test_preds.append(predict(model,test_loader))
        print(f"fold acc={accuracy_score(test_labels,np.argmax(fold_test_preds[-1],axis=1)):.4f}",flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    test_p=np.mean(fold_test_preds,axis=0)
    acc=accuracy_score(test_labels,np.argmax(test_p,axis=1)); f1=f1_score(test_labels,np.argmax(test_p,axis=1),average='macro')
    print(f"Seed {seed}: acc={acc*100:.2f}% f1={f1*100:.2f}%")
    np.savez(oof_save,oof=oof_preds); np.savez(test_save,test=test_p,acc=acc,f1=f1)
    all_results.append({'seed':seed,'acc':acc*100,'f1':f1*100})

results_df=pd.DataFrame(all_results)
print(f"\nSwin-Tiny: {results_df['acc'].mean():.2f}% +/-{results_df['acc'].std(ddof=1):.2f}%")
results_df.to_csv("/kaggle/working/swin_results.csv",index=False)
