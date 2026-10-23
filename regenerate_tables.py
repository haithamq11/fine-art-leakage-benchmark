# regenerate_tables.py
# Recomputes every statistic of the manuscript from the released per-image predictions (round-2 reproducibility item).
#   python regenerate_tables.py --pred predictions --data data --out regenerated
# Inputs : predictions/<backbone>_seed<seed>_{oof,test}.npz for resnet50, efficientnetv2b0, inceptionv3, xception, mobilenet,
#          convnext, swin; predictions/xception_nodup_seed<seed>_test.npz; data/split_assignments.csv
# Outputs: table6_per_seed.csv, table7_8_mcnemar_holm.csv, table9_ablation.csv, table11_baselines_per_seed.csv,
#          paired_confidence_intervals.csv  (all values match Tables 6-9 and 11 of the manuscript)
import argparse, os, unicodedata as ud, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
ap = argparse.ArgumentParser(); ap.add_argument('--pred', default='predictions'); ap.add_argument('--data', default='data'); ap.add_argument('--out', default='regenerated')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sp = pd.read_csv(os.path.join(a.data, 'split_assignments.csv')); sp['artist'] = sp['artist'].map(lambda s: ud.normalize('NFC', s))
classes = sorted(sp['artist'].unique()); cidx = {c: i for i, c in enumerate(classes)}
B = ['resnet50', 'efficientnetv2b0', 'inceptionv3', 'xception']; SEEDS = [42, 0, 1, 7, 123]
def load(name, seed, part): return np.load(os.path.join(a.pred, f'{name}_seed{seed}_{part}.npz'))[part]
def mcnemar(ac, bc):
    b = int(np.sum(ac & ~bc)); c = int(np.sum(~ac & bc)); n = b + c
    return b, c, (1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))
def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order): run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj
def tci(d): d = np.asarray(d); h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return d.mean(), d.mean() - h, d.mean() + h
t6, abl, mc, base = [], [], [], []
subsets = {'Full ensemble (4 backbones)': B, 'Without EfficientNetV2B0': [b for b in B if b != 'efficientnetv2b0'], 'Without InceptionV3': [b for b in B if b != 'inceptionv3'],
           'Without Xception': [b for b in B if b != 'xception'], 'Without ResNet50': [b for b in B if b != 'resnet50'], 'ResNet50 + Xception only': ['resnet50', 'xception']}
acc_sub = {k: [] for k in subsets}; acc = {k: [] for k in ('soft', 'convnext', 'swin', 'mobilenet')}
for seed in SEEDS:
    ytr = sp[(sp.seed == seed) & (sp.split == 'train')]['artist'].map(cidx).values; yte = sp[(sp.seed == seed) & (sp.split == 'test')]['artist'].map(cidx).values
    oof = {b: load(b, seed, 'oof') for b in B}; tst = {b: load(b, seed, 'test') for b in B}
    pred = {}
    for name, bs in subsets.items():
        m = LogisticRegression(max_iter=5000, C=1.0, random_state=seed).fit(np.concatenate([oof[b] for b in bs], 1), ytr)
        pred[name] = m.predict(np.concatenate([tst[b] for b in bs], 1)); acc_sub[name].append(accuracy_score(yte, pred[name]) * 100)
    ens = pred['Full ensemble (4 backbones)']; soft = np.argmax(np.mean([tst[b] for b in B], 0), 1)
    single = {b: np.argmax(tst[b], 1) for b in B}
    row = {'seed': seed, **{b: accuracy_score(yte, single[b]) * 100 for b in B}, 'soft_voting': accuracy_score(yte, soft) * 100, 'ensemble_acc': accuracy_score(yte, ens) * 100,
           'macro_f1': f1_score(yte, ens, average='macro') * 100, 'weighted_f1': f1_score(yte, ens, average='weighted') * 100}; t6.append(row); acc['soft'].append(row['soft_voting'])
    others = {}
    for name in ('convnext', 'swin', 'mobilenet'):
        try: others[name] = np.argmax(load(name, seed, 'test'), 1); acc[name].append(accuracy_score(yte, others[name]) * 100)
        except FileNotFoundError: pass
    corr = lambda p: p == yte
    fam = [('A', 'ensemble', 'resnet50', ens, single['resnet50']), ('A', 'ensemble', 'xception', ens, single['xception']), ('B', 'ensemble', 'soft_voting', ens, soft)]
    if 'convnext' in others: fam.append(('C', 'convnext', 'ensemble', others['convnext'], ens))
    if 'swin' in others: fam.append(('C', 'swin', 'ensemble', others['swin'], ens))
    fam.append(('D', 'two_backbone', 'ensemble', pred['ResNet50 + Xception only'], ens))
    for f, m_, b_, pm, pb in fam:
        b, c, p = mcnemar(corr(pm), corr(pb)); mc.append({'family': f, 'seed': seed, 'model': m_, 'baseline': b_, 'b': b, 'c': c, 'p': p})
t6 = pd.DataFrame(t6); t6.loc['mean'] = t6.mean(numeric_only=True); t6.loc['std'] = t6.iloc[:5].std(ddof=1, numeric_only=True); t6.round(2).to_csv(os.path.join(a.out, 'table6_per_seed.csv'))
mc = pd.DataFrame(mc); main = mc[mc.family.isin(list('ABC'))].copy(); main['p_holm_family'] = np.nan
for f in 'ABC':
    idx = main.family == f
    if idx.any(): main.loc[idx, 'p_holm_family'] = holm(main.loc[idx, 'p'])
main['p_holm_all'] = holm(main['p']); pd.concat([main, mc[mc.family == 'D']]).to_csv(os.path.join(a.out, 'table7_8_mcnemar_holm.csv'), index=False)
full = np.array(acc_sub['Full ensemble (4 backbones)']); rows = []
for name, v in acc_sub.items():
    v = np.array(v); m_, lo, hi = tci(v - full); rows.append({'configuration': name, 'mean_acc': v.mean(), 'std': v.std(ddof=1), 'vs_full_pp': m_, 'ci_low': lo, 'ci_high': hi})
soft = np.array(acc['soft']); m_, lo, hi = tci(soft - full); rows.append({'configuration': 'Soft voting (4 backbones)', 'mean_acc': soft.mean(), 'std': soft.std(ddof=1), 'vs_full_pp': m_, 'ci_low': lo, 'ci_high': hi})
pd.DataFrame(rows).round(2).to_csv(os.path.join(a.out, 'table9_ablation.csv'), index=False)
ci = [{'comparison': 'stacking - soft voting', **dict(zip(('mean', 'ci_low', 'ci_high'), tci(full - soft)))}]
for name in ('convnext', 'swin', 'mobilenet'):
    if len(acc[name]) == 5: ci.append({'comparison': f'{name} - ensemble', **dict(zip(('mean', 'ci_low', 'ci_high'), tci(np.array(acc[name]) - full)))})
pd.DataFrame(ci).round(2).to_csv(os.path.join(a.out, 'paired_confidence_intervals.csv'), index=False)
pd.DataFrame({'seed': SEEDS, **{k: v for k, v in acc.items() if len(v) == 5}, 'two_backbone': acc_sub['ResNet50 + Xception only'], 'ensemble': full}).round(2).to_csv(os.path.join(a.out, 'table11_baselines_per_seed.csv'), index=False)
print(open(os.path.join(a.out, 'table9_ablation.csv')).read()); print('written to', a.out)

# ---- Round-2 experiments (Tables 13-14): fixed-test-set leakage experiment (script 08) and duplicate-group-aware split (script 09) ----
rows = []
for seed in SEEDS:
    f = os.path.join(a.pred, f'fixed_testset_seed{seed}.npz')
    if not os.path.exists(f): continue
    d = np.load(f); y = d['y']; acc = {k: (d[k].argmax(1) == y).mean() * 100 for k in ('clean', 'dup', 'size')}
    rows.append({'seed': seed, 'acc_clean': acc['clean'], 'acc_dup': acc['dup'], 'acc_size': acc['size'], 'n_extra_dup': len(d['extras_B']), 'n_extra_control': len(d['extras_C']),
                 'dup_effect_pp': acc['dup'] - acc['clean'], 'size_effect_pp': acc['size'] - acc['clean'], 'dup_minus_size_pp': acc['dup'] - acc['size']})
if rows:
    t13 = pd.DataFrame(rows)
    if len(t13) > 1:
        m = {'seed': 'mean'}
        for c in ('acc_clean', 'acc_dup', 'acc_size'): m[c] = t13[c].mean()
        for c in ('dup_effect_pp', 'size_effect_pp', 'dup_minus_size_pp'):
            mu, lo, hi = tci(t13[c].values); m[c] = f'{mu:+.2f} [{lo:+.2f}, {hi:+.2f}]'
        t13 = pd.concat([t13.round(2), pd.DataFrame([m])])
    t13.to_csv(os.path.join(a.out, 'table13_fixed_testset.csv'), index=False); print(t13.to_string())
rows = []; ga_all = os.path.join(a.data, 'group_aware_split_assignments.csv')
if os.path.exists(ga_all):
    ga_all = pd.read_csv(ga_all)
    for seed in SEEDS:
        f = os.path.join(a.pred, f'group_aware_seed{seed}_test.npz')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True); y = d['y']; p = d['test'].argmax(1); groups = set(ga_all[ga_all.seed == seed]['group'])
        rep = np.array([ud.normalize('NFC', str(fn)) in groups for fn in d['test_files']])
        xcp = float(t6.loc[t6.seed == seed, 'xception'].iloc[0]) if (t6.seed == seed).any() else np.nan
        rows.append({'seed': seed, 'n_test': len(y), 'acc_all': accuracy_score(y, p) * 100, 'f1_all': f1_score(y, p, average='macro') * 100, 'n_representatives': int(rep.sum()),
                     'acc_representatives': accuracy_score(y[rep], p[rep]) * 100, 'deletion_policy_xception': xcp, 'diff_pp': accuracy_score(y[rep], p[rep]) * 100 - xcp})
if rows:
    t14 = pd.DataFrame(rows)
    if len(t14) > 1:
        m = {'seed': 'mean', 'n_test': t14.n_test.mean(), 'n_representatives': t14.n_representatives.mean()}
        for c in ('acc_all', 'f1_all', 'acc_representatives', 'deletion_policy_xception'): m[c] = f'{t14[c].mean():.2f} +/- {t14[c].std(ddof=1):.2f}'
        mu, lo, hi = tci(t14['diff_pp'].values); m['diff_pp'] = f'{mu:+.2f} [{lo:+.2f}, {hi:+.2f}]'
        t14 = pd.concat([t14.round(2), pd.DataFrame([m])])
    t14.to_csv(os.path.join(a.out, 'table14_group_aware.csv'), index=False); print(t14.to_string())
