# Turn the Kaggle notebooks into local scripts that read the dataset from F:\kagglehub
# and write all outputs under F:\runs. Usage:  python localize.py
# Source scripts are NOT modified; localized copies go to F:\runs\scripts\.
import os, re, sys, platform

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks')
RUNS_WIN = 'F:/runs'
OUT = os.path.join(RUNS_WIN, 'scripts')
os.makedirs(OUT, exist_ok=True)

HEADER = '''# ---- local environment header (added by localize.py) ----
import os as _os, glob as _glob, platform as _platform
_WIN = _platform.system() == 'Windows'
_ROOT = _os.environ.get('ART_ROOT', 'F:/kagglehub' if _WIN else '/mnt/f/kagglehub')
_RUNS = _os.environ.get('ART_RUNS', 'F:/runs' if _WIN else '/mnt/f/runs')
_c = sorted(_glob.glob(f'{_ROOT}/datasets/ikarus777/best-artworks-of-all-time/versions/*/resized/resized'))
if not _c:
    raise FileNotFoundError(f'Dataset not found under {_ROOT}; run the kagglehub download first.')
LOCAL_IMG_DIR = _c[-1]
_NW = 0 if _WIN else 2          # DataLoader workers: 0 on Windows (no fork), 2 on Linux
_os.makedirs(_RUNS, exist_ok=True)
print('LOCAL_IMG_DIR =', LOCAL_IMG_DIR)
# ---- end header ----
'''

TF_GROWTH = '''import tensorflow as tf
for _g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(_g, True)
    except Exception: pass
'''

def localize(name):
    src = open(os.path.join(SRC, name), encoding='utf-8').read()
    n0 = src
    # 1. drop the pip install call (packages are installed in the local env)
    src = re.sub(r'import subprocess\nsubprocess\.run\(\["pip", "install", "imagehash", "-q"\], check=True\)\n', '', src)
    # 2. dataset path
    src = src.replace('"/kaggle/input/best-artworks-of-all-time/resized/resized"', 'LOCAL_IMG_DIR')
    src = src.replace('"/kaggle/input/datasets/ikarus777/best-artworks-of-all-time/resized/resized"', 'LOCAL_IMG_DIR')
    # 3. all outputs under F:\runs
    src = src.replace('"/kaggle/working/', 'f"{_RUNS}/')
    # 4. restore folders (only used if present)
    src = re.sub(r'"/kaggle/input/datasets/haithamq11/predictions-([a-z]+)/"', r'f"{_RUNS}/restore_\1/"', src)
    src = re.sub(r'"/kaggle/input/datasets/haithamqutaiba/predictions-([a-z0-9]+)"', r'f"{_RUNS}/restore_\1"', src)
    # 5. local Swin weights fallback
    src = src.replace('"/kaggle/input/datasets/haithamq11/modelsss/swin_tiny_patch4_window7_224.pth"',
                      'f"{_RUNS}/swin_tiny_patch4_window7_224.pth"')
    # 6. DataLoader workers
    src = src.replace('num_workers=2', 'num_workers=_NW')
    # 7. TF memory growth
    src = src.replace('import tensorflow as tf\n', TF_GROWTH, 1)
    assert '/kaggle/' not in src, f'{name}: a Kaggle path survived localisation'
    out = os.path.join(OUT, name)
    open(out, 'w', encoding='utf-8').write(HEADER + src)
    print(f'wrote {out}  ({len(n0)} -> {len(src)} chars)')

for f in ['04_baseline_swintiny.py', '05_baseline_convnext.py', '06_baseline_ensartnet.py', '07_leakage_isolation.py']:
    localize(f)
