"""
Build the 128-pixel image cache used ONLY by the fundus-luminance method check in
calibration_audit.py (the one subgroup axis derived from the photographs).

It decodes every released PNG once, resizes the full 512x512 frame to 128x128 with
bilinear interpolation (no crop) and stores uint8 RGB. This is the same preprocessing
as the cache that the paper's run used. If you skip this step, calibration_audit.py
still runs and simply omits the luminance rows; every other headline number is
unaffected.

Output: features/imgcache_128.npy and features/imgcache_128_files.csv
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "data", "Fundus_CIMT_2903 Dataset")
OUT_DIR = os.path.join(ROOT, "features")
SIDE = 128

files = np.array(sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")))
assert len(files) == 5806, "expected 5806 images, found %d" % len(files)
cache = np.zeros((len(files), SIDE, SIDE, 3), dtype=np.uint8)
for i, fn in enumerate(files):
    im = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB").resize((SIDE, SIDE), Image.BILINEAR)
    cache[i] = np.asarray(im, dtype=np.uint8)
    if i % 1000 == 0:
        print("  %d / %d" % (i, len(files)))
os.makedirs(OUT_DIR, exist_ok=True)
np.save(os.path.join(OUT_DIR, "imgcache_128.npy"), cache)
pd.DataFrame({"file": files}).to_csv(os.path.join(OUT_DIR, "imgcache_128_files.csv"), index=False)
print("wrote features/imgcache_128.npy", cache.shape)
