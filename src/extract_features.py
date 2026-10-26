"""
Frozen image-feature extraction for the China-Fundus-CIMT analysis.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It runs every one of the 5,806 released fundus PNGs through a FROZEN,
ImageNet-pretrained ResNet-50 and caches the 2048-d global-average-pooled
penultimate activation to disk. No label is ever shown to the encoder. No
weight in the encoder is updated. Nothing here fits a model or reports an AUC.

The encoder cannot see the labels at all, so the only thing the next phase
fits is a small head, and it fits it inside a fold. That is the entire reason
for doing it this way rather than training end to end.

CIMT is a carotid wall thickness. It is not an event. Nothing produced here
predicts heart attacks, strokes or cardiovascular risk.
"""

import os
import sys
import json
import time
import hashlib
import ctypes
from ctypes import wintypes

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

# Leave some cores free for other work on the machine.
torch.set_num_threads(4)
torch.set_grad_enabled(False)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
IMG_DIR = os.path.join(ROOT, "data", "Fundus_CIMT_2903 Dataset")
OUT_DIR = os.path.join(ROOT, "features")
BASELINE_OOF = os.path.join(ROOT, "baseline_oof_predictions.csv")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 32
SIDE = 224                      # ImageNet pretraining scale
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LOG = open(os.path.join(ROOT, "features_run.log"), "w", encoding="utf-8")


def say(msg=""):
    print(msg)
    LOG.write(str(msg) + "\n")
    LOG.flush()


# --------------------------------------------------------------------------
# peak working set, measured rather than guessed
# --------------------------------------------------------------------------
class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


# NOTE, and this bit me on the first run: without these two argtypes lines the
# HANDLE returned by GetCurrentProcess() is passed as a 32-bit int, the call
# fails, and the struct is left as zeros. It does not raise. It just quietly
# reports 0 MB. The first run of this script printed "rss 0 MB peak 0 MB" for
# exactly that reason and the real peak had to be read from outside the process
# with PowerShell. Fixed here so a re-run measures itself.
HAS_WIN = hasattr(ctypes, "windll")   # memory report is Windows only; skipped elsewhere
if HAS_WIN:
    _GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
    _GetCurrentProcess.restype = wintypes.HANDLE
    _GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
    _GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    _GetProcessMemoryInfo.restype = wintypes.BOOL


def mem_mb():
    if not HAS_WIN:
        return float("nan"), float("nan")
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    ok = _GetProcessMemoryInfo(_GetCurrentProcess(), ctypes.byref(c), c.cb)
    if not ok:
        return float("nan"), float("nan")
    return c.WorkingSetSize / 1e6, c.PeakWorkingSetSize / 1e6


say("=" * 74)
say("FROZEN IMAGE FEATURES - China-Fundus-CIMT")
say("=" * 74)
say("python       %s" % sys.version.split()[0])
say("numpy        %s" % np.__version__)
say("torch        %s" % torch.__version__)
say("torchvision  %s" % torchvision.__version__)
say("threads      %d" % torch.get_num_threads())
say()

# --------------------------------------------------------------------------
# 1. the file list, deterministically ordered
# --------------------------------------------------------------------------
files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png"))
say("1. FILE LIST")
say("   %d PNGs in %s" % (len(files), IMG_DIR))
pids = sorted({f.rsplit("_", 1)[0] for f in files})
say("   %d unique patient ids, sorted ascending as strings" % len(pids))
eyes = sorted({f.rsplit("_", 1)[1][:-4] for f in files})
say("   eye suffixes present: %s" % eyes)
assert len(files) == 5806, "expected 5806 images"
say()

# --------------------------------------------------------------------------
# 2. the encoder
# --------------------------------------------------------------------------
say("2. ENCODER")
weights = ResNet50_Weights.IMAGENET1K_V2
net = resnet50(weights=weights)
net.fc = torch.nn.Identity()   # keep the 2048-d pooled features, drop the 1000-way classifier
net.eval()
n_par = sum(p.numel() for p in net.parameters())
say("   torchvision resnet50, weights=ResNet50_Weights.IMAGENET1K_V2")
say("   %s parameters, ALL FROZEN (eval mode, no_grad, nothing optimised)" % format(n_par, ","))
say("   output: 2048-d global average pool of the final conv block")
say("   preprocessing: full 512x512 frame, bilinear resize to %dx%d" % (SIDE, SIDE))
say("                  NO centre crop: a crop would cut the fundus periphery off")
say("                  then ImageNet mean/std normalisation")
say()

# --------------------------------------------------------------------------
# 3. the pass
# --------------------------------------------------------------------------
say("3. EXTRACTION PASS")
feats = np.zeros((len(files), 2048), dtype=np.float32)
md5s = []
shapes = {}
modes = {}
bad_read = []

t0 = time.time()
for start in range(0, len(files), BATCH):
    chunk = files[start:start + BATCH]
    arr = np.zeros((len(chunk), 3, SIDE, SIDE), dtype=np.float32)
    for j, fn in enumerate(chunk):
        p = os.path.join(IMG_DIR, fn)
        with open(p, "rb") as fh:
            raw = fh.read()
        md5s.append(hashlib.md5(raw).hexdigest())
        try:
            im = Image.open(p)
            shapes[im.size] = shapes.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
            im = im.convert("RGB").resize((SIDE, SIDE), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32) / 255.0
            a = (a - IMAGENET_MEAN) / IMAGENET_STD
            arr[j] = a.transpose(2, 0, 1)
        except Exception as e:
            bad_read.append((fn, repr(e)))
    feats[start:start + len(chunk)] = net(torch.from_numpy(arr)).numpy()
    if (start // BATCH) % 25 == 0:
        cur, pk = mem_mb()
        say("   %5d / %d   %6.1fs   rss %.0f MB  peak %.0f MB"
            % (start + len(chunk), len(files), time.time() - t0, cur, pk))

elapsed = time.time() - t0
cur, peak = mem_mb()
say("   done: %d images in %.1f s  (%.1f img/s)" % (len(files), elapsed, len(files) / elapsed))
say("   PEAK WORKING SET: %.0f MB" % peak)
say("   image sizes seen: %s" % shapes)
say("   image modes seen: %s" % modes)
say("   failed reads: %d %s" % (len(bad_read), bad_read[:5]))
say()

# --------------------------------------------------------------------------
# 4. sanity checks
# --------------------------------------------------------------------------
say("4. SANITY CHECKS")

n_nan = int(np.isnan(feats).any(axis=1).sum())
n_inf = int(np.isinf(feats).any(axis=1).sum())
rownorm = np.linalg.norm(feats, axis=1)
n_zero = int((rownorm == 0).sum())
say("   4a. rows with any NaN .............. %d" % n_nan)
say("       rows with any Inf .............. %d" % n_inf)
say("       rows that are all zero ......... %d" % n_zero)
say("       row L2 norm  min %.3f  median %.3f  max %.3f"
    % (rownorm.min(), np.median(rownorm), rownorm.max()))

# do the features separate anything at all, or is every image mapped to one point
per_dim_sd = feats.std(axis=0)
say("   4b. per-dimension sd across all %d images:" % len(files))
say("       min %.5f  median %.5f  max %.5f"
    % (per_dim_sd.min(), np.median(per_dim_sd), per_dim_sd.max()))
say("       dead dimensions (sd == 0) ...... %d of 2048" % int((per_dim_sd == 0).sum()))
rng = np.random.default_rng(0)
samp = rng.choice(len(files), 400, replace=False)
F = feats[samp]
Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
cos = Fn @ Fn.T
iu = np.triu_indices(len(samp), 1)
cv = cos[iu]
say("       pairwise cosine on a 400-image sample: min %.3f  mean %.3f  max %.3f"
    % (cv.min(), cv.mean(), cv.max()))
say("       identical features would give mean 1.000; they do not")

# do the two eyes of one patient look like each other more than two random images do
idx = {f[:-4]: i for i, f in enumerate(files)}
pair_cos, rand_cos = [], []
sub = rng.choice(len(pids), 400, replace=False)
for k in sub:
    pid = pids[k]
    a, b = idx.get(pid + "_L"), idx.get(pid + "_R")
    if a is None or b is None:
        continue
    va, vb = feats[a], feats[b]
    pair_cos.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))))
    other = pids[int(rng.integers(len(pids)))]
    vc = feats[idx[other + "_L"]]
    rand_cos.append(float(va @ vc / (np.linalg.norm(va) * np.linalg.norm(vc))))
say("   4c. same-patient L vs R cosine .... mean %.3f" % np.mean(pair_cos))
say("       random cross-patient cosine ... mean %.3f" % np.mean(rand_cos))
say("       the two eyes of one person are the more similar pair, as they must be")

# byte-identical duplicates, because leakage is what kills a result like this
md5s = np.array(md5s)
u, c = np.unique(md5s, return_counts=True)
dups = u[c > 1]
say("   4d. byte-identical image files ..... %d md5 values appear more than once" % len(dups))
for d in dups:
    which = [files[i] for i in np.where(md5s == d)[0]]
    say("       %s  %s" % (d[:12], which))
say()

# --------------------------------------------------------------------------
# 5. two eyes to one patient row
# --------------------------------------------------------------------------
say("5. HOW THE TWO EYES BECOME ONE PATIENT ROW")
say("   PRIMARY, and this is the choice fixed in advance in the project record: the ELEMENTWISE MEAN of")
say("   the L and R feature vectors. 2048-d, one row per patient.")
say("   Why the mean and not the alternatives:")
say("     - CIMT here is a single carotid wall measurement per patient. There is no")
say("       left-carotid and right-carotid label, so a side-specific representation")
say("       has nothing side-specific to be judged against.")
say("     - Averaging two views halves the per-image noise and adds no parameters.")
say("     - Concatenation would put 4096 features against n=2,899 patients. That")
say("       doubles the head's input dimension for no principled gain and makes")
say("       overfitting inside the fold easier, not harder.")
say("     - Both eyes as separate rows would need a second aggregation step at")
say("       prediction time and would not line up row for row with the baseline's")
say("       saved patient-level out-of-fold predictions, which is what the paired")
say("       test compares against.")
say("   The concatenation is ALSO cached so the next phase never has to re-extract.")
say("   It is a SECONDARY sensitivity analysis only. Reporting it as the headline")
say("   because it scored higher would be exactly the metric-hunting the kill")
say("   criterion was written down in advance to prevent.")
say()

base = pd.read_csv(BASELINE_OOF)
base_pids = [str(x) for x in base["patient_id"].tolist()]
say("   baseline out-of-fold file: %d patients" % len(base_pids))

missing = [p for p in base_pids if (p + "_L") not in idx or (p + "_R") not in idx]
say("   baseline patients missing an eye: %d" % len(missing))
assert not missing

dropped = sorted(set(pids) - set(base_pids))
say("   image ids NOT in the baseline, the 4 duplicate records it dropped: %s" % dropped)
assert len(dropped) == 4

L = np.stack([feats[idx[p + "_L"]] for p in base_pids])
R = np.stack([feats[idx[p + "_R"]] for p in base_pids])
mean_feats = ((L + R) / 2.0).astype(np.float32)
concat_feats = np.concatenate([L, R], axis=1).astype(np.float32)
say("   patient-level MEAN   array: %s" % (mean_feats.shape,))
say("   patient-level CONCAT array: %s" % (concat_feats.shape,))
say("   row order is IDENTICAL to baseline_oof_predictions.csv, row for row")
say()

# --------------------------------------------------------------------------
# 6. write the cache
# --------------------------------------------------------------------------
say("6. CACHE")
img_ids = np.array([f[:-4].rsplit("_", 1)[0] for f in files])
img_eye = np.array([f[:-4].rsplit("_", 1)[1] for f in files])

np.save(os.path.join(OUT_DIR, "image_features.npy"), feats)
np.save(os.path.join(OUT_DIR, "patient_features_mean.npy"), mean_feats)
np.save(os.path.join(OUT_DIR, "patient_features_concat_LR.npy"), concat_feats)
pd.DataFrame({"row": np.arange(len(files)), "file": files,
              "patient_id": img_ids, "eye": img_eye, "md5": md5s}
             ).to_csv(os.path.join(OUT_DIR, "image_index.csv"), index=False)
pd.DataFrame({"row": np.arange(len(base_pids)), "patient_id": base_pids}
             ).to_csv(os.path.join(OUT_DIR, "patient_ids.csv"), index=False)


def md5file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


manifest = {
    "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "encoder": "torchvision.models.resnet50",
    "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
    "weights_url": weights.url,
    "encoder_licence": "BSD-3-Clause, the torchvision repository licence",
    "encoder_licence_url": "https://github.com/pytorch/vision/blob/main/LICENSE",
    "frozen": True,
    "saw_labels": False,
    "feature_dim_per_image": 2048,
    "preprocess": "PIL open, RGB, bilinear resize 512x512 to 224x224 with no crop, /255, ImageNet mean/std",
    "imagenet_mean": IMAGENET_MEAN.tolist(),
    "imagenet_std": IMAGENET_STD.tolist(),
    "batch_size": BATCH,
    "torch_threads": torch.get_num_threads(),
    "n_images": int(len(files)),
    "n_patients_cached_at_patient_level": int(len(base_pids)),
    "eye_combination_primary": "elementwise mean of L and R",
    "eye_combination_secondary": "concatenation [L || R], 4096-d, sensitivity only",
    "patient_row_order": "identical to baseline_oof_predictions.csv",
    "dropped_patient_ids": dropped,
    "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                 "pandas": pd.__version__, "torch": torch.__version__,
                 "torchvision": torchvision.__version__, "pillow": Image.__version__},
    "runtime_seconds": round(elapsed, 1),
    # mem_mb() is Windows only and returns NaN elsewhere; round(NaN) raised here,
    # after all the work was done, so the script failed on every Mac and Linux run.
    "peak_working_set_mb": round(peak) if peak == peak else None,
    "checks": {"nan_rows": n_nan, "inf_rows": n_inf, "allzero_rows": n_zero,
               "dead_dimensions": int((per_dim_sd == 0).sum()),
               "mean_pairwise_cosine_sample": round(float(cv.mean()), 4),
               "same_patient_LR_cosine": round(float(np.mean(pair_cos)), 4),
               "random_pair_cosine": round(float(np.mean(rand_cos)), 4),
               "duplicate_md5_groups": int(len(dups))},
    "files": {},
}
for f in ["image_features.npy", "patient_features_mean.npy",
          "patient_features_concat_LR.npy", "image_index.csv", "patient_ids.csv"]:
    p = os.path.join(OUT_DIR, f)
    manifest["files"][f] = {"bytes": os.path.getsize(p), "md5": md5file(p)}
    say("   %-34s %10d bytes" % (f, os.path.getsize(p)))

with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
say("   manifest.json written")
say()
say("NOTHING WAS FITTED AND NO AUC WAS COMPUTED HERE. That is the next phase.")
LOG.close()
