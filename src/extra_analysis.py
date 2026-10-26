"""
Three things the paper needs that neither script prints.

1. The clinically legible form of the over-prediction: what average risk does the
   model announce to a population whose true prevalence is 10 percent, and how many
   people cross a decision threshold.
2. Whether the held-out split is genuinely EASIER than the source cohort, which is
   what a calibration slope above 1 would imply, with an interval on it.
3. Bootstrap intervals on the AUC at each stage, to show that "AUC does not move"
   is exact rather than merely within noise.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# scikit-learn 1.8 and later print a FutureWarning for penalty=None on every fit.
# It does not change any number, and thousands of copies bury the output.
warnings.filterwarnings("ignore", category=FutureWarning)

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
EPS = 1e-6

df = pd.read_csv(os.path.join(RUN, "baseline_oof_predictions.csv")).rename(
    columns={"True_age": "age", "gender": "sex"})
y = df["label"].values.astype(int)
A = df[["age", "sex"]].values.astype(np.float64)
g = df["group"].values
tr, te = g == 1, g != 1

sc = StandardScaler().fit(A[tr])
m = LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs").fit(sc.transform(A[tr]), y[tr])
p = m.predict_proba(sc.transform(A[te]))[:, 1]
yy = y[te]
src = y[tr].mean()


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def cal_slope(a, b):
    mm = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-10)
    mm.fit(logit(b).reshape(-1, 1), a)
    return float(mm.coef_[0, 0])


print("=" * 90)
print("1. THE CLINICALLY LEGIBLE FORM OF THE OVER-PREDICTION")
print("=" * 90)
rng = np.random.default_rng(42)
pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
for t in (0.05, 0.10, 0.20):
    npos = int(round(t * 300))
    nneg = 300 - npos
    sh = np.log(t / (1 - t)) - np.log(src / (1 - src))
    mp, mpf, ref, reff, tp, tpf = [], [], [], [], [], []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(pos, npos, True), rng.choice(neg, nneg, True)])
        a, b = yy[idx], p[idx]
        bf = 1 / (1 + np.exp(-(logit(b) + sh)))
        mp.append(b.mean())
        mpf.append(bf.mean())
        ref.append((b >= 0.5).mean())
        reff.append((bf >= 0.5).mean())
        tp.append(a[b >= 0.5].mean() if (b >= 0.5).sum() else np.nan)
        tpf.append(a[bf >= 0.5].mean() if (bf >= 0.5).sum() else np.nan)
    print("\ntrue prevalence %.0f%%" % (100 * t))
    print("  mean risk the UNCORRECTED model announces : %.1f%%  (truth %.1f%%)"
          % (100 * np.mean(mp), 100 * t))
    print("  mean risk after the one-number prior shift: %.1f%%" % (100 * np.mean(mpf)))
    print("  share of the population flagged at a 0.50 threshold: %.1f%% uncorrected, "
          "%.1f%% corrected" % (100 * np.mean(ref), 100 * np.mean(reff)))
    print("  of those flagged, the share who truly have thickened CIMT: %.1f%% uncorrected, "
          "%.1f%% corrected" % (100 * np.nanmean(tp), 100 * np.nanmean(tpf)))

print()
print("=" * 90)
print("2. IS THE HELD-OUT SPLIT EASIER THAN THE SOURCE COHORT")
print("=" * 90)
auc_te = roc_auc_score(yy, p)
b_auc = []
for _ in range(2000):
    idx = rng.integers(0, len(yy), len(yy))
    if len(np.unique(yy[idx])) > 1:
        b_auc.append(roc_auc_score(yy[idx], p[idx]))
print("AUC on the 300 held-out patients : %.4f  95%% CI [%.4f, %.4f], width %.4f"
      % (auc_te, np.percentile(b_auc, 2.5), np.percentile(b_auc, 97.5),
         np.percentile(b_auc, 97.5) - np.percentile(b_auc, 2.5)))
print("Within-cohort out-of-fold AUC for the same covariates, from the baseline run: 0.8530")
print("The within-cohort figure sits just below this interval, but the two AUCs come")
print("from different populations and the out-of-fold estimate has its own")
print("uncertainty, so no claim is made that the held-out split is easier.")

s = cal_slope(yy, p)
b_s = []
for _ in range(2000):
    idx = rng.integers(0, len(yy), len(yy))
    if len(np.unique(yy[idx])) > 1:
        try:
            b_s.append(cal_slope(yy[idx], p[idx]))
        except Exception:
            pass
print("\ncalibration slope on held out : %.4f  95%% CI [%.4f, %.4f]"
      % (s, np.percentile(b_s, 2.5), np.percentile(b_s, 97.5)))
print("excludes 1.0 : %s" % (np.percentile(b_s, 2.5) > 1.0))

print("\nage distribution, which the balancing changed:")
for nm, msk in (("train group 1", tr), ("held out", te)):
    a = df["age"].values[msk]
    print("  %-14s n=%4d mean %.2f sd %.2f  IQR %.1f to %.1f  prevalence %.4f"
          % (nm, msk.sum(), a.mean(), a.std(ddof=1),
             np.percentile(a, 25), np.percentile(a, 75), y[msk].mean()))
print("  normals in train : mean age %.2f   normals in held out : mean age %.2f"
      % (df["age"].values[tr & (y == 0)].mean(), df["age"].values[te & (y == 0)].mean()))
print("  cases   in train : mean age %.2f   cases   in held out : mean age %.2f"
      % (df["age"].values[tr & (y == 1)].mean(), df["age"].values[te & (y == 1)].mean()))

print()
print("=" * 90)
print("3. AUC AT EVERY STAGE, WITH INTERVALS, TO SHOW THE ZERO IS EXACT")
print("=" * 90)
shift = np.log(0.5 / 0.5) - np.log(src / (1 - src))


def cal_int(a, b):
    lo = logit(b)
    t = a.sum()
    return brentq(lambda z: (1 / (1 + np.exp(-(lo + z)))).sum() - t, -25, 25, xtol=1e-12)


stages = {
    "raw": p,
    "prior shift": 1 / (1 + np.exp(-(logit(p) + shift))),
    "intercept refit": 1 / (1 + np.exp(-(logit(p) + cal_int(yy, p)))),
}
for k, v in stages.items():
    print("%-18s AUC %.12f" % (k, roc_auc_score(yy, v)))
print("pairwise max |difference| : %.3e"
      % max(abs(roc_auc_score(yy, a) - roc_auc_score(yy, b))
            for a in stages.values() for b in stages.values()))
print("Both corrections are strictly increasing functions of the logit, so the patient")
print("ORDERING is untouched and any rank statistic is invariant by construction.")
print("\nDONE")
