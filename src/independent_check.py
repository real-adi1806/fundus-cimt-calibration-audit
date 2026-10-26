"""
INDEPENDENT RE-DERIVATION of the calibration audit's headline numbers.

Written fresh, reading only the raw files on disk. It does NOT import
calibration_audit.py. Its purpose is to check the INSTRUMENT, not to re-run the
pipeline: if two separately written implementations agree, the number belongs to
the data and not to one script's bug.

It also computes four quantities calibration_audit.py prints but does not
decompose, and which the paper needs:
  1. what FRACTION of the calibration gap the one-number prior shift closes, on
     four different scales, because "94.5 percent" depends on which one.
  2. what is LEFT after the ceiling correction, and what it is made of.
  3. whether AUC moves at any point, to twelve decimal places.
  4. the label-threshold fragility, re-derived straight from data_info.json.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

# scikit-learn 1.8 and later print a FutureWarning for penalty=None on every fit.
# It does not change any number, and thousands of copies bury the output.
warnings.filterwarnings("ignore", category=FutureWarning)

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
EPS = 1e-6

df = pd.read_csv(os.path.join(RUN, "baseline_oof_predictions.csv"))
df = df.rename(columns={"True_age": "age", "gender": "sex"})
Ximg = np.load(os.path.join(RUN, "features", "patient_features_mean.npy")).astype(np.float64)
ids = pd.read_csv(os.path.join(RUN, "features", "patient_ids.csv"))
assert (ids["patient_id"].values == df["patient_id"].values).all()
assert Ximg.shape[0] == len(df)

y = df["label"].values.astype(int)
A = df[["age", "sex"]].values.astype(np.float64)
g = df["group"].values
tr, te = g == 1, g != 1

print("n=%d  prevalence=%.6f  train n=%d prev=%.6f  heldout n=%d prev=%.6f"
      % (len(df), y.mean(), tr.sum(), y[tr].mean(), te.sum(), y[te].mean()))
print("image feature matrix: %s" % (Ximg.shape,))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cal_int(yy, pp):
    lo = logit(pp)
    t = yy.sum()

    def f(a):
        return (1 / (1 + np.exp(-(lo + a)))).sum() - t

    return brentq(f, -25, 25, xtol=1e-12)


def cal_slope(yy, pp):
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-10)
    m.fit(logit(pp).reshape(-1, 1), yy)
    return float(m.coef_[0, 0])


def ece(yy, pp, bins=10):
    n = len(yy)
    o = np.argsort(pp, kind="stable")
    yy, pp = yy[o], pp[o]
    e = np.linspace(0, n, bins + 1).astype(int)
    tot = 0.0
    for a, b in zip(e[:-1], e[1:]):
        if b > a:
            tot += (b - a) / n * abs(yy[a:b].mean() - pp[a:b].mean())
    return tot


def ece_tie_orders(yy, preds, n_orders=5000, seed=42, bins=10):
    """ECE over random orders of tied predictions (same permutation for every vector in
    the dict `preds`, then a stable sort). Returns {name: array of ECE values}."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_orders) for k in preds}
    for i in range(n_orders):
        k = rng.permutation(len(yy))
        for name, pv in preds.items():
            out[name][i] = ece(yy[k], pv[k], bins)
    return out


def ev(yy, pp, tag):
    r = dict(n=len(yy), prev=yy.mean(), auc=roc_auc_score(yy, pp), mean_pred=pp.mean(),
             OE=yy.mean() / pp.mean(), ci=cal_int(yy, pp), cs=cal_slope(yy, pp),
             ece=ece(yy, pp), brier=brier_score_loss(yy, pp))
    print("%-32s n=%4d prev=%.4f AUC=%.6f meanPred=%.6f O/E=%.6f calInt=%+.6f "
          "calSlp=%.6f ECE=%.6f Brier=%.6f"
          % (tag, r["n"], r["prev"], r["auc"], r["mean_pred"], r["OE"],
             r["ci"], r["cs"], r["ece"], r["brier"]))
    return r


# ---- model A, fitted exactly as the audit fits it: standardise on train, C=1 ----
sc = StandardScaler().fit(A[tr])
m = LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs").fit(sc.transform(A[tr]), y[tr])
p = m.predict_proba(sc.transform(A[te]))[:, 1]
yy = y[te]

print("\n--- EXPERIMENT A, model A, re-derived by independent code ---")
raw = ev(yy, p, "A age+sex -> held out pooled")
for gg in (2, 3):
    sub = g[te] == gg
    ev(yy[sub], p[sub], "   -> group %d" % gg)
    tg = ece_tie_orders(yy[sub], {"p": p[sub]})["p"]
    print("      tie averaged ECE, 5000 orders: %.6f [%.6f, %.6f]"
          % (tg.mean(), np.percentile(tg, 2.5), np.percentile(tg, 97.5)))

print("\n--- fitted coefficients, standardised scale ---")
print("intercept %+.6f   age %+.6f   sex %+.6f"
      % (m.intercept_[0], m.coef_[0][0], m.coef_[0][1]))

# ---- EXPERIMENT C, the two corrections ----
src, tgt = y[tr].mean(), yy.mean()
shift = np.log(tgt / (1 - tgt)) - np.log(src / (1 - src))
print("\n--- EXPERIMENT C, corrections ---")
print("source prevalence %.6f   target prevalence %.6f   logit shift %+.6f"
      % (src, tgt, shift))
pa = 1 / (1 + np.exp(-(logit(p) + shift)))
pb = 1 / (1 + np.exp(-(logit(p) + cal_int(yy, p))))
ra = ev(yy, pa, "(a) prior shift, deployable")
rb = ev(yy, pb, "(b) intercept refit, ceiling")

print("\n--- HOW MUCH OF THE GAP DOES THE ONE NUMBER CLOSE ---")
for name, k in (("ECE", "ece"), ("Brier", "brier"), ("|calInt|", "ci"), ("|O/E - 1|", "OE")):
    if k == "ci":
        r0, r1, r2 = abs(raw["ci"]), abs(ra["ci"]), abs(rb["ci"])
    elif k == "OE":
        r0, r1, r2 = abs(raw["OE"] - 1), abs(ra["OE"] - 1), abs(rb["OE"] - 1)
    else:
        r0, r1, r2 = raw[k], ra[k], rb[k]
    gap = r0 - r2
    closed = (r0 - r1) / gap if gap != 0 else float("nan")
    print("%-12s raw %.6f  prior-shift %.6f  ceiling %.6f  -> closes %.1f%% of the gap"
          % (name, r0, r1, r2, 100 * closed))

print("\n--- TIE AVERAGED ECE: 5000 random orders of tied predictions, seed 42 ---")
print("distinct prediction values in the %d held out predictions: %d" % (len(p), len(np.unique(p))))
orders = ece_tie_orders(yy, {"raw": p, "prior shift": pa, "intercept refit": pb})
for name, v in orders.items():
    print("%-16s mean %.6f  sd %.6f  [2.5%%, 97.5%%] [%.6f, %.6f]"
          % (name, v.mean(), v.std(ddof=1), np.percentile(v, 2.5), np.percentile(v, 97.5)))
gc = (orders["raw"] - orders["prior shift"]) / (orders["raw"] - orders["intercept refit"])
print("ECE gap closed by the prior shift, per order: mean %.2f%%  [2.5%%, 97.5%%] [%.2f%%, %.2f%%]"
      % (100 * gc.mean(), 100 * np.percentile(gc, 2.5), 100 * np.percentile(gc, 97.5)))

print("\n--- DOES AUC MOVE. printed to 12 decimals ---")
print("raw          %.12f" % raw["auc"])
print("prior shift  %.12f" % ra["auc"])
print("refit        %.12f" % rb["auc"])
print("max abs difference from raw: %.3e"
      % max(abs(ra["auc"] - raw["auc"]), abs(rb["auc"] - raw["auc"])))

print("\n--- WHAT IS LEFT AFTER THE CEILING, AND WHAT IT IS MADE OF ---")
print("ceiling ECE %.6f with calibration slope %.6f (1.000 is perfect)."
      % (rb["ece"], rb["cs"]))
s = cal_slope(yy, p)
pc = 1 / (1 + np.exp(-(s * logit(p))))
pc = 1 / (1 + np.exp(-(logit(pc) + cal_int(yy, pc))))
rc = ev(yy, pc, "(c) slope AND intercept refit")
tc = ece_tie_orders(yy, {"ceiling": pb, "slope+intercept": pc})
print("tie averaged ECE, 5000 orders: ceiling %.6f, slope AND intercept refit %.6f [%.6f, %.6f]"
      % (tc["ceiling"].mean(), tc["slope+intercept"].mean(),
         np.percentile(tc["slope+intercept"], 2.5), np.percentile(tc["slope+intercept"], 97.5)))
print("So the residual after an intercept-only fix is a SLOPE problem: refitting the")
print("slope as well takes ECE %.6f -> %.6f. An intercept correction cannot touch it."
      % (rb["ece"], rc["ece"]))

# ---- EXPERIMENT D, resampling to screening prevalence ----
print("\n--- EXPERIMENT D, resampled to target prevalence, 2000 draws, seed 42 ---")
rng = np.random.default_rng(42)
rng_tie = np.random.default_rng(43)                      # tie order, kept apart from the draws
pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
print("held out positives %d, negatives %d" % (len(pos), len(neg)))
for t in (0.05, 0.10, 0.20, 0.30, 0.50, float(src)):
    npos = max(2, int(round(t * 300)))
    nneg = max(2, 300 - npos)
    sh = np.log(t / (1 - t)) - np.log(src / (1 - src))
    oes, fx, ec = [], [], []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(pos, npos, True), rng.choice(neg, nneg, True)])
        a, b = yy[idx], p[idx]
        oes.append(a.mean() / b.mean())
        k = rng_tie.permutation(len(a))                   # random tie order per draw
        ec.append(ece(a[k], b[k]))
        bf = 1 / (1 + np.exp(-(logit(b) + sh)))
        fx.append(a.mean() / bf.mean())
    oes = np.array(oes)
    print("prev %.4f  O/E %.4f [%.4f, %.4f]  over-predicts %.2f x  ECE %.4f  "
          "after one-number fix O/E %.4f"
          % (t, oes.mean(), np.percentile(oes, 2.5), np.percentile(oes, 97.5),
             1 / oes.mean(), np.mean(ec), np.mean(fx)))

# ---- the label threshold fragility, re-derived from the raw metadata ----
print("\n--- LABEL THRESHOLD, re-derived from data_info.json itself ---")
info = json.load(open(os.path.join(RUN, "data", "data_info.json")))
th = np.array([v["thickness"] for v in info.values()], float)
lb = np.array([v["label"] for v in info.values()], int)
ag = np.array([v["True_age"] for v in info.values()], float)
sx = np.array([v["gender"] for v in info.values()], int)
print("records %d   keys per record %s" % (len(th), sorted(set(tuple(sorted(v.keys())) for v in info.values()))[0]))
print("label == (thickness >= 0.9) mismatches: %d" % int((lb != (th >= 0.9)).sum()))
print("distinct thickness values: %d   range %.2f to %.2f"
      % (len(np.unique(th)), th.min(), th.max()))
n09 = int((th == 0.9).sum())
print("patients with thickness EXACTLY 0.9: %d (%.2f%%), all label 1: %s"
      % (n09, 100 * n09 / len(th), bool((lb[th == 0.9] == 1).all())))
print("prevalence at the authors' 0.9 cut: %.4f" % lb.mean())
for cut in (0.8, 1.0, 1.1):
    print("   if the cut were %.1f mm instead: prevalence %.4f  (%+.1f percentage points)"
          % (cut, (th >= cut).mean(), 100 * ((th >= cut).mean() - lb.mean())))
print("mean age normal %.2f  thickened %.2f  gap %.2f years"
      % (ag[lb == 0].mean(), ag[lb == 1].mean(), ag[lb == 1].mean() - ag[lb == 0].mean()))
n0, n1 = (lb == 0).sum(), (lb == 1).sum()
sd = np.sqrt(((n0 - 1) * ag[lb == 0].std(ddof=1) ** 2 + (n1 - 1) * ag[lb == 1].std(ddof=1) ** 2)
             / (len(ag) - 2))
print("pooled SD %.2f  Cohen d %.2f" % (sd, (ag[lb == 1].mean() - ag[lb == 0].mean()) / sd))
print("sex coding: female(0) n=%d, male(1) n=%d" % (int((sx == 0).sum()), int((sx == 1).sum())))

print("\nDONE")
