"""
CALIBRATION AUDIT under distribution shift, on the China-Fundus-CIMT benchmark.

WHAT THIS ASKS
    Not "can a retinal model predict carotid thickening". That is answered and
    unimpressive: age and sex alone reach AUC 0.8530 and adding 2,048 frozen image
    features moves it +0.0004. See src/image_test.py, which produced that comparison.

    This asks the question underneath: when a model is moved from the population it
    was fitted on to a population with a different case mix, WHAT BREAKS? The answer
    on this benchmark is that discrimination survives and CALIBRATION does not. Of
    the abstracts read for the works citing this benchmark, none mentions
    calibration; no claim is made about their full texts or the wider literature.

READ THIS BEFORE INTERPRETING ANY SUBGROUP NUMBER BELOW
    THIS DATASET CARRIES NO ETHNICITY FIELD AND NO SKIN TONE FIELD.
    Measured, not assumed: all 2,903 records carry exactly 8 keys, which are
    gender, thickness, label, group, True_age, age, right_eye, left_eye. A search
    of the raw JSON for ethnic, race, skin, colour, ancestry and nation returns
    nothing. It is a single centre cohort at the Second Affiliated Hospital of
    Anhui Medical University, so it is close to single ancestry.

    No demographic column was invented to fill that gap. The subgroup axes used
    here are the ones that genuinely exist (sex, age band, and the data authors
    own split), plus one image derived stratum that is labelled for what it is:
    a measurement of the PHOTOGRAPH, not of the person. See section 2.

LINEAGE
    The per-group audit structure (predicted over observed per group, then recalibrate
    per group and show the ratio move toward 1) follows an earlier fairness pipeline by the
    same author, which expected an ethnicity column. That column does not exist in this
    dataset, so the method runs on the axes that are real here. The data and the shift
    experiment are new.

METHOD REFERENCE for the calibration hierarchy used below (verified live via
Europe PMC, hitCount 1, on the date this was written):
    Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW.
    "Calibration: the Achilles heel of predictive analytics." BMC Medicine 2019;17:230.
    DOI 10.1186/s12916-019-1466-7   PMID 31842878   PMCID PMC6912996
    https://doi.org/10.1186/s12916-019-1466-7

DATA
    China-Fundus-CIMT, Figshare DOI 10.6084/m9.figshare.27907056, licence CC0 1.0
    (confirmed live from https://api.figshare.com/v2/articles/27907056).
    2,903 patients, 5,806 fundus images. 2,899 patients after 4 duplicate
    records (byte-identical images under differing patient ids) were dropped
    in baseline.py, section 2.

RUNS ON: CPU only. No GPU. No network. Nothing is downloaded.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_BOOT = 2000
N_TIE_ORDERS = 5000                          # random tie orders for the tie averaged ECE
EPS = 1e-6
K_UNPEN = 100.0                              # covariate scale up, see _prep
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)     # ascending: a tie keeps the stronger penalty

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)                  # repo root
OUT = os.path.join(RUN, "out")


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------

def load_real():
    """Load the real cohort from disk. Returns None if it is not there."""
    oof = os.path.join(RUN, "baseline_oof_predictions.csv")
    feat = os.path.join(RUN, "features", "patient_features_mean.npy")
    pids = os.path.join(RUN, "features", "patient_ids.csv")
    if not (os.path.exists(oof) and os.path.exists(feat) and os.path.exists(pids)):
        return None

    df = pd.read_csv(oof)
    X_img = np.load(feat).astype(np.float32)
    order = pd.read_csv(pids)

    # The manifest states patient_row_order is identical to the oof csv. Verify it
    # rather than trust it. A silent row misalignment would invent a result.
    assert len(order) == len(df), "row count mismatch %d vs %d" % (len(order), len(df))
    assert (order["patient_id"].values == df["patient_id"].values).all(), \
        "patient_id order differs between patient_ids.csv and the oof csv"
    assert X_img.shape[0] == len(df), "feature rows %d vs %d" % (X_img.shape[0], len(df))

    df = df.rename(columns={"True_age": "age", "gender": "sex"})
    return df, X_img, "REAL"


def make_synthetic(n=900, d=64, seed=RANDOM_STATE):
    """
    ###  SYNTHETIC STAND IN. NOT REAL DATA. NOT A RESULT.  ###
    Only used when the real cohort is absent, so the code path can be exercised.
    Every number printed under this path is meaningless and is labelled SYNTHETIC.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(48, 14, n).clip(13, 86)
    sex = rng.binomial(1, 0.64, n)
    lin = -6.0 + 0.13 * age + 0.10 * sex
    y = rng.binomial(1, 1 / (1 + np.exp(-lin)))
    # image features that mostly encode age, which is what the real ones do
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    X[:, :8] += ((age - age.mean()) / age.std())[:, None] * 1.2
    grp = np.ones(n, dtype=int)
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    take = min(len(idx0), len(idx1), 100)
    sel = np.concatenate([rng.choice(idx0, take, replace=False),
                          rng.choice(idx1, take, replace=False)])
    grp[sel] = 3
    df = pd.DataFrame({"patient_id": np.arange(n), "age": age, "sex": sex,
                       "thickness": 0.9, "label": y, "group": grp})
    return df, X, "SYNTHETIC"


# ----------------------------------------------------------------------------
# 2. THE IMAGE DERIVED STRATUM, AND WHAT IT IS NOT
# ----------------------------------------------------------------------------

def fundus_luminance(df):
    """
    Median green channel inside the retinal disc, averaged over a patient two eyes.

    WHAT THIS IS: a measurement of the photograph. Fundus background luminance falls
    as choroidal and retinal pigment epithelium melanin rises, which is the basis of
    tessellated versus dark fundus grading in ophthalmology.

    WHAT THIS IS NOT, AND MUST NEVER BE CALLED: ethnicity, race, skin tone, or
    ancestry. Three reasons, all of which hold here.
      1. This is a near single ancestry cohort, so within it the variation is mostly
         individual ocular biology, not population structure.
      2. Camera model, flash power and exposure move this number directly, and the
         dataset publishes no acquisition metadata, so pigmentation and acquisition
         are CONFOUNDED and cannot be separated with what is on disk.
      3. It is a proxy that has never been validated against a pigmentation grade
         in this cohort.

    WHY IT IS HERE ANYWAY: it is the only axis available that tests whether the
    audit method detects calibration drift across an IMAGE property as opposed to a
    recorded covariate. If it works here it is the code you point at a multi ancestry
    cohort, where the stratum can be a real recorded one. Read it as a method check.
    """
    cache = os.path.join(RUN, "features", "imgcache_128.npy")
    files = os.path.join(RUN, "features", "imgcache_128_files.csv")
    if not (os.path.exists(cache) and os.path.exists(files)):
        return None

    arr = np.load(cache, mmap_mode="r")          # (n_img, 128, 128, 3) uint8
    names = pd.read_csv(files)["file"].tolist()
    assert len(names) == arr.shape[0]

    vals = np.empty(len(names), dtype=np.float64)
    for i in range(0, len(names), 256):          # chunked: this machine is RAM tight
        blk = np.asarray(arr[i:i + 256], dtype=np.uint8)
        g = blk[..., 1].reshape(blk.shape[0], -1)
        lum = blk.mean(axis=3).reshape(blk.shape[0], -1)
        for j in range(blk.shape[0]):
            m = lum[j] > 12                      # drop the black surround outside the disc
            vals[i + j] = np.median(g[j][m]) if m.sum() > 500 else np.nan

    pid = [n.split("_")[0] for n in names]
    per_img = pd.DataFrame({"patient_id": pid, "lum": vals})
    per_pat = per_img.groupby("patient_id", as_index=False)["lum"].mean()
    per_pat["patient_id"] = per_pat["patient_id"].astype(np.int64)
    merged = df[["patient_id"]].merge(per_pat, on="patient_id", how="left")
    assert len(merged) == len(df)
    # Fail loudly rather than let a NaN fall silently into the tertile cut points.
    assert np.isfinite(vals).all(), \
        "%d of %d images have no luminance measurement" % ((~np.isfinite(vals)).sum(), len(vals))
    assert np.isfinite(merged["lum"].values).all(), \
        "%d of %d patients have no luminance measurement" % (
            (~np.isfinite(merged["lum"].values)).sum(), len(merged))
    return merged["lum"].values


# ----------------------------------------------------------------------------
# 3. METRICS. Discrimination is one number. Calibration is four.
# ----------------------------------------------------------------------------

def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cal_intercept(y, p):
    """
    Calibration in the large on the logit scale. Fit y ~ sigmoid(logit(p) + a),
    slope held at 1, solve for a. a = 0 is perfect. a < 0 means the model predicts
    too high and must be pulled down.
    """
    lo = _logit(p)
    target = y.sum()

    def f(a):
        return (1 / (1 + np.exp(-(lo + a)))).sum() - target

    if f(-25) > 0 or f(25) < 0:
        return np.nan
    return brentq(f, -25, 25, xtol=1e-10)


def cal_slope(y, p):
    """
    Fit y ~ a + b*logit(p), report b. b = 1 is perfect. b < 1 means predictions are
    too spread out (over confident at both ends), which is the classic overfit shape.
    """
    lo = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-10)
    m.fit(lo, y)
    return float(m.coef_[0, 0])


def ece(y, p, bins=10):
    """Expected calibration error, equal count bins. Stable sort, so tied predictions
    keep their input order and the value is the same on every machine (this matches
    the calibration-shift package)."""
    n = len(y)
    if n < bins * 2:
        bins = max(2, n // 5)
    order = np.argsort(p, kind="stable")
    y, p = y[order], p[order]
    edges = np.linspace(0, n, bins + 1).astype(int)
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        tot += (b - a) / n * abs(y[a:b].mean() - p[a:b].mean())
    return tot


def ece_tie_orders(y, preds, bins=10, n_orders=N_TIE_ORDERS, seed=RANDOM_STATE):
    """
    ECE over random orderings of tied predictions. Equal count bins can split a run of
    identical predictions across a bin edge, so the single stable-sort value depends on
    row order. Each draw permutes the rows at random (the same permutation for every
    prediction vector in `preds`, a dict name -> p), then sorts stably, so ties are
    broken at random. Returns {name: array of n_orders ECE values}.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_orders) for k in preds}
    for i in range(n_orders):
        perm = rng.permutation(len(y))
        for k, pv in preds.items():
            out[k][i] = ece(y[perm], np.asarray(pv, dtype=float)[perm], bins)
    return out


def tie_note(name, y, p):
    """Print and return the tie averaged ECE for one prediction vector."""
    v = ece_tie_orders(y, {"p": p})["p"]
    lo, hi = np.percentile(v, [2.5, 97.5])
    print("%-26s   %d distinct predictions of %d; tie averaged ECE %.4f [%.4f, %.4f], %d orders"
          % ("", len(np.unique(p)), len(p), v.mean(), lo, hi, N_TIE_ORDERS))
    return {"ECE_tie_mean": float(v.mean()), "ECE_tie_lo": float(lo), "ECE_tie_hi": float(hi),
            "n_distinct": int(len(np.unique(p)))}


def evaluate(y, p):
    """The full row: discrimination, then the calibration hierarchy."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    obs, exp = y.mean(), p.mean()
    return {
        "n": len(y),
        "prev": obs,
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "mean_pred": exp,
        "OE": obs / exp if exp > 0 else np.nan,   # observed over expected
        "cal_intercept": cal_intercept(y, p),
        "cal_slope": cal_slope(y, p),
        "ECE": ece(y, p),
        "brier": brier_score_loss(y, p) if len(np.unique(y)) > 1 else np.nan,
    }


def boot_ci(y, p, fn, n_boot=N_BOOT, seed=RANDOM_STATE):
    """Percentile bootstrap CI for any scalar metric of (y, p)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            v = fn(y[idx], p[idx])
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ----------------------------------------------------------------------------
# 4. MODELS. Two, and the second must beat the first to be worth anything.
# ----------------------------------------------------------------------------

def _prep(Xtr, Xte, n_cov):
    """
    Standardise on the TRAINING rows only, then scale the first n_cov columns up
    by K_UNPEN so their ridge penalty is K_UNPEN^2 = 10,000 times smaller.

    WHY THIS EXISTS, and it is not a detail. Without it, one C is applied to age,
    sex and 2,048 image features alike, so the C that best regularises the images
    also CRUSHES THE AGE COEFFICIENT. Model B then scores worse than model A for a
    reason that is entirely about the fitting and nothing to do with the images.
    This was measured on the first version of this script: B came out at AUC 0.8105
    against A at 0.8535, and the age band calibration looked broken, both artefacts.

    With the scale up, B strictly NESTS A: at small C the image coefficients vanish
    and B collapses onto A. The fully penalised variant is still reported below as a
    sensitivity, because hiding it would be choosing the flattering specification.

    Method taken from src/image_test.py, which anticipated this.
    """
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    if n_cov:
        a[:, :n_cov] *= K_UNPEN
        b[:, :n_cov] *= K_UNPEN
    return a, b


def fit_predict(Xtr, ytr, Xte, C=1.0, n_cov=0):
    """Standardise on train only, L2 logistic, return probabilities on test."""
    a, b = _prep(Xtr, Xte, n_cov)
    m = LogisticRegression(C=C, max_iter=4000, solver="lbfgs")
    m.fit(a, ytr)
    return m.predict_proba(b)[:, 1]


def pick_C(Xtr, ytr, grid=C_GRID, seed=RANDOM_STATE, n_cov=0):
    """Inner CV on the training rows only. No test data is touched."""
    if Xtr.shape[1] <= 4:
        return 1.0
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best, bestC = -np.inf, grid[0]
    for C in grid:                      # ascending, so a tie keeps the MORE regularised C
        aucs = []
        for tr, va in skf.split(Xtr, ytr):
            p = fit_predict(Xtr[tr], ytr[tr], Xtr[va], C=C, n_cov=n_cov)
            aucs.append(roc_auc_score(ytr[va], p))
        m = float(np.mean(aucs))
        if m > best:
            best, bestC = m, C
    return bestC


def oof_predictions(X, y, n_splits=5, seed=RANDOM_STATE, n_cov=0):
    """Out of fold probabilities over the whole cohort, for the subgroup audit."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        C = pick_C(X[tr], y[tr], seed=seed, n_cov=n_cov)
        oof[te] = fit_predict(X[tr], y[tr], X[te], C=C, n_cov=n_cov)
    return oof


# ----------------------------------------------------------------------------
# 5. REPORTING
# ----------------------------------------------------------------------------

HDR = ("%-26s %6s %7s %7s %8s %7s %8s %7s %7s" %
       ("stratum", "n", "prev", "AUC", "meanPred", "O/E", "calInt", "calSlp", "ECE"))


def row(name, r):
    return ("%-26s %6d %7.3f %7.4f %8.3f %7.3f %8.3f %7.3f %7.4f" %
            (name[:26], r["n"], r["prev"], r["auc"], r["mean_pred"],
             r["OE"], r["cal_intercept"], r["cal_slope"], r["ECE"]))


def section(t):
    print()
    print("=" * 108)
    print(t)
    print("=" * 108)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    loaded = load_real()
    if loaded is None:
        print("!" * 108)
        print("!!  REAL COHORT NOT FOUND ON DISK. RUNNING ON A SYNTHETIC STAND IN.")
        print("!!  EVERY NUMBER BELOW IS MEANINGLESS AND MUST NOT BE QUOTED.")
        print("!!  Expected at: %s" % RUN)
        print("!" * 108)
        df, X_img, mode = make_synthetic()
        lum = None
    else:
        df, X_img, mode = loaded
        lum = fundus_luminance(df)

    y = df["label"].values.astype(int)
    A = df[["age", "sex"]].values.astype(np.float64)          # the honest baseline
    B = np.hstack([A, X_img]).astype(np.float64)              # baseline plus images

    section("DATA  [%s]" % mode)
    print("patients            : %d" % len(df))
    print("image feature dim   : %d" % X_img.shape[1])
    print("prevalence overall  : %.4f" % y.mean())
    print("age mean / sd       : %.2f / %.2f" % (df["age"].mean(), df["age"].std()))
    print("female fraction     : %.4f" % (1 - df["sex"].mean()))
    print("DEMOGRAPHIC COLUMNS PRESENT: age, sex. ETHNICITY: ABSENT. SKIN TONE: ABSENT.")
    if lum is not None:
        ok = np.isfinite(lum)
        print("fundus luminance    : measured on %d of %d patients, median %.1f, IQR %.1f to %.1f"
              % (ok.sum(), len(lum), np.nanmedian(lum),
                 np.nanpercentile(lum, 25), np.nanpercentile(lum, 75)))
        print("                      THIS IS A PROPERTY OF THE PHOTOGRAPH, NOT OF THE PERSON.")

    results = {"mode": mode, "n": int(len(df)), "prevalence": float(y.mean())}

    # ------------------------------------------------------------------
    # EXPERIMENT A: the shift that is already inside the published benchmark
    # ------------------------------------------------------------------
    section("EXPERIMENT A.  TRAIN ON THE AUTHORS TRAIN SPLIT, DEPLOY ON THEIR HELD OUT SPLITS")
    tr_mask = df["group"].values == 1
    print("The dataset ships a group column. Its three levels are not a random split:")
    for g in sorted(df["group"].unique()):
        s = df[df["group"] == g]
        print("   group %d : n=%4d  prevalence=%.4f  mean age=%.2f  female=%.3f"
              % (g, len(s), s["label"].mean(), s["age"].mean(), 1 - s["sex"].mean()))
    print()
    print("Train prevalence %.4f. Held out prevalence %.4f. That gap is the whole experiment."
          % (y[tr_mask].mean(), y[~tr_mask].mean()))
    print()
    print(HDR)
    print("-" * 108)

    expA = {}
    specs = (("A age+sex", A, 0), ("B age+sex+image", B, 2))
    for name, Xall, n_cov in specs:
        C = pick_C(Xall[tr_mask], y[tr_mask], n_cov=n_cov)
        for g in sorted(df["group"].unique()):
            if g == 1:
                continue
            te = df["group"].values == g
            p = fit_predict(Xall[tr_mask], y[tr_mask], Xall[te], C=C, n_cov=n_cov)
            r = evaluate(y[te], p)
            print(row("%s -> group %d" % (name, g), r))
            r.update(tie_note(name, y[te], p))
            expA["%s|group%d" % (name, g)] = r
        # both held out splits pooled, for a usable n
        te = ~tr_mask
        p = fit_predict(Xall[tr_mask], y[tr_mask], Xall[te], C=C, n_cov=n_cov)
        r = evaluate(y[te], p)
        lo, hi = boot_ci(y[te], p, lambda a, b: a.mean() / b.mean())
        ilo, ihi = boot_ci(y[te], p, cal_intercept)
        print(row("%s -> held out POOLED" % name, r))
        print("%-26s   O/E 95%% CI [%.3f, %.3f]   calInt 95%% CI [%.3f, %.3f]   (C=%.3g)"
              % ("", lo, hi, ilo, ihi, C))
        r.update(tie_note(name, y[te], p))
        expA["%s|pooled" % name] = dict(r)
        expA["%s|pooled" % name].update({"OE_lo": lo, "OE_hi": hi,
                                         "calInt_lo": ilo, "calInt_hi": ihi, "C": C})
        expA["%s|pooled" % name]["_p"] = p.tolist()
        expA["%s|pooled" % name]["_y"] = y[te].tolist()

    # Sensitivity: the fully penalised variant, where age and sex carry the same L2
    # penalty as the image features. Reported because choosing only the flattering
    # specification is how a result becomes untrue.
    Cfp = pick_C(B[tr_mask], y[tr_mask], n_cov=0)
    pfp = fit_predict(B[tr_mask], y[tr_mask], B[~tr_mask], C=Cfp, n_cov=0)
    rfp = evaluate(y[~tr_mask], pfp)
    print(row("B FULLY PENALISED (sens)", rfp))
    rfp.update(tie_note("B fully penalised", y[~tr_mask], pfp))
    print("%-26s   age and sex penalised like the images. C=%.3g. This is the variant"
          % ("", Cfp))
    print("%-26s   that would MISSTATE the images as harmful. It is the fit, not the data."
          % "")
    expA["B fully penalised|pooled"] = rfp
    results["experiment_A"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                               for k, v in expA.items()}

    # ------------------------------------------------------------------
    # EXPERIMENT B: subgroup calibration on the pooled cohort, out of fold
    # ------------------------------------------------------------------
    section("EXPERIMENT B.  OUT OF FOLD SUBGROUP AUDIT ON THE POOLED COHORT")
    print("5 fold stratified. Every patient is scored by a model that never saw them.")
    print("This is the July pipeline step 2 and 3, on real data, on the axes that exist.")

    oofA = oof_predictions(A, y, n_cov=0)
    oofB = oof_predictions(B, y, n_cov=2)

    strata = {"sex": np.where(df["sex"].values == 0, "female", "male")}
    strata["age band"] = pd.cut(df["age"], [0, 40, 50, 60, 200], right=False,
                                labels=["<40", "40-49", "50-59", "60+"]).astype(str).values
    if lum is not None and np.isfinite(lum).sum() > 100:
        q = np.nanpercentile(lum[np.isfinite(lum)], [33.333, 66.667])
        lab = np.full(len(lum), "unknown", dtype=object)
        ok = np.isfinite(lum)
        lab[ok & (lum <= q[0])] = "T1 darkest"
        lab[ok & (lum > q[0]) & (lum <= q[1])] = "T2 mid"
        lab[ok & (lum > q[1])] = "T3 lightest"
        strata["fundus luminance"] = lab

    expB = {}
    for mname, oof in (("A age+sex", oofA), ("B age+sex+image", oofB)):
        print()
        print("--- model %s   (whole cohort AUC %.4f, O/E %.3f) ---"
              % (mname, roc_auc_score(y, oof), y.mean() / oof.mean()))
        print(HDR)
        print("-" * 108)
        for axis, lab in strata.items():
            for lv in sorted(set(lab)):
                if lv == "unknown":
                    continue
                m = lab == lv
                if m.sum() < 30 or len(np.unique(y[m])) < 2:
                    continue
                r = evaluate(y[m], oof[m])
                print(row("%s = %s" % (axis, lv), r))
                expB["%s|%s=%s" % (mname, axis, lv)] = r
            print("-" * 108)
    results["experiment_B"] = expB

    # ------------------------------------------------------------------
    # EXPERIMENT C: the fix, and how much of it is deployable
    # ------------------------------------------------------------------
    section("EXPERIMENT C.  RECALIBRATION. DOES A SINGLE NUMBER FIX IT, AND AT WHAT COST")
    print("July pipeline step 3: recalibrate and show the ratio move toward 1.")
    print("Two corrections, and the difference between them is what you can actually ship.")
    print("  (a) PRIOR SHIFT: add logit(target prev) - logit(source prev). Needs only the")
    print("      target PREVALENCE, which a deploying hospital knows. Deployable.")
    print("  (b) INTERCEPT REFIT: fit the intercept on the target labels. Needs outcomes")
    print("      from the target site. This is the ceiling, not a shippable method.")
    print()
    print("%-34s %8s %8s %8s %8s" % ("", "AUC", "O/E", "calInt", "ECE"))
    print("-" * 70)

    expC = {}
    for name in ("A age+sex", "B age+sex+image"):
        key = "%s|pooled" % name
        p = np.array(expA[key]["_p"])
        yy = np.array(expA[key]["_y"])
        src = y[tr_mask].mean()
        tgt = yy.mean()

        r0 = evaluate(yy, p)
        shift = np.log(tgt / (1 - tgt)) - np.log(src / (1 - src))
        p_a = 1 / (1 + np.exp(-(_logit(p) + shift)))
        r_a = evaluate(yy, p_a)
        p_b = 1 / (1 + np.exp(-(_logit(p) + cal_intercept(yy, p))))
        r_b = evaluate(yy, p_b)

        for tag, r in (("raw", r0), ("(a) prior shift", r_a), ("(b) intercept refit", r_b)):
            print("%-34s %8.4f %8.3f %8.3f %8.4f"
                  % ("%s  %s" % (name, tag), r["auc"], r["OE"], r["cal_intercept"], r["ECE"]))
            expC["%s|%s" % (name, tag)] = r
        print("   applied logit shift for (a): %+.4f" % shift)
        print("   distinct prediction values: %d of %d" % (len(np.unique(p)), len(p)))
        orders = ece_tie_orders(yy, {"raw": p, "(a) prior shift": p_a, "(b) intercept refit": p_b})
        for tag, v in orders.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            print("   tie averaged ECE, %d random orders, %-20s mean %.4f  [2.5%%, 97.5%%] [%.4f, %.4f]"
                  % (N_TIE_ORDERS, tag, v.mean(), lo, hi))
            expC["%s|%s" % (name, tag)].update(
                {"ECE_tie_mean": float(v.mean()), "ECE_tie_lo": float(lo), "ECE_tie_hi": float(hi)})
        gc = (orders["raw"] - orders["(a) prior shift"]) / (orders["raw"] - orders["(b) intercept refit"])
        glo, ghi = np.percentile(gc, [2.5, 97.5])
        print("   ECE gap closed by prior shift, tie averaged: mean %.1f%%  [2.5%%, 97.5%%] [%.1f%%, %.1f%%]"
              % (100 * gc.mean(), 100 * glo, 100 * ghi))
        expC["%s|gap_closed_tie" % name] = {"mean": float(gc.mean()), "lo": float(glo), "hi": float(ghi)}
        print("-" * 70)
    results["experiment_C"] = expC

    # ------------------------------------------------------------------
    # EXPERIMENT D: the deployment the modality is actually proposed for
    # ------------------------------------------------------------------
    section("EXPERIMENT D.  WHAT HAPPENS AT SCREENING PREVALENCE")
    print("This cohort is INPATIENT ENDOCRINOLOGY. Inclusion was inpatients in the")
    print("Department of Endocrinology who had both fundus imaging and carotid ultrasound.")
    print("That is why 70.8 percent of them have thickened CIMT. A retinal camera used for")
    print("SCREENING meets a far healthier population, so the shift at deployment is much")
    print("larger than the 0.731 to 0.500 measured in experiment A.")
    print()
    print("Method: resample the 300 held out patients to a target prevalence, with")
    print("replacement, %d draws. This MEASURES the shift rather than assuming it." % N_BOOT)
    print("Held out positives and negatives available: %d and %d."
          % (int((np.array(expA["A age+sex|pooled"]["_y"]) == 1).sum()),
             int((np.array(expA["A age+sex|pooled"]["_y"]) == 0).sum())))
    print()
    print("%-38s %8s %8s %8s %8s" % ("", "O/E", "O/E lo", "O/E hi", "ECE"))
    print("-" * 74)

    expD = {}
    key = "A age+sex|pooled"
    p_all = np.array(expA[key]["_p"])
    y_all = np.array(expA[key]["_y"])
    pos = np.where(y_all == 1)[0]
    neg = np.where(y_all == 0)[0]
    rng = np.random.default_rng(RANDOM_STATE)
    rng_tie = np.random.default_rng(RANDOM_STATE + 1)    # tie order, kept apart from the draws
    src = y[tr_mask].mean()

    for tgt in (0.05, 0.10, 0.20, 0.30, 0.50, 0.7314):
        n_tot = 300
        n_pos = max(2, int(round(tgt * n_tot)))
        n_neg = max(2, n_tot - n_pos)
        oes, eces, oes_fix = [], [], []
        shift = np.log(tgt / (1 - tgt)) - np.log(src / (1 - src))
        for _ in range(N_BOOT):
            idx = np.concatenate([rng.choice(pos, n_pos, replace=True),
                                  rng.choice(neg, n_neg, replace=True)])
            yy, pp = y_all[idx], p_all[idx]
            oes.append(yy.mean() / pp.mean())
            k = rng_tie.permutation(len(yy))                  # random tie order per draw
            eces.append(ece(yy[k], pp[k]))
            pf = 1 / (1 + np.exp(-(_logit(pp) + shift)))
            oes_fix.append(yy.mean() / pf.mean())
        oe = float(np.mean(oes))
        lo, hi = float(np.percentile(oes, 2.5)), float(np.percentile(oes, 97.5))
        tag = "  (this cohort, train)" if abs(tgt - src) < 0.01 else ""
        print("%-38s %8.3f %8.3f %8.3f %8.4f"
              % ("target prevalence %.2f%s" % (tgt, tag), oe, lo, hi, float(np.mean(eces))))
        print("%-38s over predicts by %.2f x   after the one number prior shift fix: O/E %.3f"
              % ("", 1 / oe, float(np.mean(oes_fix))))
        expD["prev_%.4f" % tgt] = {"OE": oe, "OE_lo": lo, "OE_hi": hi,
                                   "ECE": float(np.mean(eces)),
                                   "OE_after_prior_shift": float(np.mean(oes_fix))}
    results["experiment_D"] = expD
    print("-" * 74)
    print("The uncorrected model is a risk INFLATOR, and the healthier the population the")
    print("worse it gets. Nothing in an AUC would tell you that.")
    print()
    print("THE ASSUMPTION IN THIS TABLE, AND IT IS A REAL LIMIT. Resampling changes the MIX")
    print("of cases and controls. It cannot change what a case or a control LOOKS like. A")
    print("real screening population has milder disease in its positives and healthier")
    print("negatives than an endocrinology ward does, and this cannot capture that. So read")
    print("these as the PREVALENCE COMPONENT of the deployment shift, isolated, not as a")
    print("forecast of a screening study. The direction of the residual is not known and is")
    print("not claimed. What is solid: the component that IS measured here is large, and one")
    print("number removes most of it.")

    # ------------------------------------------------------------------
    section("WHAT THIS DOES AND DOES NOT SHOW")
    print("DOES: whether discrimination and calibration come apart under the case mix")
    print("      shift that is already built into this published benchmark, and whether")
    print("      a one parameter correction closes the calibration gap.")
    print("DOES NOT: anything about ethnicity or skin tone. There is no such column here.")
    print("      The fundus luminance rows are a method check on an image property that is")
    print("      confounded with camera exposure. They are not a fairness result.")
    print("DOES NOT: predict cardiovascular events. The label is carotid wall thickness at")
    print("      a 0.9 mm cut, which is a marker, not an outcome. 13.2 percent of patients")
    print("      sit exactly on that cut.")
    print()
    print("THE ONE LINE THAT CONNECTS THIS TO THE ETHNICITY QUESTION, and it is the dataset")
    print("authors saying it, not us. They justify the 0.9 mm cut from a Chinese cohort and")
    print("explicitly acknowledge THE VARIABILITY OF CIMT THRESHOLDS ACROSS ETHNICITIES AND")
    print("AGE GROUPS. So on this task the ETHNICITY DEPENDENCE IS IN THE LABEL DEFINITION,")
    print("before any model exists. Transport this model to a population with a different")
    print("appropriate threshold and you have label shift on top of the prevalence shift")
    print("measured above. That is the thing worth studying, and it needs a cohort with a")
    print("recorded ethnicity column, which this one does not have.")

    out = os.path.join(OUT, "calibration_audit_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print()
    print("runtime %.1f s   wrote %s" % (time.time() - t0, out))


if __name__ == "__main__":
    sys.exit(main())
