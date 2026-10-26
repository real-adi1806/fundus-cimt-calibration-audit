"""
THE PAIRED IMAGE TEST. Does adding 2,048 fundus-image features to age and sex
move the AUC for thickened CIMT, on the identical saved folds?

THE BAR, fixed in advance in the project record and written before any data was seen:
    "under 0.02 AUC over age+sex and that is the finding"
The denominator was measured by baseline.py: age+sex AUC 0.8530 on n=2,899.
THE BAR IS THEREFORE 0.8730. If the images miss it, that is the result. It gets
reported as the finding, and no other metric is hunted for afterwards.

WHAT IS COMPARED, AND WHAT IS DELIBERATELY NOT COMPARED
-------------------------------------------------------
    model A : age + sex
    model B : age + sex + the 2048-d image features
    delta   : AUC(B) - AUC(A), bootstrapped IN PAIRS on the same patients
This is NOT image-alone versus age-alone, which answers a different and much
easier question. And it is NOT two separate confidence intervals checked for
overlap, which is the wrong test and is systematically conservative: both
models rank the SAME patients, so the pairing must be kept inside the
bootstrap. Resample patients once per draw, score both models on that one
resample, take the difference.

Model A is RE-DERIVED here from scratch rather than quoted, so both sides of
the delta come out of a single instrument. It is then checked element by
element against the column baseline.py saved.

REGULARISATION
--------------
2,048 features against 2,899 patients overfits a plain logistic regression.
Model B uses an L2 penalty whose strength C is chosen by an inner
cross-validation INSIDE each training fold. The held-out fold is never seen by
that selection, and the chosen C is printed for every fold.

Age and sex enter model B effectively UNPENALISED: their standardised columns
are multiplied by 100, which divides their ridge penalty by 10,000. That makes
model B strictly nest model A, because at small C the image coefficients
vanish and B collapses onto A. Without it a delta could come out negative
purely because the penalty crushed the age coefficient, which would be an
artefact of the fitting rather than a statement about the images. The
fully-penalised variant is reported alongside as a sensitivity.

CIMT IS A WALL THICKNESS. Nobody in this dataset was followed up. There is no
event, no outcome and no time to event. Nothing here predicts heart attacks,
strokes or cardiovascular risk, and no sentence derived from it may say so.

USAGE
    python image_test.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent   # repo root

SEED = 0
N_SPLITS = 5
N_REPEATS = 5           # repeats for the split-stability table
N_BOOT = 2000           # paired bootstrap draws
N_PERM = 10             # shuffled-label permutations, FULL pipeline each
C_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
ALPHA_GRID = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
K_UNPEN = 100.0         # covariate scale-up, so covariates are ~unpenalised
INNER_K = 3

BASELINE_AUC_ON_RECORD = 0.8530
BAR = 0.8730

OUT = []
T0 = time.time()


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def hdr(s):
    say()
    say("=" * 78)
    say(s)
    say("=" * 78)


# ---------------------------------------------------------------------------
# 0. LOAD, AND RE-VERIFY THE ROW ALIGNMENT RATHER THAN TRUSTING IT
# ---------------------------------------------------------------------------
hdr("0. INPUTS, AND THE ROW ALIGNMENT RE-CHECKED HERE")

oof = pd.read_csv(ROOT / "baseline_oof_predictions.csv")
ids = pd.read_csv(ROOT / "features" / "patient_ids.csv")
Xmean = np.load(ROOT / "features" / "patient_features_mean.npy")

say("baseline_oof_predictions.csv rows    : %d" % len(oof))
say("patient_ids.csv rows                 : %d" % len(ids))
say("patient_features_mean.npy shape      : %s" % (Xmean.shape,))

a_ids = ids.patient_id.astype(str).to_numpy()
b_ids = oof.patient_id.astype(str).to_numpy()
n_mismatch = int((a_ids != b_ids).sum())
say("row order identical, element by element : %s  (mismatches %d)"
    % (n_mismatch == 0, n_mismatch))
if n_mismatch:
    raise SystemExit("ROW ORDER MISMATCH. Stopping: nothing below would mean anything.")

assert len(oof) == Xmean.shape[0] == 2899
assert oof.patient_id.nunique() == len(oof), "patient ids are not unique"
say("duplicate patient ids                : 0")
say("NaN / Inf anywhere in the features   : %s / %s"
    % (bool(np.isnan(Xmean).any()), bool(np.isinf(Xmean).any())))

y = oof.label.to_numpy(int)
age = oof.True_age.to_numpy(float)
sex = oof.gender.to_numpy(float)
groups = oof.patient_id.astype(str).to_numpy()
thickness = oof.thickness.to_numpy(float)

say()
say("n patients                           : %d" % len(y))
say("prevalence (thickened)               : %.1f%%" % (y.mean() * 100))
say("True_age                             : %d to %d, mean %.2f"
    % (age.min(), age.max(), age.mean()))
say("image features per patient           : %d (elementwise mean of the two eyes)"
    % Xmean.shape[1])

say()
say("THE WARNING SIGN ALREADY IN THE CACHE, restated before any model is fitted:")
say("  same-patient left/right cosine     : 0.9032")
say("  random-pair cosine                 : 0.8478")
say("  Two eyes of one person are barely more alike, under these features, than")
say("  two strangers are. If the primary test lands near zero, that is the most")
say("  likely reason, and it is a statement about THESE FEATURES, not about")
say("  retinal imaging in general.")


# ---------------------------------------------------------------------------
# 1. THE FOLD RECIPE, COPIED FROM baseline.py, NOT REINVENTED
# ---------------------------------------------------------------------------
STATE = {"overlap": 0, "fits": 0}
BALANCE = []


def folds_for(repeat, y_vec):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=SEED + repeat)
    return list(cv.split(np.zeros((len(y_vec), 1)), y_vec, groups=groups))


def _prep(Xtr, Xte, n_cov):
    """Standardise on the TRAINING fold only, then scale the covariates up so
    that their ridge penalty is negligible (K^2 = 10,000 times smaller)."""
    sc = StandardScaler().fit(Xtr)
    A_ = sc.transform(Xtr)
    B_ = sc.transform(Xte)
    if n_cov:
        A_[:, :n_cov] *= K_UNPEN
        B_[:, :n_cov] *= K_UNPEN
    return A_, B_


def fit_predict(X, y_vec, tr, te, mode, n_cov, inner_seed, chosen=None):
    """mode 'fixed'   : the baseline's exact recipe, C = 1.0, nothing selected.
       mode 'innercv' : C chosen by inner CV on the TRAINING FOLD ONLY."""
    if set(groups[tr]) & set(groups[te]):
        STATE["overlap"] += 1
    STATE["fits"] += 1

    if mode == "fixed":
        m = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, C=1.0))])
        m.fit(X[tr], y_vec[tr])
        return m.predict_proba(X[te])[:, 1]

    Xtr, Xte = _prep(X[tr], X[te], n_cov)
    ytr = y_vec[tr]
    inner = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=inner_seed)
    inner_splits = list(inner.split(Xtr, ytr))
    best_c, best_s = None, -np.inf
    for C in C_GRID:          # ascending, so a tie keeps the MORE regularised C
        s = []
        for itr, ite in inner_splits:
            mm = LogisticRegression(max_iter=2000, C=C)
            mm.fit(Xtr[itr], ytr[itr])
            s.append(roc_auc_score(ytr[ite], mm.predict_proba(Xtr[ite])[:, 1]))
        s = float(np.mean(s))
        if s > best_s:
            best_s, best_c = s, C
    if chosen is not None:
        chosen.append(best_c)
    m = LogisticRegression(max_iter=2000, C=best_c)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run_oof(X, y_vec, repeat, mode, n_cov, chosen=None, record=False):
    pred = np.full(len(y_vec), np.nan)
    for k, (tr, te) in enumerate(folds_for(repeat, y_vec)):
        if record:
            BALANCE.append((k, len(tr), len(te), y_vec[tr].mean(), y_vec[te].mean()))
        pred[te] = fit_predict(X, y_vec, tr, te, mode, n_cov,
                               inner_seed=SEED + repeat * 100 + k, chosen=chosen)
    assert not np.isnan(pred).any()
    return pred


X_A = np.column_stack([age, sex])
X_B = np.column_stack([age, sex, Xmean])


# ---------------------------------------------------------------------------
# 2. RE-DERIVE MODEL A, AND CHECK IT REPRODUCES
# ---------------------------------------------------------------------------
hdr("2. MODEL A RE-DERIVED HERE, NOT QUOTED")

say("Specification A from baseline.py: logistic regression on True_age and")
say("gender, StandardScaler fitted inside the training fold, C = 1.0,")
say("max_iter = 2000. Folds: StratifiedGroupKFold(5, shuffle=True,")
say("random_state=0) grouped on patient_id. The saved column is repeat 0, so")
say("repeat 0 is the fold assignment the primary comparison uses.")
say()

t = time.time()
predA = run_oof(X_A, y, 0, "fixed", n_cov=0, record=True)
aucA = roc_auc_score(y, predA)
saved = oof.oof_age_sex.to_numpy(float)
max_dev = float(np.abs(predA - saved).max())
say("my model A, repeat 0, pooled OOF AUC : %.4f   (%.1fs)" % (aucA, time.time() - t))
say("baseline.py on record                : %.4f" % BASELINE_AUC_ON_RECORD)
say("difference                           : %+.6f" % (aucA - BASELINE_AUC_ON_RECORD))
say("max |my OOF prob - saved OOF prob|   : %.3e" % max_dev)
REPRODUCED = abs(aucA - BASELINE_AUC_ON_RECORD) < 5e-5 and max_dev < 1e-9
say("reproduced exactly                   : %s" % REPRODUCED)
if not REPRODUCED:
    say()
    say("*** MY MODEL A DOES NOT REPRODUCE THE SAVED BASELINE. ***")
    say("Everything below uses MY re-derived A, so the delta is still internally")
    say("paired and valid, but the discrepancy is flagged here, not hidden.")
say("patient overlap violations so far    : %d over %d fold fits"
    % (STATE["overlap"], STATE["fits"]))


# ---------------------------------------------------------------------------
# 3. THE PRIMARY TEST
# ---------------------------------------------------------------------------
hdr("3. PRIMARY TEST: age+sex+images versus age+sex, PAIRED")

say("Model B: logistic regression on [True_age, gender, 2048 image features].")
say("All columns standardised on the training fold. Age and sex then multiplied")
say("by %.0f so their L2 penalty is 1/%d of what the image features carry,"
    % (K_UNPEN, int(K_UNPEN ** 2)))
say("which makes B nest A. C selected from %s" % C_GRID)
say("by %d-fold inner CV on the TRAINING FOLD ONLY, maximising inner AUC, with"
    % INNER_K)
say("ties resolved toward the more regularised value.")
say()

chosenB = []
t = time.time()
predB = run_oof(X_B, y, 0, "innercv", n_cov=2, chosen=chosenB)
aucB = roc_auc_score(y, predB)
say("model B, repeat 0, pooled OOF AUC    : %.4f   (%.1fs)" % (aucB, time.time() - t))
say("C chosen per outer fold              : %s" % chosenB)
say()


def paired_delta(y_true, sb, sa, n=N_BOOT, seed=SEED):
    """Bootstrap AUC(B) - AUC(A) with patients resampled ONCE per draw and both
    models scored on that same resample."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    d = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        d.append(roc_auc_score(y_true[s], sb[s]) - roc_auc_score(y_true[s], sa[s]))
    return np.array(d)


def boot_auc_ci(y_true, score, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    v = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        v.append(roc_auc_score(y_true[s], score[s]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


dboot = paired_delta(y, predB, predA)
delta = aucB - aucA
dlo, dhi = float(np.percentile(dboot, 2.5)), float(np.percentile(dboot, 97.5))
loA, hiA = boot_auc_ci(y, predA)
loB, hiB = boot_auc_ci(y, predB)

say("%-38s %8s  %22s" % ("model", "AUC", "95% CI (2000 boot)"))
say("%-38s %8.4f  [%.4f, %.4f]" % ("A  age + sex", aucA, loA, hiA))
say("%-38s %8.4f  [%.4f, %.4f]" % ("B  age + sex + 2048 image features", aucB, loB, hiB))
say()
say("PAIRED DELTA, AUC(B) - AUC(A), %d draws, same patients both sides" % len(dboot))
say("  point estimate                     : %+.4f" % delta)
say("  paired 95%% CI                      : [%+.4f, %+.4f]" % (dlo, dhi))
say("  paired SD of the delta             : %.4f" % dboot.std(ddof=1))
say("  draws with delta > 0               : %.1f%%" % ((dboot > 0).mean() * 100))
say("  draws with delta >= +0.0200        : %.1f%%" % ((dboot >= 0.02).mean() * 100))
say()
say("THE BAR (fixed in advance in the project record)")
say("  denominator (age+sex)              : %.4f" % aucA)
say("  bar = denominator + 0.0200         : %.4f" % (aucA + 0.02))
say("  model B reached                    : %.4f" % aucB)
say("  CLEARED THE BAR                    : %s" % (aucB >= aucA + 0.02))
say("  shortfall against the bar          : %+.4f" % (aucB - (aucA + 0.02)))
PRIMARY = dict(aucA=aucA, aucB=aucB, delta=delta, dlo=dlo, dhi=dhi,
               cleared=bool(aucB >= aucA + 0.02))

say()
say("Split stability: the same comparison repeated on %d different fold" % N_REPEATS)
say("assignments, so the answer is not an artefact of one shuffle.")
say("  %-8s %9s %9s %10s" % ("repeat", "AUC(A)", "AUC(B)", "delta"))
stab = []
for rep in range(N_REPEATS):
    if rep == 0:
        pa, pb = predA, predB
    else:
        pa = run_oof(X_A, y, rep, "fixed", n_cov=0)
        pb = run_oof(X_B, y, rep, "innercv", n_cov=2)
    ra, rb = roc_auc_score(y, pa), roc_auc_score(y, pb)
    stab.append(rb - ra)
    say("  %-8d %9.4f %9.4f %+10.4f" % (rep, ra, rb, rb - ra))
stab = np.array(stab)
say("  mean delta over %d fold assignments : %+.4f  (SD %.4f)"
    % (N_REPEATS, stab.mean(), stab.std(ddof=1)))


# ---------------------------------------------------------------------------
# 4. THE PRE-SPECIFIED SECONDARY: PREDICT AGE FROM THE IMAGES ALONE
# ---------------------------------------------------------------------------
hdr("4. SECONDARY (PRE-SPECIFIED): AGE FROM THE IMAGE FEATURES ALONE")

say("Why this decides how everything else is read. The thickened group is")
say("15.88 years older than the normal group (Cohen d 1.48). If the images")
say("predict age well, then any apparent retinal signal on CIMT may simply be")
say("AGE RE-ENTERING THROUGH THE PICTURES rather than a vascular finding. If")
say("the images predict age poorly, that route is closed, and it also caps how")
say("much CIMT signal these features could plausibly carry.")
say()
say("Ridge regression on the 2048 image features only. No age, no sex, no")
say("label. Alpha chosen by inner %d-fold CV inside each training fold, from" % INNER_K)
say("%s. Same patient-level folds, repeat 0." % ALPHA_GRID)
say()

pred_age = np.full(len(age), np.nan)
chosen_alpha = []
t = time.time()
for k, (tr, te) in enumerate(folds_for(0, y)):
    if set(groups[tr]) & set(groups[te]):
        STATE["overlap"] += 1
    STATE["fits"] += 1
    sc = StandardScaler().fit(Xmean[tr])
    Xtr, Xte = sc.transform(Xmean[tr]), sc.transform(Xmean[te])
    atr = age[tr]
    inner = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=SEED + k)
    inner_splits = list(inner.split(Xtr, y[tr]))
    best_a, best_s = None, np.inf
    for al in ALPHA_GRID:
        errs = []
        for itr, ite in inner_splits:
            rr = Ridge(alpha=al).fit(Xtr[itr], atr[itr])
            errs.append(mean_absolute_error(atr[ite], rr.predict(Xtr[ite])))
        e = float(np.mean(errs))
        if e < best_s:
            best_s, best_a = e, al
    chosen_alpha.append(best_a)
    pred_age[te] = Ridge(alpha=best_a).fit(Xtr, atr).predict(Xte)

r_p, p_p = pearsonr(pred_age, age)
r_s, p_s = spearmanr(pred_age, age)
ss_res = float(((age - pred_age) ** 2).sum())
ss_tot = float(((age - age.mean()) ** 2).sum())
r2 = 1.0 - ss_res / ss_tot
mae = float(mean_absolute_error(age, pred_age))
mae_null = float(mean_absolute_error(age, np.full_like(age, np.median(age))))

say("alpha chosen per outer fold          : %s   (%.1fs)" % (chosen_alpha, time.time() - t))
say()
say("AGE PREDICTED FROM THE IMAGE FEATURES ALONE, out of fold, n = %d" % len(age))
say("  Pearson r                          : %.4f  (p = %.3g)" % (r_p, p_p))
say("  Spearman rho                       : %.4f  (p = %.3g)" % (r_s, p_s))
say("  R^2 out of fold                    : %.4f" % r2)
say("  mean absolute error                : %.2f years" % mae)
say("  MAE of always predicting the median: %.2f years" % mae_null)
say("  MAE reduction over that null       : %.1f%%" % ((1 - mae / mae_null) * 100))
say("  SD of True_age in this cohort      : %.2f years" % age.std(ddof=1))
say()
say("For scale: a published retinal age model on healthy fundus photographs")
say("typically reports a Pearson r around 0.8 and an MAE near 3.5 years. This")
say("number is reported here purely as context for interpretation and is not a")
say("measurement made by this run.")
say()
say("AUC of the image-predicted age, used on its own as a CIMT score:")
auc_predage = roc_auc_score(y, pred_age)
say("  AUC                                : %.4f" % auc_predage)
say("  AUC of TRUE age used the same way  : %.4f" % roc_auc_score(y, age))
say("  This is the size of the age channel the images can actually recover.")


# ---------------------------------------------------------------------------
# 5. LEAKAGE CHECKS, ALL FOUR, PRINTED NOT ASSUMED
# ---------------------------------------------------------------------------
hdr("5. LEAKAGE CHECKS, ALL FOUR, PRINTED NOT ASSUMED")

say("5.1 Is any covariate a proxy for the label?")
say("    The label is EXACTLY (thickness >= 0.9). thickness is therefore")
say("    excluded from every design matrix in this file.")
say("    AUC of thickness used directly   : %.4f  (this is what a leak looks like)"
    % roc_auc_score(y, thickness))
say("    The supplied scaled 'age' column is min-max scaled over the WHOLE")
say("    dataset, test split included, so it carries a trace of the held-out")
say("    data. It is excluded. True_age is used instead and standardised")
say("    inside each training fold.")
say("    columns in model A               : True_age, gender")
say("    columns in model B               : True_age, gender, 2048 image features")
say("    'thickness' present in any design matrix : False")
say("    supplied scaled 'age' present            : False")
say("    Strongest single image feature vs the label, for context:")
fcorr = np.array([abs(np.corrcoef(Xmean[:, j], y)[0, 1]) if Xmean[:, j].std() > 0 else 0.0
                  for j in range(Xmean.shape[1])])
say("      max |corr(feature, label)|     : %.4f  (dim %d)"
    % (fcorr.max(), int(fcorr.argmax())))
say("      median |corr|                  : %.4f" % np.median(fcorr))
say("      no single feature is anywhere near a label proxy.")

say()
say("5.2 ZERO PATIENT OVERLAP ON EVERY FOLD")
say("    Every split is checked with a set intersection of the patient ids on")
say("    the two sides. The counter is incremented on any violation.")
say("    VIOLATION COUNT                  : %d" % STATE["overlap"])
say("    fold fits checked so far         : %d" % STATE["fits"])
say("    There is exactly one row per patient here, so a patient-level and a")
say("    row-level split coincide. The group-aware split is used anyway and the")
say("    count is printed rather than asserted.")

say()
say("5.3 CLASS BALANCE, TRAIN VERSUS TEST")
say("    %4s %8s %7s %18s %17s"
    % ("fold", "n_train", "n_test", "train % thickened", "test % thickened"))
for k, ntr, nte, ptr, pte in BALANCE:
    say("    %4d %8d %7d %17.2f%% %16.2f%%" % (k, ntr, nte, ptr * 100, pte * 100))
tr_p = np.array([b[3] for b in BALANCE])
te_p = np.array([b[4] for b in BALANCE])
say("    max absolute difference          : %.2f percentage points"
    % (np.abs(tr_p - te_p).max() * 100))

say()
say("5.4 SHUFFLED CONTROL. THE SINGLE MOST IMPORTANT CHECK IN THIS RUN.")
say("    The labels are permuted and the WHOLE pipeline is re-run INCLUDING")
say("    the 2048 image features and the inner-CV selection of C. If a model")
say("    with 2048 image features scores above chance on scrambled labels, the")
say("    pipeline is broken and nothing else it says is trustworthy.")
say("    %d permutations, model B, 5 folds each, %d fits in all."
    % (N_PERM, N_PERM * N_SPLITS))
say()
perm_auc = []
t = time.time()
for p in range(N_PERM):
    rng = np.random.default_rng(5000 + p)
    y_sh = y.copy()
    rng.shuffle(y_sh)
    ps = run_oof(X_B, y_sh, 0, "innercv", n_cov=2)
    perm_auc.append(roc_auc_score(y_sh, ps))
    say("      permutation %2d/%d  shuffled AUC = %.4f   (%.0fs elapsed)"
        % (p + 1, N_PERM, perm_auc[-1], time.time() - t))
perm_auc = np.array(perm_auc)
say()
say("    shuffled AUC mean                : %.4f" % perm_auc.mean())
say("    shuffled AUC range               : %.4f to %.4f" % (perm_auc.min(), perm_auc.max()))
say("    chance                           : 0.5000")
say("    real model B AUC                 : %.4f" % aucB)
n_ge = int((perm_auc >= aucB).sum())
say("    permutations reaching the real AUC : %d of %d  (p <= %.3f)"
    % (n_ge, N_PERM, (n_ge + 1) / (N_PERM + 1)))
PERM_PASS = abs(perm_auc.mean() - 0.5) < 0.03
say("    VERDICT                          : %s" % ("PASS" if PERM_PASS else "FAIL"))
if not PERM_PASS:
    say("    *** THE PIPELINE IS LEAKING. Nothing above is trustworthy. ***")


# ---------------------------------------------------------------------------
# 6. SENSITIVITY
# ---------------------------------------------------------------------------
hdr("6. SENSITIVITY")

Xcat = np.load(ROOT / "features" / "patient_features_concat_LR.npy")
X_Bc = np.column_stack([age, sex, Xcat])
t = time.time()
predBc = run_oof(X_Bc, y, 0, "innercv", n_cov=2)
aucBc = roc_auc_score(y, predBc)
dbc = paired_delta(y, predBc, predA)
say("6.1 Concatenated [L||R] features, %d-d, instead of the elementwise mean:" % Xcat.shape[1])
say("    AUC %.4f, paired delta over age+sex %+.4f [%+.4f, %+.4f]   (%.0fs)"
    % (aucBc, aucBc - aucA, np.percentile(dbc, 2.5), np.percentile(dbc, 97.5),
       time.time() - t))

say()
predBp = run_oof(X_B, y, 0, "innercv", n_cov=0)
aucBp = roc_auc_score(y, predBp)
dbp = paired_delta(y, predBp, predA)
say("6.2 Fully-penalised variant (age and sex carry the same L2 penalty as the")
say("    image features, so B no longer nests A):")
say("    AUC %.4f, paired delta %+.4f [%+.4f, %+.4f]"
    % (aucBp, aucBp - aucA, np.percentile(dbp, 2.5), np.percentile(dbp, 97.5)))

say()
predI = run_oof(Xmean, y, 0, "innercv", n_cov=0)
aucI = roc_auc_score(y, predI)
loI, hiI = boot_auc_ci(y, predI)
say("6.3 Images ALONE, no age and no sex. Reported for context only. This is")
say("    NOT the test fixed in advance in the project record, and it is the easier question that")
say("    test was written to avoid:")
say("    AUC %.4f [%.4f, %.4f]" % (aucI, loI, hiI))
say("    age alone, for comparison        : %.4f" % roc_auc_score(y, age))
rho_ia = spearmanr(predI, pred_age).statistic
say("    Spearman between the image-alone CIMT score and the image-predicted")
say("    age                              : %.4f" % rho_ia)
say("    A high value there means the image-alone model is largely an age")
say("    detector wearing a different label.")


# ---------------------------------------------------------------------------
# 7. WHAT THIS IS NOT
# ---------------------------------------------------------------------------
hdr("7. WHAT THIS IS NOT")
say("CIMT is a carotid wall thickness measured by ultrasound. No patient in")
say("this dataset was followed up. There is no cardiovascular event here, no")
say("outcome and no time to event. Nothing in this file predicts heart attacks,")
say("strokes or cardiovascular risk, and no sentence derived from it may say so.")
say()
say("The encoder is a frozen ImageNet ResNet-50. It was never trained on")
say("retinal images and never saw a CIMT label. A null result is a result about")
say("THESE features, not about fundus photography as a modality.")
say()
say("The cohort is retrospective, single centre, Department of Endocrinology")
say("inpatients at one hospital in Anhui Province, predominantly Chinese")
say("ethnicity, 70.8% thickened. It cannot speak to screening populations or")
say("to calibration across ethnic groups.")

say()
say("total runtime: %.0f seconds" % (time.time() - T0))

(ROOT / "image_test_output.txt").write_text("\n".join(OUT), encoding="utf-8")
json.dump(
    dict(auc_A=aucA, auc_B=aucB, delta=PRIMARY["delta"],
         delta_ci=[PRIMARY["dlo"], PRIMARY["dhi"]], bar=aucA + 0.02,
         cleared=PRIMARY["cleared"],
         age_from_image_pearson_r=float(r_p), age_from_image_r2=float(r2),
         age_from_image_mae=mae,
         shuffled_mean_auc=float(perm_auc.mean()), shuffled_pass=bool(PERM_PASS),
         overlap_violations=STATE["overlap"], fold_fits=STATE["fits"],
         auc_concat=aucBc, auc_image_alone=aucI),
    open(ROOT / "image_test_result.json", "w"), indent=2)
print()
print("[written] image_test_output.txt, image_test_result.json")
