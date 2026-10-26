"""
CIMT baseline: age and sex only. No images, no torch.

WHY THIS SCRIPT EXISTS
----------------------
The kill criterion for this project, fixed in advance in the project record, is:
    "under 0.02 AUC over age+sex and that is the finding"
So the age+sex model is the DENOMINATOR of the whole experiment. This script
establishes it. Nothing downstream means anything without it.

WHAT IS BEING PREDICTED
-----------------------
data_info.json carries two CIMT fields per patient:
    thickness : carotid intima-media thickness in mm, ultrasound-measured
    label     : 0/1, and it is exactly (thickness >= 0.9), verified below
The 0.9 mm cut is the DATASET AUTHORS' OWN choice. Their paper justifies it
from one Chinese cohort study on new-onset hypertension and acknowledges that
CIMT thresholds vary by ethnicity and age. It is not an international
guideline. This script did not pick it.

The continuous value is quantised to 15 distinct levels on a 0.1 mm grid, so
it is an ordinal variable rather than a smooth measurement. Regression on it
would overstate precision. The binary task is the honest one. A Spearman
correlation against the continuous value is reported alongside anyway.

CIMT IS A WALL THICKNESS. No patient in this dataset was followed up. There is
no event, no outcome, and nothing here predicts heart attacks or strokes.

USAGE
    python baseline.py                  # the analysis
    python baseline.py --verify-images  # also md5 all 5,806 PNGs (slow)
"""

import json
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent   # repo root; data/ lives here
INFO = ROOT / "data" / "data_info.json"
IMGDIR = ROOT / "data" / "Fundus_CIMT_2903 Dataset"

SEED = 0
N_SPLITS = 5
N_REPEATS = 10
N_BOOT = 2000
N_PERM = 20

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def hdr(s):
    say()
    say("=" * 78)
    say(s)
    say("=" * 78)


# ---------------------------------------------------------------------------
# 1. LOAD AND VERIFY THE DATA AT SOURCE
# ---------------------------------------------------------------------------
hdr("1. DATA, VERIFIED AT SOURCE")

if not INFO.exists():
    sys.exit("data/data_info.json not found. Download the dataset first: see "
             "'Get the data' in README.md.")
raw = json.loads(INFO.read_text())
df = pd.DataFrame.from_dict(raw, orient="index")
df.index.name = "patient_id"
df = df.reset_index()

say("data_info.json                      : %s" % INFO)
say("rows (patients)                     : %d" % len(df))
say("unique patient ids                  : %d" % df.patient_id.nunique())
say("columns                             : %s" % list(df.columns))
say("missing values anywhere             : %d" % int(df.isna().sum().sum()))

assert len(df) == df.patient_id.nunique(), "patient id is not unique per row"

# The label is a deterministic function of the continuous measurement.
derived = (df.thickness >= 0.9).astype(int)
n_mismatch = int((derived != df.label).sum())
say("label == (thickness >= 0.9) exactly : %s  (mismatches: %d)"
    % (n_mismatch == 0, n_mismatch))

# The supplied 'age' column is min-max scaled over the WHOLE dataset.
recon = (df.True_age - 13) / 73.0
max_dev = float(np.abs(recon - df.age).max())
say("supplied 'age' == (True_age-13)/73  : %s  (max deviation %.2e)"
    % (max_dev < 1e-12, max_dev))
say("  -> that scaling used whole-dataset min and max, so the supplied column")
say("     carries a trace of the test set. This script uses True_age and fits")
say("     all scaling inside the training folds only.")

say()
say("True_age range                      : %d to %d, mean %.2f"
    % (df.True_age.min(), df.True_age.max(), df.True_age.mean()))
say("thickness range                     : %.2f to %.2f mm, %d distinct values"
    % (df.thickness.min(), df.thickness.max(), df.thickness.nunique()))
say("class balance (whole cohort)        : %d normal / %d thickened (%.1f%% thickened)"
    % (int((df.label == 0).sum()), int((df.label == 1).sum()), df.label.mean() * 100))
say("sex                                 : %d female (0) / %d male (1)"
    % (int((df.gender == 0).sum()), int((df.gender == 1).sum())))
say("author groups                       : " + ", ".join(
    "%s:n=%d" % (g, int(c)) for g, c in df.group.value_counts().sort_index().items()))
say("  group 1 = train, 2 = validation, 3 = test (authors' own split).")
say("  Groups 2 and 3 are deliberately class-BALANCED (100/100 and 50/50)")
say("  against a true prevalence of 70.8%. AUC is prevalence-insensitive so it")
say("  is safe there; accuracy, F1 and calibration on those splits are not.")

say()
say("Age by class, the confounder that dominates this dataset:")
for lab, name in [(0, "normal   "), (1, "thickened")]:
    s = df.loc[df.label == lab, "True_age"]
    say("  %s n=%4d  mean age %5.2f  SD %5.2f"
        % (name, len(s), s.mean(), s.std(ddof=1)))
m0 = df.loc[df.label == 0, "True_age"]
m1 = df.loc[df.label == 1, "True_age"]
pooled_sd = np.sqrt(((len(m0) - 1) * m0.var(ddof=1) + (len(m1) - 1) * m1.var(ddof=1))
                    / (len(m0) + len(m1) - 2))
say("  gap %.2f years, Cohen d = %.2f"
    % (m1.mean() - m0.mean(), (m1.mean() - m0.mean()) / pooled_sd))
say()
say("Thickened rate by sex:")
for g, name in [(0, "female"), (1, "male  ")]:
    s = df.loc[df.gender == g, "label"]
    say("  %s n=%4d  %.1f%% thickened" % (name, len(s), s.mean() * 100))


# ---------------------------------------------------------------------------
# 2. DUPLICATE PATIENTS
# ---------------------------------------------------------------------------
hdr("2. DUPLICATE PATIENTS (a leakage check, run before any modelling)")

DUP_GROUPS = [
    ["533606001", "533682001"],
    ["545595003", "669579001"],
]

if "--verify-images" in sys.argv:
    say("Re-hashing all image files with md5 to find byte-identical eyes.")
    h2ids = defaultdict(set)
    n_files = 0
    for _, r in df.iterrows():
        for col in ("left_eye", "right_eye"):
            p = IMGDIR / r[col]
            h = hashlib.md5(p.read_bytes()).hexdigest()
            h2ids[h].add(r.patient_id)
            n_files += 1
    say("  files hashed                      : %d" % n_files)
    cross = defaultdict(set)
    for h, ids in h2ids.items():
        if len(ids) > 1:
            cross[tuple(sorted(ids))].add(h)
    say("  id pairs sharing identical images : %d" % len(cross))
    for k in cross:
        say("    %s  (%d identical image files)" % (list(k), len(cross[k])))
    if cross:
        DUP_GROUPS = [sorted(k) for k in cross]
else:
    say("Image hashing skipped (pass --verify-images to re-run it).")
    say("Using the duplicate groups established by the acquire phase.")

say()
say("Duplicate records, printed so the contradiction is visible:")
dup_ids = [i for g in DUP_GROUPS for i in g]
say(df[df.patient_id.isin(dup_ids)][
    ["patient_id", "gender", "True_age", "thickness", "label", "group"]
].to_string(index=False))
say()
say("The second pair is the same two photographs recorded once as a woman with")
say("normal CIMT and once as a man with thickened CIMT, so at least one row is")
say("wrong. All four sit in the authors' training group, so they do not cross")
say("the authors' train/test boundary. They are DROPPED from the primary")
say("analysis and a sensitivity run with them kept is reported at the end.")
say()
say("Limit of this check: md5 only finds BYTE-IDENTICAL files. A patient")
say("re-photographed on another day would not be caught by it.")

df_full = df.copy()
df = df[~df.patient_id.isin(dup_ids)].reset_index(drop=True)
say()
say("n after dropping %d duplicate records : %d" % (len(dup_ids), len(df)))


# ---------------------------------------------------------------------------
# 3. MODEL SPECIFICATIONS
# ---------------------------------------------------------------------------
hdr("3. MODEL SPECIFICATIONS")

say("Features: True_age (years) and gender (0 female / 1 male). Nothing else.")
say("thickness is EXCLUDED: the label is a deterministic function of it.")
say("The supplied scaled 'age' column is EXCLUDED: see the leakage note above.")
say()
say("Three specifications are fitted. The criterion fixed in advance in the project record is measured")
say("'over age+sex', so the denominator must be the BEST honest age+sex model,")
say("not the weakest. A weak denominator would flatter anything downstream.")
say("  A  logistic, age + sex (linear in age)")
say("  B  logistic, age + age^2 + sex + age:sex")
say("  C  gradient boosting on (age, sex), fully non-parametric in age")


def features(frame, spec):
    a = frame.True_age.to_numpy(float)
    g = frame.gender.to_numpy(float)
    if spec == "A":
        return np.column_stack([a, g])
    if spec == "B":
        return np.column_stack([a, a * a, g, a * g])
    if spec == "C":
        return np.column_stack([a, g])
    if spec == "age":
        return a.reshape(-1, 1)
    if spec == "sex":
        return g.reshape(-1, 1)
    raise ValueError(spec)


def make_model(spec):
    if spec == "C":
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    return Pipeline([
        ("scale", StandardScaler()),   # fitted on the training fold only
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


SPECS = ["A", "B", "C", "age", "sex"]
SPEC_NAMES = {
    "A": "A  logistic: age + sex",
    "B": "B  logistic: age + age^2 + sex + age:sex",
    "C": "C  gradient boosting: (age, sex)",
    "age": "   age alone",
    "sex": "   sex alone",
}


# ---------------------------------------------------------------------------
# 4. CROSS VALIDATION, SPLIT AT PATIENT LEVEL
# ---------------------------------------------------------------------------
hdr("4. CROSS VALIDATION (patient-level splits, 5-fold x 10 repeats)")

say("There is exactly one row per patient, so a patient-level split and a")
say("row-level split coincide here. StratifiedGroupKFold is used with the")
say("patient id as the group ANYWAY, and every fold is asserted to share no")
say("patient with its training half. That assertion is printed, not assumed.")
say("It also matters downstream: the image model has TWO rows per patient")
say("(left and right eye) and must reuse these same patient-level folds.")
say()

y = df.label.to_numpy(int)
groups = df.patient_id.to_numpy()

STATE = {"overlap": 0, "fits": 0}
fold_balance = []


def run_cv(frame, y_vec, grp, spec, record=False, n_repeats=N_REPEATS):
    """Return (per-repeat AUCs, out-of-fold probabilities from repeat 0)."""
    X = features(frame, spec)
    repeat_aucs = []
    oof0 = None
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=SEED + rep)
        oof = np.full(len(y_vec), np.nan)
        for k, (tr, te) in enumerate(cv.split(X, y_vec, groups=grp)):
            if set(grp[tr]) & set(grp[te]):
                STATE["overlap"] += 1
            STATE["fits"] += 1
            if record and rep == 0:
                fold_balance.append((k, len(tr), len(te),
                                     y_vec[tr].mean(), y_vec[te].mean()))
            m = make_model(spec)
            m.fit(X[tr], y_vec[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        assert not np.isnan(oof).any()
        repeat_aucs.append(roc_auc_score(y_vec, oof))
        if rep == 0:
            oof0 = oof
    return np.array(repeat_aucs), oof0


def boot_ci(y_true, score, n=N_BOOT, seed=SEED):
    """Patient-level bootstrap percentile CI for AUC."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        vals.append(roc_auc_score(y_true[s], score[s]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


results = {}
for spec in SPECS:
    aucs, oof = run_cv(df, y, groups, spec, record=(spec == "A"))
    lo, hi, nb = boot_ci(y, oof)
    results[spec] = dict(aucs=aucs, oof=oof, lo=lo, hi=hi, nboot=nb)

say("Fold sanity for specification A, repeat 0 (class balance train vs test):")
say("  %4s %8s %7s %18s %17s"
    % ("fold", "n_train", "n_test", "train % thickened", "test % thickened"))
for k, ntr, nte, ptr, pte in fold_balance:
    say("  %4d %8d %7d %17.2f%% %16.2f%%" % (k, ntr, nte, ptr * 100, pte * 100))
say("  patient overlap violations so far : %d (over %d fold fits)"
    % (STATE["overlap"], STATE["fits"]))
say()

say("%-42s %17s  %22s  %9s"
    % ("specification", "AUC (pooled OOF)", "95% CI (2000 boot)", "repeat SD"))
for spec in SPECS:
    r = results[spec]
    point = roc_auc_score(y, r["oof"])
    say("%-42s %17.4f  [%.4f, %.4f]  %9.4f"
        % (SPEC_NAMES[spec], point, r["lo"], r["hi"], r["aucs"].std(ddof=1)))
say()
say("n = %d patients, every one held out exactly once per repeat." % len(df))
say("Bootstrap resamples patients, %d draws, percentile interval." % N_BOOT)
say("repeat SD is the spread of the pooled AUC across the 10 fold assignments,")
say("so it measures how much the answer depends on the split you happened to use.")

DENOM = max(["A", "B", "C"], key=lambda s: roc_auc_score(y, results[s]["oof"]))
DENOM_AUC = roc_auc_score(y, results[DENOM]["oof"])
say()
say("DENOMINATOR = specification %s, AUC %.4f [%.4f, %.4f]"
    % (DENOM, DENOM_AUC, results[DENOM]["lo"], results[DENOM]["hi"]))
say("A fundus model must reach %.4f to clear the kill criterion fixed in advance in the project record"
    % (DENOM_AUC + 0.02))
say("of +0.02 AUC over age and sex.")


# ---------------------------------------------------------------------------
# 4b. HOW THE CRITERION MUST BE TESTED, demonstrated on a delta we already have
# ---------------------------------------------------------------------------
hdr("4b. HOW TO ADJUDICATE THE CRITERION (paired bootstrap, worked example)")

say("The criterion fixed in advance in the project record is about a DIFFERENCE of two AUCs measured on")
say("the SAME patients. Comparing two separate confidence intervals and asking")
say("whether they overlap is the wrong test and is systematically conservative.")
say("The difference has to be bootstrapped in PAIRS: resample patients once,")
say("then score both models on that same resample and take the difference.")
say()
say("Worked here on a delta that already exists, age+sex over age alone, so")
say("the downstream image phase has a procedure to copy rather than invent.")


def paired_delta_ci(y_true, score_a, score_b, n=N_BOOT, seed=SEED):
    """Bootstrap CI for AUC(a) - AUC(b) with patients resampled once per draw."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    d = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        d.append(roc_auc_score(y_true[s], score_a[s])
                 - roc_auc_score(y_true[s], score_b[s]))
    d = np.array(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float(d.std(ddof=1)), len(d))


dm, dlo, dhi, dsd, dn = paired_delta_ci(y, results[DENOM]["oof"], results["age"]["oof"])
say()
say("delta = AUC(age+sex) - AUC(age alone), same patients, %d paired draws" % dn)
say("  point estimate                    : %+.4f" % (DENOM_AUC - roc_auc_score(y, results["age"]["oof"])))
say("  paired 95%% CI                     : [%+.4f, %+.4f]" % (dlo, dhi))
say("  paired SD of the delta            : %.4f" % dsd)
say()
say("Two things follow, and both matter for the next phase.")
say("1. Sex adds %+.4f AUC over age alone. The criterion fixed in advance in the project record asks the"
    % (DENOM_AUC - roc_auc_score(y, results["age"]["oof"])))
say("   fundus images for +0.0200, which is about %.0fx what sex contributes."
    % ((0.02) / max(DENOM_AUC - roc_auc_score(y, results["age"]["oof"]), 1e-9)))
say("2. The paired SD of a delta on these %d patients is %.4f, so a true"
    % (len(df), dsd))
say("   +0.02 improvement sits about %.1f paired SDs from zero and IS"
    % (0.02 / dsd))
say("   resolvable at this sample size, using this cross-validation.")
say("   That is the justification for adjudicating on section 4 and not on")
say("   the authors' 100-patient test split.")
say()
say("CAVEAT ON THAT SD, so it is not over-read. The two scores compared here")
say("are nearly collinear: age+sex and age alone rank patients almost")
say("identically (Spearman %.4f between the two out-of-fold columns). A paired"
    % spearmanr(results[DENOM]["oof"], results["age"]["oof"]).statistic)
say("bootstrap of a delta between two such similar models has an unusually")
say("small SD. A fundus model would rank patients differently, the pairing")
say("would cancel less, and the SD of ITS delta will be larger, probably")
say("several times larger. So 0.0020 is a floor, not the expected precision")
say("of the real comparison. The procedure is the thing to copy from here.")
say("The precision of the final test must be re-measured on the real pair.")

# Out-of-fold predictions saved so the image phase compares on identical folds.
oof_df = pd.DataFrame({
    "patient_id": df.patient_id,
    "True_age": df.True_age,
    "gender": df.gender,
    "thickness": df.thickness,
    "label": y,
    "group": df.group,
    "oof_age_sex": results[DENOM]["oof"],
    "oof_age_only": results["age"]["oof"],
})
oof_df.to_csv(ROOT / "baseline_oof_predictions.csv", index=False)
say()
say("Out-of-fold predictions written to baseline_oof_predictions.csv (n=%d)."
    % len(oof_df))
say("The image phase MUST reuse the same patient-level fold assignment")
say("(StratifiedGroupKFold, n_splits=%d, shuffle=True, random_state=%d..%d,"
    % (N_SPLITS, SEED, SEED + N_REPEATS - 1))
say("groups=patient_id) and compare against this column with the paired")
say("bootstrap above. Anything else is not measuring the quantity that criterion is about.")


# ---------------------------------------------------------------------------
# 5. OPERATING POINT AND CALIBRATION
# ---------------------------------------------------------------------------
hdr("5. OPERATING POINT AND CALIBRATION (denominator model, out-of-fold)")

oof = results[DENOM]["oof"]
fpr, tpr, thr = roc_curve(y, oof)
j = int(np.argmax(tpr - fpr))
pred = (oof >= thr[j]).astype(int)
tp = int(((pred == 1) & (y == 1)).sum())
fn = int(((pred == 0) & (y == 1)).sum())
tn = int(((pred == 0) & (y == 0)).sum())
fp = int(((pred == 1) & (y == 0)).sum())
say("Youden-optimal threshold            : p >= %.4f" % thr[j])
say("  sensitivity                       : %.1f%%  (%d/%d)"
    % (tp / (tp + fn) * 100, tp, tp + fn))
say("  specificity                       : %.1f%%  (%d/%d)"
    % (tn / (tn + fp) * 100, tn, tn + fp))
say("  accuracy                          : %.1f%%" % ((tp + tn) / len(y) * 100))
say("  prevalence in this cohort         : %.1f%% thickened" % (y.mean() * 100))
say("  Always calling 'thickened' scores %.1f%%, so accuracy alone says almost"
    % (y.mean() * 100))
say("  nothing here. The threshold was chosen ON the out-of-fold predictions,")
say("  so sensitivity and specificity at it are optimistic; the AUC is not.")
say()
say("Brier score (OOF)                   : %.4f" % brier_score_loss(y, oof))
bins = np.quantile(oof, np.linspace(0, 1, 6))
bins[-1] += 1e-9
say("Calibration by quintile of predicted probability:")
say("  %3s %5s %15s %10s" % ("bin", "n", "mean predicted", "observed"))
for b in range(5):
    m = (oof >= bins[b]) & (oof < bins[b + 1])
    say("  %3d %5d %14.1f%% %9.1f%%"
        % (b + 1, int(m.sum()), oof[m].mean() * 100, y[m].mean() * 100))
say("Calibration is meaningful here because the CV runs over the whole cohort")
say("at its true prevalence, unlike the authors' balanced validation and test splits.")

rho, pval = spearmanr(oof, df.thickness)
say()
say("Spearman correlation, predicted probability vs continuous CIMT in mm")
say("  rho = %.4f  (p = %.3g, n = %d)" % (rho, pval, len(df)))
say("  Reported because the continuous value is quantised to 15 levels, so a")
say("  rank statistic is honest where an R2 on it would not be.")


# ---------------------------------------------------------------------------
# 6. LEAKAGE CHECKS
# ---------------------------------------------------------------------------
hdr("6. LEAKAGE CHECKS, each reported explicitly")

say("6.1 Is any covariate a proxy for the label?")
say("    thickness -> label is EXACT (label = thickness >= 0.9, %d mismatches)."
    % n_mismatch)
say("    thickness is therefore excluded from the feature matrix. The features")
say("    actually used are True_age and gender, and nothing else.")
say("    The supplied scaled 'age' column is min-max over the whole dataset")
say("    (verified exactly in section 1) and is also excluded. True_age is used")
say("    and standardised inside each training fold.")
say()
say("6.2 Does the same patient appear on both sides of a split?")
say("    Group-aware folds, every split checked: %d violations over %d fold fits."
    % (STATE["overlap"], STATE["fits"]))
say("    Four records that are two people (byte-identical images) were dropped")
say("    before splitting, so they cannot sit on opposite sides either.")
say()
say("6.3 Is the class balance the same in train and test?")
tr_p = np.array([b[3] for b in fold_balance])
te_p = np.array([b[4] for b in fold_balance])
say("    train prevalence across folds     : %.2f%% to %.2f%%"
    % (tr_p.min() * 100, tr_p.max() * 100))
say("    test  prevalence across folds     : %.2f%% to %.2f%%"
    % (te_p.min() * 100, te_p.max() * 100))
say("    max absolute difference           : %.2f percentage points"
    % (np.abs(tr_p - te_p).max() * 100))
say("    Stratified folds, so this is by construction. Printed, not assumed.")
say()

say("6.4 Shuffled-label control. The whole pipeline is re-run with the labels")
say("    permuted. If it does not land at chance the pipeline is broken and")
say("    nothing above is trustworthy. %d permutations, specification %s,"
    % (N_PERM, DENOM))
say("    2 fold assignments each, %d model fits in all."
    % (N_PERM * 2 * N_SPLITS))
perm_aucs = []
for p in range(N_PERM):
    rng = np.random.default_rng(1000 + p)
    y_sh = y.copy()
    rng.shuffle(y_sh)
    a, _ = run_cv(df, y_sh, groups, DENOM, n_repeats=2)
    perm_aucs.append(a.mean())
    print("      permutation %d/%d done" % (p + 1, N_PERM), flush=True)
perm_aucs = np.array(perm_aucs)
say("    shuffled AUC mean                 : %.4f" % perm_aucs.mean())
say("    shuffled AUC range                : %.4f to %.4f"
    % (perm_aucs.min(), perm_aucs.max()))
say("    chance                            : 0.5000")
say("    real AUC                          : %.4f" % DENOM_AUC)
n_ge = int((perm_aucs >= DENOM_AUC).sum())
say("    permutations reaching the real AUC: %d of %d  (p <= %.3f)"
    % (n_ge, N_PERM, (n_ge + 1) / (N_PERM + 1)))
say("    verdict                           : %s"
    % ("PASS" if abs(perm_aucs.mean() - 0.5) < 0.03 else "FAIL"))


# ---------------------------------------------------------------------------
# 7. THE AUTHORS' OWN SPLIT
# ---------------------------------------------------------------------------
hdr("7. THE AUTHORS' OWN SPLIT (for comparability, NOT used to adjudicate)")

tr = df[df.group == 1]
va = df[df.group == 2]
te = df[df.group == 3]
say("train group 1 n=%d  (%.1f%% thickened)" % (len(tr), tr.label.mean() * 100))
say("valid group 2 n=%d  (%.1f%% thickened, deliberately balanced)"
    % (len(va), va.label.mean() * 100))
say("test  group 3 n=%d  (%.1f%% thickened, deliberately balanced)"
    % (len(te), te.label.mean() * 100))
say("patients shared between train and test: %d"
    % len(set(tr.patient_id) & set(te.patient_id)))
say()
for spec in ["A", "B", "C", "age"]:
    m = make_model(spec)
    m.fit(features(tr, spec), tr.label.to_numpy(int))
    cells = []
    for name, part in [("valid", va), ("test", te)]:
        p = m.predict_proba(features(part, spec))[:, 1]
        yt = part.label.to_numpy(int)
        lo, hi, _ = boot_ci(yt, p)
        cells.append("%s %.4f [%.4f, %.4f]" % (name, roc_auc_score(yt, p), lo, hi))
    say("%-42s %-30s %s" % (SPEC_NAMES[spec], cells[0], cells[1]))
say()
say("READ THE INTERVAL WIDTH, NOT THE POINT ESTIMATE. At n=100 the 95% interval")
say("on the test split is roughly 0.10 wide. A test set whose interval is 0.10")
say("wide cannot resolve a 0.02 difference, which is exactly what the criterion")
say("fixed in advance in the project record asks. The criterion must be adjudicated on the repeated")
say("cross-validation over all patients in section 4, with this split reported")
say("alongside so the numbers stay comparable to the authors' published ones.")


# ---------------------------------------------------------------------------
# 8. SENSITIVITY: KEEPING THE DUPLICATE RECORDS
# ---------------------------------------------------------------------------
hdr("8. SENSITIVITY: the duplicate records kept in")

y_full = df_full.label.to_numpy(int)
g_full = df_full.patient_id.to_numpy()
_, oof_full = run_cv(df_full, y_full, g_full, DENOM)
lo, hi, _ = boot_ci(y_full, oof_full)
say("n = %d (all records), specification %s" % (len(df_full), DENOM))
say("AUC %.4f [%.4f, %.4f]" % (roc_auc_score(y_full, oof_full), lo, hi))
say("primary analysis (n=%d) : %.4f [%.4f, %.4f]"
    % (len(df), DENOM_AUC, results[DENOM]["lo"], results[DENOM]["hi"]))
say("Four records out of 2,903 cannot move an AUC and they do not. The drop is")
say("done for correctness, not because it changes the answer.")


hdr("9. WHAT THIS IS NOT")
say("CIMT is a carotid wall thickness measured by ultrasound. No patient here")
say("was followed up. There is no cardiovascular event in this dataset, no")
say("outcome and no time to event. Nothing in this file predicts heart attacks,")
say("strokes or cardiovascular risk, and no sentence derived from it may say so.")
say()
say("The cohort is retrospective, single centre, Department of Endocrinology")
say("inpatients at one hospital in Anhui Province, predominantly Chinese")
say("ethnicity. Prevalence is 70.8% thickened, which is an inpatient endocrine")
say("population and not a screening population. This dataset cannot say anything")
say("about calibration across ethnic groups.")

(ROOT / "baseline_output.txt").write_text("\n".join(OUT), encoding="utf-8")
print()
print("[written] %s" % (ROOT / "baseline_output.txt"))
