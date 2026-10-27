"""
Does the choice of ridge penalty C change the calibration result?

A reader can fairly ask whether model B's chosen C (0.0001, the strongest on the
audit's grid) shrank the image weights to nothing and so "found no signal" by
construction. The audit chose C by 3 fold inner cross validation on the training
split only, with the held out rows untouched. This script shows the answer does not
depend on that choice: it refits model B at every C from 1e-6 to 100 and prints the
inner CV AUC (the quantity C was chosen on) beside the held out result.

For each C: inner CV AUC on the training split (same folds and seed as pick_C),
then held out AUC, O/E, calibration intercept, calibration slope and ECE, with a
bootstrap 95% interval on ECE (the audit's own ece() and boot_ci(), same seed).
Model A (age and sex, C=1 as in the audit) is printed first for comparison.

The ECE interval is here, not in calibration_audit.py, so that file's output and
expected-output/ are untouched.

Needs the dataset (see README). Run:  python src/c_sweep.py
Its recorded output is expected-output/c_sweep_out.txt.
"""
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibration_audit as ca  # noqa: E402

warnings.filterwarnings("ignore")

C_WIDE = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def inner_cv_auc(Xtr, ytr, C, n_cov):
    """Mean AUC over the same 3 stratified folds pick_C uses."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=ca.RANDOM_STATE)
    return float(np.mean([
        roc_auc_score(ytr[va], ca.fit_predict(Xtr[tr], ytr[tr], Xtr[va], C=C, n_cov=n_cov))
        for tr, va in skf.split(Xtr, ytr)]))


def main():
    loaded = ca.load_real()
    if loaded is None:
        sys.exit("The dataset features are not on disk, so there is nothing to sweep. "
                 "See 'Get the data' in README.md.")
    df, X_img, _ = loaded
    y = df["label"].values.astype(int)
    A = df[["age", "sex"]].values.astype(np.float64)
    B = np.hstack([A, X_img]).astype(np.float64)
    tr = df["group"].values == 1
    te = ~tr

    print("C sweep for model B (age+sex+image), trained on %d, held out %d, held out prevalence %.3f"
          % (tr.sum(), te.sum(), y[te].mean()))
    print("The audit's grid is %s and it picked C=%g for B." % (
        ", ".join("%g" % c for c in ca.C_GRID), ca.pick_C(B[tr], y[tr], n_cov=2)))
    print()
    hdr = ("%-12s %8s %8s %7s %8s %7s %7s   %s" %
           ("model", "innerCV", "AUC", "O/E", "calInt", "calSlp", "ECE", "ECE 95% CI"))
    print(hdr)
    print("-" * len(hdr))

    def line(name, cv, p):
        r = ca.evaluate(y[te], p)
        lo, hi = ca.boot_ci(y[te], p, ca.ece)
        print("%-12s %8s %8.4f %7.3f %8.3f %7.3f %7.4f   [%.4f, %.4f]"
              % (name, cv, r["auc"], r["OE"], r["cal_intercept"], r["cal_slope"], r["ECE"], lo, hi))
        return r

    pa = ca.fit_predict(A[tr], y[tr], A[te], C=1.0, n_cov=0)
    line("A (C=1)", "", pa)
    oes = []
    for C in C_WIDE:
        p = ca.fit_predict(B[tr], y[tr], B[te], C=C, n_cov=2)
        oes.append(line("B C=%g" % C, "%.4f" % inner_cv_auc(B[tr], y[tr], C, 2), p)["OE"])
    print()
    print("Held out O/E for B across the grid: min %.3f, max %.3f (model A: %.3f)."
          % (min(oes), max(oes), ca.evaluate(y[te], pa)["OE"]))


if __name__ == "__main__":
    main()
