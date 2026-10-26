"""
THE PAPER'S TWO FIGURES, drawn from what calibration_audit.py computes and writes.

    python src/make_figures.py                 writes out/figures/
    python src/make_figures.py --out DIR       writes DIR/

Run src/calibration_audit.py first: this reads its out/calibration_audit_result.json.

Figure 1  Reliability plot for model A on the pooled held-out split (n = 300): raw,
          prior-shifted and intercept-refitted predictions against the diagonal.
          The predictions are refitted here with the audit's OWN functions (imported,
          not rewritten), then checked against the audit's JSON to twelve decimals
          before anything is drawn. The bins are the ECE bins: ten equal-count bins
          over a stable sort, so the points reproduce the stable-sort ECE exactly
          (asserted below). The tie-averaged ECE is a mean over 5,000 orders and has
          no single set of bins, so it cannot be drawn; the figure shows the
          stable-sort binning and the caption must say so.

Figure 2  Observed over expected (O/E) against target prevalence in the resampling
          simulation (experiment D), raw and prior-shifted, read straight from the
          JSON. The raw intervals are the 2.5th and 97.5th percentiles over the 2,000
          resampled draws. The audit records no interval for the prior-shifted O/E,
          so none is drawn.

Nothing in this file is a typed-in result. If the JSON was written by the synthetic
stand-in, or if the refitted predictions disagree with it, the script stops.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calibration_audit as ca

TOL = 1e-12                                   # refitted numbers must match the JSON to this
KEY = "A age+sex"                             # model A, as the audit names it
STAGES = ("raw", "(a) prior shift", "(b) intercept refit")

# Okabe-Ito colours (colour-blind safe), each with its own marker and line style so
# no series is told apart by colour alone. Raw and prior shift keep the same colour
# in both figures.
STYLE = {
    "raw": dict(color="#D55E00", marker="o", linestyle="-", label="Raw"),
    "(a) prior shift": dict(color="#0072B2", marker="s", linestyle="--",
                            label="Prior shift (deployable)"),
    "(b) intercept refit": dict(color="#009E73", marker="^", linestyle=":",
                                label="Intercept refit (ceiling)"),
}
INK = "#222222"
GRID = "#DDDDDD"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.hashsalt": "paper-b",                # stable ids, so a rerun gives the same SVG
})


def load_result():
    path = os.path.join(ca.OUT, "calibration_audit_result.json")
    if not os.path.exists(path):
        sys.exit("No %s. Run src/calibration_audit.py first." % path)
    with open(path) as f:
        res = json.load(f)
    if res.get("mode") != "REAL":
        sys.exit("The audit JSON is mode %r, not REAL. No figure is drawn from synthetic data."
                 % res.get("mode"))
    return res


def model_a_predictions(res):
    """Model A on the pooled held-out split, fitted as experiments A and C fit it, using
    the audit's functions. Returns y and {stage: predictions}, checked against the JSON."""
    loaded = ca.load_real()
    if loaded is None:
        sys.exit("Real cohort not found on disk. Figure 1 needs it (see README).")
    df, _, _ = loaded
    y = df["label"].values.astype(int)
    A = df[["age", "sex"]].values.astype(np.float64)
    tr = df["group"].values == 1

    C = ca.pick_C(A[tr], y[tr], n_cov=0)
    p = ca.fit_predict(A[tr], y[tr], A[~tr], C=C, n_cov=0)
    yy = y[~tr]
    src, tgt = y[tr].mean(), yy.mean()
    shift = np.log(tgt / (1 - tgt)) - np.log(src / (1 - src))
    preds = {
        "raw": p,
        "(a) prior shift": 1 / (1 + np.exp(-(ca._logit(p) + shift))),
        "(b) intercept refit": 1 / (1 + np.exp(-(ca._logit(p) + ca.cal_intercept(yy, p)))),
    }

    # The refit must be the audit's run, not a near copy of it.
    for stage, pv in preds.items():
        mine = ca.evaluate(yy, pv)
        rec = res["experiment_C"]["%s|%s" % (KEY, stage)]
        for k in ("n", "auc", "mean_pred", "OE", "cal_intercept", "ECE", "brier"):
            if abs(mine[k] - rec[k]) > TOL:
                sys.exit("Refitted %s %s = %.17g but the audit JSON says %.17g. Stopping."
                         % (stage, k, mine[k], rec[k]))
    return yy, preds


def reliability_bins(y, p, bins=10):
    """Size, mean prediction and observed rate of each ECE bin: equal-count bins over a
    stable sort, the same binning as calibration_audit.ece."""
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    edges = np.linspace(0, len(y), bins + 1).astype(int)
    return np.array([(b - a, ps[a:b].mean(), ys[a:b].mean())
                     for a, b in zip(edges[:-1], edges[1:]) if b > a])


def save(fig, out, name):
    for ext, kw in (("svg", {"metadata": {"Date": None}}),
                    ("png", {"dpi": 300, "metadata": {"Software": None}})):
        path = os.path.join(out, "%s.%s" % (name, ext))
        fig.savefig(path, facecolor="white", **kw)
        print("wrote %s" % path)
    plt.close(fig)


def figure1(res, out):
    y, preds = model_a_predictions(res)
    assert len(y) >= 20, "ece() changes its bin count below 20 rows; this binning would not match"
    n = len(y)

    print()
    print("FIGURE 1. model A, pooled held-out split, n = %d, %d distinct predictions"
          % (n, len(np.unique(preds["raw"]))))
    print("ten equal-count bins, stable sort (the binning of calibration_audit.ece)")
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color="#888888", linestyle=(0, (4, 3)), linewidth=1,
            label="Perfect calibration", zorder=1)
    observed = None
    for depth, (stage, pv) in zip((5, 4, 3), preds.items()):     # raw drawn on top
        rows = reliability_bins(y, pv)
        ece_bins = float((rows[:, 0] / n * np.abs(rows[:, 2] - rows[:, 1])).sum())
        ece_audit = ca.ece(y, pv)
        assert abs(ece_bins - ece_audit) < TOL, (stage, ece_bins, ece_audit)
        rec = res["experiment_C"]["%s|%s" % (KEY, stage)]
        assert abs(ece_audit - rec["ECE"]) < TOL, (stage, ece_audit, rec["ECE"])
        if observed is None:
            observed = rows[:, 2]
        print("  %-20s ECE from these bins %.6f (stable sort; audit %.6f); "
              "tie-averaged ECE %.4f [%.4f, %.4f]"
              % (stage, ece_bins, rec["ECE"], rec["ECE_tie_mean"],
                 rec["ECE_tie_lo"], rec["ECE_tie_hi"]))
        print("      bin sizes  %s" % " ".join("%d" % v for v in rows[:, 0]))
        print("      mean pred  %s" % " ".join("%.3f" % v for v in rows[:, 1]))
        print("      observed   %s" % " ".join("%.3f" % v for v in rows[:, 2]))
        st = STYLE[stage]
        ax.plot(rows[:, 1], rows[:, 2], color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], linewidth=1.4, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6, label=st["label"], zorder=depth,
                clip_on=False)                  # bins at observed 1.0 sit on the top edge
    same = all(np.array_equal(observed, reliability_bins(y, pv)[:, 2]) for pv in preds.values())
    print("  observed rate in every bin identical across the three stages: %s" % same)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted risk in bin (proportion)")
    ax.set_ylabel("Observed proportion with thickened CIMT")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, out, "figure1_calibration_model_a")


def figure2(res, out):
    d = res["experiment_D"]
    keys = sorted(d, key=lambda k: float(k.split("_")[1]))
    prev = np.array([float(k.split("_")[1]) for k in keys])
    oe = np.array([d[k]["OE"] for k in keys])
    lo = np.array([d[k]["OE_lo"] for k in keys])
    hi = np.array([d[k]["OE_hi"] for k in keys])
    fix = np.array([d[k]["OE_after_prior_shift"] for k in keys])

    print()
    print("FIGURE 2. experiment D, O/E against target prevalence (from the audit JSON)")
    print("  %-10s %8s %8s %8s %16s" % ("target", "O/E", "2.5%", "97.5%", "O/E prior shift"))
    for row in zip(prev, oe, lo, hi, fix):
        print("  %-10.4f %8.4f %8.4f %8.4f %16.4f" % row)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ref = ax.axhline(1.0, color="#888888", linestyle=(0, (4, 3)), linewidth=1,
                     label="O/E = 1 (calibrated)", zorder=1)
    raw, ps = STYLE["raw"], STYLE["(a) prior shift"]
    h_raw = ax.errorbar(prev, oe, yerr=[oe - lo, hi - oe], color=raw["color"],
                        marker=raw["marker"], linestyle=raw["linestyle"], linewidth=1.4,
                        markersize=5.5, capsize=3, elinewidth=1, markeredgecolor="white",
                        markeredgewidth=0.6, label="Raw (bars: 2.5th to 97.5th percentile)",
                        zorder=3)
    h_fix, = ax.plot(prev, fix, color=ps["color"], marker=ps["marker"],
                     linestyle=ps["linestyle"], linewidth=1.4, markersize=5.5,
                     markeredgecolor="white", markeredgewidth=0.6, label=ps["label"], zorder=3)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(prev)
    ax.set_xticklabels(["%.2f" % v for v in prev])
    ax.set_xlabel("Target prevalence of thickened CIMT (proportion)")
    ax.set_ylabel("Observed over expected (O/E)")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[h_raw, h_fix, ref], loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out, "figure2_oe_by_prevalence")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ca.OUT, "figures"),
                    help="folder for the SVG and 300 dpi PNG files (default out/figures)")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    res = load_result()
    figure1(res, out)
    figure2(res, out)


if __name__ == "__main__":
    sys.exit(main())
