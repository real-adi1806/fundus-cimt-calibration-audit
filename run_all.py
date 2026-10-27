"""
The one command:  python run_all.py

Runs the whole pipeline in order, stopping at the first failure:
  1. baseline        age + sex logistic regression, out-of-fold predictions
  2. features        frozen ImageNet ResNet-50 features for all 5,806 images
  3. imgcache        128 px cache for the luminance method check
  4. calibration     the paper's primary run (calibration_audit.py)
  5. independent     the separately written re-derivation
  6. extra           announced-risk table, slope interval, AUC invariance
  7. csweep          model B at every ridge penalty C from 1e-6 to 100, with an ECE interval
  8. figures         the paper's Figures 1 and 2 (make_figures.py)
Optional, slow (595 seconds on an Apple M5 laptop on 23 Oct 2026; longer on
older CPUs), not run by default:
  python src/image_test.py   the paired test of image features against age and sex
                             (the +0.0004 result that the paper's Supplement S3 cites)

Needs the dataset in data/ first (see README). CPU only. The first run downloads
the torchvision ResNet-50 weights (about 100 MB) unless they are already cached.
Use  python run_all.py --from calibration  to skip steps 1 to 3 once they have run.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("baseline", "baseline.py"),
    ("features", "extract_features.py"),
    ("imgcache", "build_imgcache.py"),
    ("calibration", "calibration_audit.py"),
    ("independent", "independent_check.py"),
    ("extra", "extra_analysis.py"),
    ("csweep", "c_sweep.py"),
    ("figures", "make_figures.py"),
]

def main():
    start = 0
    if "--from" in sys.argv:
        name = sys.argv[sys.argv.index("--from") + 1]
        names = [s[0] for s in STEPS]
        if name not in names:
            sys.exit("--from must be one of: %s" % ", ".join(names))
        start = names.index(name)
    if start == 0 and not os.path.exists(os.path.join(ROOT, "data", "data_info.json")):
        sys.exit("data/data_info.json not found. Download the dataset first: see "
                 "'Get the data' in README.md.")
    needed = [os.path.join(ROOT, "baseline_oof_predictions.csv"),
              os.path.join(ROOT, "features", "patient_features_mean.npy")]
    if start >= 3 and not all(os.path.exists(p) for p in needed):
        sys.exit("The baseline and feature steps have not been run, so there is "
                 "nothing to audit. Run `python run_all.py` without --from first "
                 "(see 'Run it' in README.md).")
    for name, script in STEPS[start:]:
        print("\n=== %s: %s ===" % (name, script), flush=True)
        code = subprocess.call([sys.executable, os.path.join(ROOT, "src", script)], cwd=ROOT)
        if code != 0:
            print("STOPPED: %s exited with code %d" % (script, code))
            sys.exit(code)
    print("\nAll steps finished. Machine-readable result: out/calibration_audit_result.json")

if __name__ == "__main__":
    main()
