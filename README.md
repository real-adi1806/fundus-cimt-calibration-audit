# fundus-cimt-calibration-audit

[![smoke](https://github.com/real-adi1806/fundus-cimt-calibration-audit/actions/workflows/smoke.yml/badge.svg)](https://github.com/real-adi1806/fundus-cimt-calibration-audit/actions/workflows/smoke.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](requirements.txt)
[![data: CC0](https://img.shields.io/badge/data-CC0%201.0-lightgrey.svg)](https://doi.org/10.6084/m9.figshare.27907056)

**Calibration audit of the China-Fundus-CIMT benchmark: the code behind the paper.**

**What this shows, in plain English.** A model that predicts thickened neck-artery walls (carotid intima-media thickness, CIMT) from age and sex, or from age, sex and eye photographs, ranks patients about as well in a differently mixed group (held-out AUC 0.8947), but the risk numbers it gives become wrong. On the public China-Fundus-CIMT data the ranking score (AUC) is identical before and after the correction while the model predicts about 39 per cent more risk than really exists (observed over expected 0.719, 95% CI 0.653 to 0.782). A simple correction that needs only the new population's prevalence repairs most of the problem but not all of it, and adding image features changes none of it. The code here reproduces those numbers.

This is a calibration analysis on one retrospective single-centre dataset. It is not a clinical tool, it does not predict heart attacks or strokes (the dataset contains no events), and it makes no statement about ethnicity or skin tone (the dataset has no such field). The words for the shift are "case-mix shift" and nothing stronger.

## Quick start

```
git clone https://github.com/real-adi1806/fundus-cimt-calibration-audit
cd fundus-cimt-calibration-audit
pip install -r requirements.txt
# put the dataset in data/ first: see "Get the data" below
python run_all.py
```

Without the dataset, `python src/calibration_audit.py` runs on a synthetic
stand-in, labels every number SYNTHETIC, and takes a few seconds. That is what
the smoke test above runs. It checks that the code works, not the paper's
numbers; those need the data.

To run the same checks on your own model's predictions instead of this
dataset, use [calibration-shift](https://github.com/real-adi1806/calibration-shift),
the package written alongside this analysis. It takes a CSV of outcomes and
predicted risks.

## Get the data (not included here)

The dataset is released under CC0 1.0 and needs no account or application.

- Figshare: https://doi.org/10.6084/m9.figshare.27907056 (article page https://figshare.com/articles/dataset/27907056; the licence and file list are also visible at https://api.figshare.com/v2/articles/27907056)
- Two files, 1,608,385,785 bytes in total:
  - `Fundus_CIMT_2903.zip` (the 5,806 fundus images)
  - `data_info.json` (age, sex, CIMT thickness, label and split for 2,903 patients)
- Put them in a folder called `data/` at the top of this repository, and unzip the images so the folder `data/Fundus_CIMT_2903 Dataset/` holds 5,806 PNG files. `data/` is git-ignored on purpose; do not commit the data.

Dataset paper: Guo N, Fu W, Li H, et al. High-resolution fundus images for ophthalmomics and early cardiovascular disease prediction. Sci Data 2025;12:568. https://doi.org/10.1038/s41597-025-04930-z

## Run it

```
pip install -r requirements.txt
python run_all.py
```

That is the one command. It runs, in order: the age and sex baseline, frozen ResNet-50 feature extraction for all 5,806 images, a small 128 px image cache, the calibration audit (the paper's primary run), the independent re-derivation, the extra analysis, the penalty sweep (about 20 seconds), and the paper's two figures (written to `out/figures/`). CPU only, no GPU. The first run downloads the torchvision ResNet-50 weights (about 100 MB) unless they are already cached on the machine. If you only want the audit and have already built the features, `python run_all.py --from calibration`.

Results are written to `out/calibration_audit_result.json`, and each script prints its numbers to the terminal. After the audit has run, `python src/make_figures.py` draws the paper's Figures 1 and 2 (SVG and 300 dpi PNG) into `out/figures/` from that file; it stops if the file came from the synthetic stand-in or if its refit of model A does not match the file. `expected-output/` holds the recorded outputs of the run the paper reports (the 21 October 2026 rerun: `console_output_2026-10-21_rerun.txt`, `independent_check_out.txt`, `extra_analysis_out.txt`, `calibration_audit_result.json`), so you can compare. `c_sweep_out.txt` is the output of the penalty sweep described below. `console_output_2026-09-22_rerun.txt` is the earlier run, kept for the comparison the paper's Section 8 describes.

Optional and slower (595 seconds on an Apple M5 laptop on 23 October 2026; longer on older CPUs): `python src/image_test.py`, the paired test of image features against age and sex that the paper's Supplement S3 cites (+0.0004). It is not part of the one command. The end-to-end fine-tuning experiment the paper also mentions (a resnet18 at 128 pixels) is not included here.

## What is in this repository

| path | what it is |
|---|---|
| `run_all.py` | runs the pipeline |
| `src/baseline.py` | age and sex logistic regression, out-of-fold predictions |
| `src/extract_features.py` | frozen ImageNet ResNet-50 features, 2,048 per image |
| `src/build_imgcache.py` | 128 px cache for the one image-derived method check |
| `src/calibration_audit.py` | the primary run: discrimination and the calibration hierarchy under a prevalence shift |
| `src/independent_check.py` | the re-derivation, written without reference to the above |
| `src/extra_analysis.py` | announced-risk table, slope interval, AUC invariance |
| `src/c_sweep.py` | model B at every ridge penalty C from 1e-6 to 100, with an interval on ECE; run on its own with `python src/c_sweep.py` or as the `csweep` step of `run_all.py` |
| `src/make_figures.py` | the paper's two figures (model A calibration plot, O/E against target prevalence), drawn from the audit's output |
| `src/image_test.py` | the optional paired image test |
| `expected-output/` | recorded console outputs and result JSON |

## What the output looks like

The primary run's main table, copied from a fresh clone run on 26 October 2026
(model A is age and sex; model B adds the image features; the model is trained
on the authors' training split at 73.1% prevalence and applied to their held-out
splits at 50%):

```
stratum                         n    prev     AUC meanPred     O/E   calInt  calSlp     ECE
A age+sex -> held out POOL    300   0.500  0.8947    0.696   0.719   -1.324   1.311  0.1995
                             O/E 95% CI [0.653, 0.782]   calInt 95% CI [-1.577, -1.061]   (C=1)
B age+sex+image -> held ou    300   0.500  0.8985    0.695   0.719   -1.333   1.313  0.1991
                             O/E 95% CI [0.654, 0.783]   calInt 95% CI [-1.585, -1.067]   (C=0.0001)
```

The 73.1% and 50% prevalences are the dataset's own training and held-out
split (its `group` column), not a real deployment: nobody deployed this model
anywhere, and the gap between the two splits is a property of how the benchmark
was divided.

The calibration slope of 1.311 (95% CI 1.042 to 1.734, from
`src/extra_analysis.py`) is above 1, which means the predictions are too bunched
together on this held-out split. It reflects the case-mix difference between the
dataset's two splits, and it cannot be fixed without outcome labels from the new
population. Refitting the slope on the held-out labels would use the answers, so
it is not done.

The expected calibration error on 300 patients has an interval too. Both
models' ECE of 0.1995 and 0.1991 come with a bootstrap 95% interval of
[0.1611, 0.2457] and [0.1611, 0.2451] (2,000 draws, the audit's own `ece()` and
seed), printed by `src/c_sweep.py`. So the ECE is somewhere between about
0.16 and 0.25.

On that run, the whole result file and the re-derivation and extra-analysis
outputs were identical to `expected-output/`, the console output differed only
in the line naming the output path, and the two figures were byte-identical to
the paper's.

**One thing that does not repeat exactly.** Extracting the image features
again (step 2) can move model B's numbers slightly. An earlier extraction, made
on a different computer on 15 September 2026, gave model B results differing in
about the fifth decimal, and two printed out-of-fold subgroup values in the fourth (a
subgroup ECE of 0.0279 against 0.0260, and a subgroup AUC of 0.8534 against
0.8533). The paper's numbers come from the 18 October 2026 extraction. Model A,
which carries the headline, does not use the image features and reproduced
exactly from both.

## Does the ridge penalty change the result?

Model B was fitted with C = 0.0001, the strongest penalty on the audit's grid,
and it is fair to ask whether that shrank the image weights to nothing. C was
chosen by 3 fold cross validation inside the training split only, never on the
held-out rows. `src/c_sweep.py` refits B at every C from 1e-6 to 100 and prints
the inner cross validation AUC (what C was chosen on) beside the held-out
result. Recorded in `expected-output/c_sweep_out.txt`:

```
model         innerCV      AUC     O/E   calInt  calSlp     ECE   ECE 95% CI
A (C=1)                 0.8947   0.719   -1.324   1.311  0.1995   [0.1611, 0.2457]
B C=1e-06      0.8529   0.8899   0.711   -1.149   1.804  0.2120   [0.1798, 0.2641]
B C=1e-05      0.8536   0.8958   0.718   -1.296   1.372  0.2035   [0.1635, 0.2479]
B C=0.0001     0.8536   0.8985   0.719   -1.333   1.313  0.1991   [0.1611, 0.2451]
B C=0.001      0.8418   0.8915   0.718   -1.404   1.154  0.1985   [0.1585, 0.2435]
B C=0.01       0.7960   0.8535   0.716   -1.730   0.682  0.1986   [0.1564, 0.2471]
B C=0.1        0.7505   0.8022   0.721   -2.679   0.285  0.2058   [0.1679, 0.2625]
B C=1          0.7338   0.7841   0.727   -4.304   0.157  0.2544   [0.2057, 0.3098]
B C=10         0.7250   0.7772   0.727   -6.484   0.117  0.2843   [0.2292, 0.3389]
B C=100        0.7157   0.7620   0.722  -10.518   0.090  0.3145   [0.2496, 0.3623]
```

The penalty was chosen by inner cross validation, which peaks at 1e-5 to 1e-4,
and the held-out O/E is between 0.711 and 0.727 at every C in the grid, so the
conclusion does not depend on the penalty. Weaker penalties overfit the 2,048
image features: the held-out AUC falls from 0.8985 at C = 0.0001 to 0.7620 at
C = 100, and ECE rises from 0.1991 to 0.3145. The image weights are not zero at
C = 0.0001: model B gives 300 distinct predictions on the held-out patients where
model A gives 101, and they differ from model A's by up to 0.062.

## How to read the output

- AUC is a ranking statistic. It does not change under any strictly increasing transformation of the predictions, so it cannot detect wrong risk magnitudes. `src/extra_analysis.py` prints it to twelve decimal places at three stages to show that.
- O/E is observed over expected. 1.0 is right. 0.719 means the model predicts about 39 per cent more risk than occurs.
- The "one-number correction" shifts the logit by the log-odds difference between the new and old prevalence. It needs only the new population's prevalence.
- The screening-prevalence table resamples the 300 held-out patients to a target prevalence. That isolates the prevalence part of a shift only; it does not capture that a real screening population has milder disease.

## Citing it

GitHub's "Cite this repository" button (from `CITATION.cff`) gives the citation
for the code. The paper is in preparation. Please also cite the dataset paper
above.

## Contributing

Bug reports and questions are welcome as issues. If a number does not reproduce
on your machine, an issue with your Python and package versions and the output
that differs is the most useful thing to send.

## Licence

Code: MIT, see `LICENSE`. The dataset is separate and is CC0 1.0 (the dataset paper's text is CC BY-NC-ND 4.0, which does not cover the deposited data).

## Seeds and versions

Seeds are fixed (`RANDOM_STATE = 42` in the audit, 2,000 bootstrap draws). Results were produced on Python 3.11 with the versions in `requirements.txt`; other versions will probably agree but exact decimals are only checked on those.
