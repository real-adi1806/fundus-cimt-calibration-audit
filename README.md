# Calibration audit of the China-Fundus-CIMT benchmark

**What this shows, in plain English.** A model that predicts thickened neck-artery walls (carotid intima-media thickness, CIMT) from age and sex, or from age, sex and eye photographs, ranks patients about as well in a differently mixed group (held-out AUC 0.8947), but the risk numbers it gives become wrong. On the public China-Fundus-CIMT data the ranking score (AUC) is identical before and after the correction while the model predicts about 39 per cent more risk than really exists (observed over expected 0.719, 95% CI 0.653 to 0.782). A simple correction that needs only the new population's prevalence repairs most of the problem but not all of it, and adding image features changes none of it. The code here reproduces those numbers.

This is a calibration analysis on one retrospective single-centre dataset. It is not a clinical tool, it does not predict heart attacks or strokes (the dataset contains no events), and it makes no statement about ethnicity or skin tone (the dataset has no such field). The words for the shift are "case-mix shift" and nothing stronger.

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

That is the one command. It runs, in order: the age and sex baseline, frozen ResNet-50 feature extraction for all 5,806 images, a small 128 px image cache, the calibration audit (the paper's primary run), the independent re-derivation, the extra analysis, and the paper's two figures (written to `out/figures/`). CPU only, no GPU. The first run downloads the torchvision ResNet-50 weights (about 100 MB) unless they are already cached on the machine. If you only want the audit and have already built the features, `python run_all.py --from calibration`.

Results are written to `out/calibration_audit_result.json`, and each script prints its numbers to the terminal. After the audit has run, `python src/make_figures.py` draws the paper's Figures 1 and 2 (SVG and 300 dpi PNG) into `out/figures/` from that file; it stops if the file came from the synthetic stand-in or if its refit of model A does not match the file. `expected-output/` holds the recorded outputs of the run the paper reports (the 21 October 2026 rerun: `console_output_2026-10-21_rerun.txt`, `independent_check_out.txt`, `extra_analysis_out.txt`, `calibration_audit_result.json`), so you can compare. `console_output_2026-09-22_rerun.txt` is the earlier run, kept for the comparison the paper's Section 8 describes. The three files with `pre-v3` in their names (`calibration_audit_result.pre-v3.json`, `independent_check_out.pre-v3.txt`, `extra_analysis_out.pre-v3.txt`) were recorded before the paper's version 3, which changed how tied predictions are ordered when computing the expected calibration error (ECE). They are kept unedited for comparison only; the files without `pre-v3` are the ones the paper reports.

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
| `src/make_figures.py` | the paper's two figures (model A calibration plot, O/E against target prevalence), drawn from the audit's output |
| `src/image_test.py` | the optional paired image test |
| `expected-output/` | recorded console outputs and result JSON |

## How to read the output

- AUC is a ranking statistic. It does not change under any strictly increasing transformation of the predictions, so it cannot detect wrong risk magnitudes. `src/extra_analysis.py` prints it to twelve decimal places at three stages to show that.
- O/E is observed over expected. 1.0 is right. 0.719 means the model predicts about 39 per cent more risk than occurs.
- The "one-number correction" shifts the logit by the log-odds difference between the new and old prevalence. It needs only the new population's prevalence.
- The screening-prevalence table resamples the 300 held-out patients to a target prevalence. That isolates the prevalence part of a shift only; it does not capture that a real screening population has milder disease.

## Licence and citation

Code: MIT, see `LICENSE`. Please cite via `CITATION.cff`. The dataset is separate and is CC0 1.0 (the dataset paper's text is CC BY-NC-ND 4.0, which does not cover the deposited data).

## Seeds and versions

Seeds are fixed (`RANDOM_STATE = 42` in the audit, 2,000 bootstrap draws). Results were produced on Python 3.11 with the versions in `requirements.txt`; other versions will probably agree but exact decimals are only checked on those.
