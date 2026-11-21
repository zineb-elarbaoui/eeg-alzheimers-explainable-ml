# Reliable and Explainable Machine Learning for Alzheimer's Disease Detection Using EEG Signals

Independent replication and robustness analysis of an EEG-based Alzheimer's disease (AD) classification study, using leakage-free leave-one-subject-out cross-validation and SHAP-based interpretation.

> **Reference study:** Shen L, Tong W, Zhao Y, Wu B, Zhang P, Kan J (2026). *Explainable machine learning for Alzheimer's disease characterization using small-sample EEG data.* Front. Aging Neurosci. 18:1900857.

---

## Table of contents

1. [Overview](#overview)
2. [Research questions](#research-questions)
3. [Dataset](#dataset)
4. [Pipeline](#pipeline)
5. [Repository structure](#repository-structure)
6. [Getting started](#getting-started)
7. [Key results](#key-results)
8. [Limitations](#limitations)
9. [Project status](#project-status)
10. [References](#references)
11. [Acknowledgements](#acknowledgements)

---

## Overview

Early detection of Alzheimer's disease is hard: neuroimaging and biomarker panels are expensive and not always accessible. Resting-state EEG is non-invasive and low-cost, and AD leaves measurable traces in it:

- **Spectral slowing**: more delta/theta power, less alpha/beta power.
- **Reduced signal complexity**: captured here with permutation entropy and differential entropy.
- **Altered scalp topography**: captured here with hemispheric asymmetry indices.

With small cohorts and hundreds of correlated features, EEG machine-learning studies can easily produce optimistic results if any data-dependent step (feature selection, scaling, imputation) touches the test subject. This project therefore has two parts:

| Part | Goal |
| --- | --- |
| **Replication** | Reproduce the published pipeline and its four linear models, with every data-dependent step refit inside each training fold. |
| **Extension** | Stress-test the pipeline: how robust are the findings to feature-selection choices, training-subject composition, number of features, demographics, and model choice? No new model families are added. |

The goal is to measure **robustness**, not to chase a new best score.

## Research questions

| # | Question |
| --- | --- |
| RQ1 | Can the published pipeline and results be reproduced independently? |
| RQ2 | How much does performance depend on each part of the feature-selection pipeline (AUC ranking, Spearman pruning, selection itself)? |
| RQ3 | Are the selected EEG features stable across training subsets and across the number of selected features *k*? |
| RQ4 | Do the four linear models converge on the same EEG patterns associated with AD? |
| RQ5 | How sensitive are the findings to *k*, epoch sampling, age and sex, and random seeds? |

## Dataset

**AHEPA dataset** ([OpenNeuro ds004504](https://openneuro.org/datasets/ds004504); Miltiadous et al., 2023): resting-state, eyes-closed EEG, 19 scalp channels, 500 Hz, recorded at AHEPA University Hospital, Thessaloniki, Greece. The preprocessed derivatives (ASR + ICA already applied by the dataset authors) are used.

| Group | Diagnosis | n (full dataset) | Used here |
| --- | --- | --- | --- |
| A | Alzheimer's disease (AD) | 36 | Yes |
| C | Healthy control (HC) | 29 | Yes |
| F | Frontotemporal dementia (FTD) | 23 | No |

**Working cohort: 65 subjects (36 AD, 29 HC).** The reference paper analyses 63 subjects (36 AD, 27 HC) because two HC derivative files could not be read; in this project all 65 files loaded. Small differences from the paper's numbers are expected for this reason.

The data are **not** included in this repository. Download them from OpenNeuro (see [Getting started](#getting-started)).

## Pipeline

```
AHEPA derivatives (.set)
   -> 19 channels -> common average reference -> 1-40 Hz band-pass (FIR, Hamming)
   -> resample 500 -> 100 Hz -> discard first 60 s -> 5 s non-overlapping epochs
   -> amplitude-based epoch rejection (3 rules)
   -> cap at 60 random epochs per subject (seed 2026)
   -> 373 features per subject
   -> fold-wise LOOCV (65 folds) -> 4 linear models
   -> ablation, stability, cross-model SHAP, sensitivity analyses
```

### Step overview

| Step | Title | Part | Script |
| --- | --- | --- | --- |
| 1 | Acquire and define the cohort | Replication | `01_define_cohort.py` |
| 2 | Preprocess EEG | Replication | `02_preprocess.py` |
| 3 | Extract spectral and complexity features | Replication | `03_extract_features.py` |
| 4 | Leakage-free LOOCV framework | Replication | `04_loocv_framework.py` |
| 5 | Reproduce the four linear models | Replication | `05_run_loocv.py` |
| 6 | Feature-selection ablation | Extension | `06_feature_selection_ablation.py` |
| 7 | Feature stability | Extension | `07_feature_stability.py` |
| 8 | Cross-model interpretation | Extension | `08_cross_model_interpretation.py` |
| 9 | Sensitivity analyses | Extension | `09_sensitivity_analyses.py` |
| 10 | Final research conclusion | Both | Write-up |

### Features (373 per subject)

| Family | Count |
| --- | --- |
| Absolute spectral power (delta, theta, alpha, beta × 31 spatial units) | 124 |
| Relative spectral power | 124 |
| Differential entropy | 31 |
| Permutation entropy (m = 3, τ = 1) | 31 |
| Spectral asymmetry (5 homologous region pairs × 4 bands) | 20 |
| Theta/alpha ratio | 11 |
| Theta/beta ratio | 11 |
| Slow/fast ratio ((δ+θ)/(α+β)) | 11 |
| PE/DE asymmetry | 10 |

The 31 spatial units are the 19 channels, 10 lateralised regions, midline, and global. Spectral features use Welch PSD (2.56 s Hann windows, 50% overlap).

### Leakage-free LOOCV

Every data-dependent step is refit **inside each of the 65 training folds**. The held-out subject is only ever transformed or predicted:

1. Median imputation (training medians only)
2. Remove zero-variance features
3. Rank features by univariate |AUC − 0.5|, keep the top 120
4. Spearman redundancy pruning (|ρ| ≥ 0.90) down to *k* = 12 features
5. Z-score standardisation (training mean/SD only)
6. Fit the classifier, then apply the identical transforms to the held-out subject

The framework was verified with a synthetic-data self-test and a direct leakage check (a fold's selection matches an independent fit on its 64 training subjects).

### Models

Ridge Logistic, ElasticNet Logistic, Linear SVM, and Shrinkage LDA, with the reference paper's hyperparameters.

## Repository structure

> Adjust to match your actual layout.

```
.
├── 01_define_cohort.py
├── 02_preprocess.py
├── 03_extract_features.py
├── 04_loocv_framework.py
├── 05_run_loocv.py
├── 06_feature_selection_ablation.py
├── 07_feature_stability.py
├── 08_cross_model_interpretation.py
├── 09_sensitivity_analyses.py
├── outputs/
│   ├── cohort_manifest.csv       # subject_id, label, age, sex, MMSE, path, status
│   ├── features.csv              # 65 subjects x 373 features
│   └── step9_sensitivity/        # k sweep, demographics, seed results
├── docs/                         # methods log, roadmap, study notes
└── README.md
```

## Getting started

### Requirements

- Python 3.10+
- `mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `shap`, `matplotlib`

```bash
pip install mne numpy scipy pandas scikit-learn shap matplotlib
```

### Run

1. Download OpenNeuro `ds004504` (including the `derivatives/` folder) and set the dataset path in the scripts.
2. Run the scripts in order:

```bash
python 01_define_cohort.py
python 02_preprocess.py
python 03_extract_features.py
python 04_loocv_framework.py      # framework + self-tests
python 05_run_loocv.py            # four linear models
python 06_feature_selection_ablation.py
python 07_feature_stability.py
python 08_cross_model_interpretation.py
python 09_sensitivity_analyses.py
```

Random seeds are fixed (epoch sampling uses seed 2026; solvers use `random_state=2026`).

## Key results

### Replication (Step 5, LOOCV, *k* = 12)

| Model | Our AUC | Paper AUC | Our ACC | Paper ACC |
| --- | --- | --- | --- | --- |
| Ridge Logistic | 0.876 | 0.912 | 0.831 | 0.857 |
| ElasticNet Logistic | 0.882 | 0.910 | 0.831 | 0.857 |
| Linear SVM | 0.851 | 0.914 | 0.738 | 0.810 |
| Shrinkage LDA | 0.862 | 0.862 | 0.800 | 0.857 |

All AUCs fall inside the band the paper itself reports across epoch resamples (Ridge: 0.829–0.921). Differences are expected from the 65- vs 63-subject cohort and from epoch sampling. The same feature families are recovered (frontal theta, occipital slow/fast and theta/alpha ratios, temporal theta).

### Extension

- **Ablation (Step 6):** removing Spearman pruning lowered AUC for all four models (most for Shrinkage LDA: 0.862 → 0.767). Random feature ordering gave lower mean AUC, but paired bootstrap intervals include zero. Removing selection entirely had model-dependent effects (ElasticNet unchanged at 0.882).
- **Stability (Step 7):** six features are selected in ≥ 90% of LOOCV folds, but exact feature sets overlap far less under 80% subsampling (mean Jaccard 0.553 → 0.203). Stability is much higher at family/region level. Spectral power and ratios dominate; entropy and asymmetry features are almost never selected.
- **Cross-model interpretation (Step 8):** strong agreement under the shared selection (mean rank correlation 0.95), weaker without selection (0.72) but still high at family–region level (0.89). Higher relative theta pushes predictions toward AD; higher occipital alpha pushes away, consistent with spectral slowing. Univariate ranking agrees poorly with the multivariate models.
- **Sensitivity (Step 9):** AUC varies by about 0.02 or less per model over *k* = 4–16. Adding age and sex does not improve AUC and neither is ever selected. Sex is imbalanced between groups (Fisher p = 0.026), so demographic confounding is *not strongly supported* but is **not ruled out**.

**Central takeaway:** stable AUC does not imply stable feature identity. Several different feature subsets can give similar predictions when features are correlated.

## Limitations

- Single public, single-site cohort of 65 subjects; no external validation. AD and HC are clinical extremes (mean MMSE 17.8 vs 30).
- *k* and the model were chosen from the same LOOCV profile used to report AUC (non-nested), so results are exploratory upper-bound estimates.
- Bootstrap intervals resample only the test subjects of fixed out-of-fold predictions and understate re-training variability. "Within noise" is meaningful; "better/worse" is suggestive, not a formal test.
- LOOCV stability is optimistic by construction (training sets share 63 of 64 subjects).
- Only 19 channels; ASR/ICA cleaning was done upstream by the dataset authors; the fixed 60 s cut does not control for drowsiness.
- Demographic checks cover age and sex only.
- Selected features describe the models' learned patterns, not validated biomarkers.

## Project status

- **Done:** Steps 1–5 (replication), Steps 6–9 (extension runs).
- **Open items before the write-up (Step 10):**
  - Inspect paired-bootstrap intervals for the *k* sweep.
  - Verify that seed settings actually reach the solvers.
  - Compute chance baselines for group-level Jaccard values.
  - Compare selected feature identities (not just counts) with the paper.
  - Optional 63-subject re-run once the two excluded HC subjects are identified.
- **Scope note:** Random Forest and XGBoost were run only as a short pilot (AUC 0.830 and 0.839 vs 0.851–0.882 for the linear models) and are reported as a side note.

## References

- Shen L, Tong W, Zhao Y, Wu B, Zhang P, Kan J (2026). Explainable machine learning for Alzheimer's disease characterization using small-sample EEG data. *Frontiers in Aging Neuroscience*, 18, 1900857.
- Miltiadous A, et al. (2023). A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG. *Data*, 8(6), 95.
- Bandt C, Pompe B (2002). Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, 88, 174102.
- Welch PD (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15, 70–73.

## Acknowledgements

Supervised by Prof. Diego. Dataset by Miltiadous et al. (OpenNeuro ds004504).

## Author

Zineb, Data Engineering & AI, ENSA Safi.

## License

_Add a license (e.g., MIT) before publishing._
