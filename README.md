# Frozen LLM-Generated Preprocessing for Medical-Claims Adherence Classification

Supplementary material for the IJEEEMI submission (Yuliansyah, Abrar, Biddinika):
*An Internal Model-Sensitivity Study*.

The analysis is split into stage-by-stage notebooks that **recompute everything** from the
frozen LLM code and the CC0 dataset — no API calls are made at run time. Each notebook
writes its tables to `results/*.xlsx` and its figures to `results/*.png`, and embeds all
outputs so they can be read directly on GitHub.

## Notebooks

| Notebook | Stage |
|---|---|
| `01_data_profiling_eda.ipynb` | Cleaning, class balance, descriptive statistics, EDA, LLM data profile |
| `02_llm_generation_prompts.ipynb` | System prompt, raw responses, frozen code, target-correlation sensitivity |
| `03_preprocessing_pipelines.ipynb` | The five pipelines, feature sets, 18-vs-17 demonstration |
| `04_cross_validation_equivalence.ipynb` | Repeated 5×10-fold CV, corrected equivalence (accuracy, MCC, AUC) |
| `05_model_sensitivity_robustness.ipynb` | Without UNITSTOTAL, ablation, LightGBM vs Logistic Regression |
| `06_interpretation_shap_calibration.ipynb` | SHAP, calibration, error overlap, subgroup intervals |
| `07_revision_tables.ipynb` | Tables added or expanded during the reviewer revision |

## Which notebook produces which table

| Manuscript table | Notebook | Output file |
|---|---|---|
| Table 2 — class distribution | 01 | `results/01_profiling.xlsx` (`class_balance`) |
| Table 3 — descriptive statistics, all 11 predictors | 01 | `results/01_profiling.xlsx` (`descriptive`) |
| Tables 4–5 — pipelines and comparator asymmetry | 03 | `results/03_pipeline_features.xlsx` |
| Table 6 — complete preprocessing definitions | — | `llm_raw/*.py` (the frozen code itself) |
| Table 7 — estimator configuration and convergence | 07 | `results/07_table7_estimator_config.xlsx` |
| Table 8 — core results under LightGBM | 04 | `results/04_equivalence.xlsx` (`cv_means`) |
| Table 9 — corrected equivalence with TOST p | 07 | `results/07_table9_equivalence_tost.xlsx` |
| Table 10 — pairwise Cohen's d_z | 07 | `results/07_table10_cohen_dz.xlsx` |
| Table 11 — ablation | 05 | `results/05_ablation.xlsx` |
| Table 12 — repeated 5×10-fold validation | 04 | `results/04_equivalence.xlsx` |
| Table 13 — model-class sensitivity | 07 | `results/07_table13_model_sensitivity.xlsx` |
| Table 14 — calibration | 06 | `results/06_calibration.xlsx` |
| Table 15 — analysis without UNITSTOTAL | 07 | `results/07_table15_no_unitstotal.xlsx` |
| Table 16 — subgroup performance with intervals | 06 | `results/06_subgroup_ci.xlsx` |
| Figures 1–2 | — | drawn diagrams, not analysis output |
| Figures 3–9 | 04, 05, 06 | `results/*.png` |

## Resampling protocols

Two designs appear in the manuscript and both are reproduced here:

- **Single stratified 10-fold**, `random_state=42` — Tables 8, 10, 11, 13, 14 and Figures 3–7.
- **Repeated 5×10-fold**, seeds `2026+r` — Tables 9, 12, 15 and the corrected equivalence tests.

Descriptive statistics (Table 3) are computed on the full analysis set of 24,071 records,
not on the training partition.

## Contents

- `llm_raw/` — verbatim model responses and the frozen preprocessing code that was executed
- `results/` — every table (`.xlsx`) and figure (`.png`) the notebooks produce
- Dataset: Mendeley Data v2, CC0 1.0 (DOI: 10.17632/zkp7sbbx64.2)

## Running

```bash
pip install -r requirements.txt
jupyter notebook
```

Notebooks 04, 05 and 07 refit the pipelines across many folds and take roughly 10–30
minutes each on a laptop CPU.

License: code MIT, dataset CC0 1.0.
