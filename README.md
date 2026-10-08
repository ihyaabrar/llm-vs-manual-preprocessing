# Frozen LLM-Generated Preprocessing for Medical-Claims Adherence Classification

Code, frozen LLM pipelines, and results for the article

> H. Yuliansyah, I. N. Abrar, and M. K. Biddinika, "Frozen LLM-Generated Preprocessing for
> Medical-Claims Adherence Classification: An Internal Model-Sensitivity Study,"
> *Indonesian Journal of Electronics, Electromedical Engineering, and Medical Informatics*, 2026 (in press).

The analysis is split into stage-by-stage notebooks that recompute the results from the
frozen LLM code and the CC0 dataset. No API calls are made at run time. Each notebook
writes its tables to `results/*.xlsx` and its figures to `results/*.png`, and embeds all
outputs so they can be read directly on GitHub. The result files the manuscript cites are
in `outputs/`.

## Notebooks

| Notebook | Stage |
|---|---|
| `01_data_profiling_eda.ipynb` | Cleaning, class balance, descriptive statistics, EDA, LLM data profile |
| `02_llm_generation_prompts.ipynb` | System prompt, raw responses, frozen code, target-correlation sensitivity |
| `03_preprocessing_pipelines.ipynb` | The five pipelines, feature sets, 18-vs-17 demonstration |
| `04_cross_validation_equivalence.ipynb` | Repeated 5 × 10-fold CV, corrected equivalence (accuracy, MCC, AUC) |
| `05_model_sensitivity_robustness.ipynb` | Without UNITSTOTAL, ablation, LightGBM vs Logistic Regression |
| `06_interpretation_shap_calibration.ipynb` | SHAP, calibration, error overlap, subgroup intervals |
| `07_revision_tables.ipynb` | Tables added or expanded during the reviewer revision |

## Where each table and figure comes from

Numbering follows the published article (Tables 1–15, Figs. 1–5).

| Article | Notebook | File |
|---|---|---|
| Table 1 — predictor variables and outcome | — | variable definitions from the dataset documentation |
| Table 2 — descriptive statistics, 11 predictors | 01 | `results/01_profiling.xlsx` |
| Table 3 — the five pipelines and their design differences | 03 | `results/03_pipeline_features.xlsx` |
| Table 4 — transformations and fold-wise fitted components | — | `llm_raw/*.py` (the frozen code itself) |
| Table 5 — classifier configuration and convergence | 07 | `results/07_table7_estimator_config.xlsx` |
| Table 6 — CV and locked-holdout performance (LightGBM) | 07 | `outputs/final_canonical_results.json` (`pipelines`); CV column also in `results/07_table13_model_sensitivity.xlsx` |
| Table 7 — corrected equivalence with TOST p | 04, 07 | `results/07_table9_equivalence_tost.xlsx`, `outputs/revision_core_results.json` |
| Table 8 — pairwise Cohen's d_z | 07 | `results/07_table10_cohen_dz.xlsx` |
| Table 9 — ablation | 05 | `results/05_ablation.xlsx` |
| Table 10 — repeated validation, holdout bootstrap, subgroup gaps | — | `outputs/advanced_validation_results.json` (see note 1) |
| Table 11 — model-class sensitivity | 05, 07 | `results/07_table13_model_sensitivity.xlsx`, `outputs/revision_core_results.json` (`part_B_logreg`) |
| Table 12 — calibration | 06 | `results/06_calibration.xlsx`, `outputs/final_canonical_results.json` |
| Table 13 — analysis without UNITSTOTAL | 05, 07 | `results/07_table15_no_unitstotal.xlsx`, `outputs/revision_core_results.json` (`no_units`) |
| Table 14 — subgroup performance with intervals | 06 | `outputs/subgroup_ci_results.json`; `results/06_subgroup_ci.xlsx` (subset) |
| Table 15 — comparison with related work | — | literature, no computation |
| Figs. 1–2 | — | drawn diagrams, not analysis output |
| Figs. 3–5 | 04, 06 | `results/04_cv_boxplot.png`, `results/06_error_overlap.png`, `results/06_calibration.png`, `results/06_shap.png` (notebook versions of the article figures; the article's Fig. 3 shows the single 10-fold run) |

## Resampling protocols

- **Single stratified 10-fold**, `random_state=42`: Tables 6, 8, 9, 11 and 12 and Figs. 3–5
  (`outputs/final_canonical_results.json`).
- **Repeated 5 × 10-fold**, seeds `2026 + r`: Tables 7 and 13 and the corrected equivalence
  tests (`outputs/revision_core_results.json`, reproduced by notebook 04).

Descriptive statistics (Table 2) are computed on the full analysis set of 24,071 records,
not on the training partition.

## Notes on reported values

These notes reconcile three values in the published article with the result files. None of
them changes a conclusion: every corrected 90% interval lies inside the ±0.01 margin in all runs.

1. **Table 10, repeated-CV columns.** The repeated-CV accuracy, MCC, AUC and corrected 90%
   intervals in Table 10, and the range "0.8198 to 0.8230" quoted in the Abstract, Results and
   Conclusion, come from an earlier repeated run (`outputs/advanced_validation_results.json`,
   `repeated_cv` and `corrected_pairwise`). The protocol described in Algorithm 1 (seed 2026 + r)
   is the one in `outputs/revision_core_results.json` and notebook 04, which gives a mean-accuracy
   range of 0.8204 to 0.8229 and the intervals reported in Table 7. The holdout bootstrap and
   subgroup-gap columns of Table 10 come from the same locked holdout and are unaffected.
2. **CAAFE single-run difference (Results, Section III-A).** The value "−0.0028" is CAAFE minus
   Manual Expert. In the Manual-minus-candidate convention of Tables 7 and 8 it is +0.0028, the
   same direction as the repeated-CV estimate (+0.0010), not the opposite sign.
3. **SHAP rank correlation (Results, Section III-F).** In the final five-pipeline run the
   pairwise rank correlations on shared features range from 0.70 to 0.99 (mean 0.86;
   `outputs/final_canonical_results.json`, `shap`). The range 0.36 to 0.98 in the article comes
   from an earlier four-pipeline analysis.

## Contents

- `llm_raw/` — verbatim model responses and the frozen preprocessing code that was executed
- `results/` — every table (`.xlsx`) and figure (`.png`) the notebooks produce
- `outputs/` — the recorded result files cited by the article
- `Final Prepared Dataset - Diabetes and Hypertension Data.xlsx` — the analytic workbook,
  redistributed from Mendeley Data v2 (DOI: 10.17632/zkp7sbbx64.2) under CC0 1.0

## Running

```bash
pip install -r requirements.txt
jupyter notebook
```

Notebooks 04, 05 and 07 refit the pipelines across many folds and take roughly 10–30
minutes each on a laptop CPU.

## License and citation

Code is released under the MIT License (see `LICENSE`); the dataset remains under CC0 1.0.
Citation metadata is in `CITATION.cff`.
