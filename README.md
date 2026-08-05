# LLM-Generated versus Manual Preprocessing for Medication Adherence Classification

Supplementary material (Yuliansyah, Abrar, Biddinika). The analysis is split into stage-by-stage notebooks that **recompute everything** from the frozen LLM code and the CC0 dataset (no API calls). Each notebook writes its tables to `results/*.xlsx` and its figures to `results/*.png`, and embeds all outputs for viewing directly on GitHub.

| Notebook | Stage |
|---|---|
| `01_data_profiling_eda.ipynb` | Cleaning, class balance, descriptive stats, EDA, LLM data profile |
| `02_llm_generation_prompts.ipynb` | System prompt, raw responses, frozen code, target-corr sensitivity |
| `03_preprocessing_pipelines.ipynb` | The five pipelines, feature sets, 18-vs-17 demonstration |
| `04_cross_validation_equivalence.ipynb` | Repeated 5x10-fold CV, corrected TOST (acc/MCC/AUC) |
| `05_model_sensitivity_robustness.ipynb` | Without UNITSTOTAL, ablation, LightGBM vs Logistic Regression |
| `06_interpretation_shap_calibration.ipynb` | SHAP, calibration, error overlap, subgroup CIs |

- `llm_raw/` — verbatim model responses and frozen preprocessing code
- Dataset: Mendeley Data v2, CC0 (DOI: 10.17632/zkp7sbbx64.2)

```bash
pip install -r requirements.txt
jupyter notebook
```
License: code MIT, dataset CC0 1.0.
