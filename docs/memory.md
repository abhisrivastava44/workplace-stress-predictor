# memory.md - Living project memory (UPDATE AFTER EVERY PHASE)

The agent must read this file first in every session and update it at the end of every phase.

## Project
Explainable and Personalized Workplace Stress Prediction System (office-going adults, 22-40). Team of 5, B.Tech IT final year, Techno Main Salt Lake. Supervisor: Ms. Anandita Saha.

## Current status
- Current phase: Phase 3 Part A complete (awaiting approval for Phase 3 Part B: charts, sanity-check comparison, and pytest tests)
- Last completed phase: Phase 3 Part A - Core SHAP explainability (`ml/src/explain.py`)
- File currently being worked on: `ml/src/explain.py`

## Completed
- Phase 0: Created folder structure, Python `venv/`, pinned `requirements.txt` from `pip freeze`, `shared/features.py`, and `ml/src/load_data.py`. Cleaned and validated `data/raw/survey_responses.csv` into `data/processed/clean.csv`.
- Phase 1: Created and executed `ml/notebooks/01_eda.ipynb` on `data/processed/clean.csv` and saved 8 labelled figures to `artifacts/figures/`.
- Phase 2: Created and ran `ml/src/train.py` and `ml/notebooks/02_model_comparison.ipynb` with nested CV (outer 5x10 RepeatedStratifiedKFold, inner 3-fold GridSearchCV). Saved `artifacts/stress_pipeline.joblib`, `artifacts/label_mapping.json`, `artifacts/metrics.json`, and 2 figures (`09_model_comparison.png`, `10_best_model_confusion_matrix.png`). Verified fresh-process `joblib.load` on raw string input.
- Phase 3 (Part A): Updated `docs/Architecture.md` section 6 and created `ml/src/explain.py` with cached `TreeExplainer`, one-hot aggregation, `stress_push = SHAP(High) - SHAP(Low)` direction logic, input validation, and `explain_one()`. Verified SHAP additivity (max |error| = `6.66e-16` across 10 rows x 3 classes) and tested 3 real rows (Low, Medium, High).

## Phase 1 EDA findings (68-row sample; exploratory only, not proof)
1. Class imbalance is clear: Medium (30, 44.1%) and High (27, 39.7%) make up most of the sample, while Low is only 11 (16.2%); demographics skew young (22-28: 46/68), male (50/68), and single (48/68).
2. Top Spearman correlations with stress suggest `workload_level` (+0.67), `work_life_balance` (-0.63), `job_satisfaction` (-0.61), and `career_growth_pressure` (+0.60) are the strongest linear/monotonic drivers, followed by `overtime_frequency` (+0.55) and `after_hours_communication` (+0.54).
3. In this sample, 100% of respondents reporting `Often` (8/8) or `Very Often` (6/6) overtime are in the High-stress group, and Poor/Very Poor work-life balance heavily clusters in High stress (21/26).
4. By job level, Senior respondents show a higher High-stress share (8/15) than Junior (13/37) or Mid-Manager (6/16), though group sizes are small.
5. **Surprising finding:** Among those sleeping `More than 8 hours` (7 rows), 4 reported High stress (vs 0/16 for `7-8 hours`), suggesting `sleep_quality` (r = -0.51) may matter more than raw sleep duration alone; also `financial_pressure` did not reach the top 10 Spearman correlations.

## Key decisions
- Task: 3-class classification (Low / Medium / High).
- Evaluation: RepeatedStratifiedKFold (5 folds x 10 repeats, random_state=42) per `docs/DATA_NOTES.md`, then fit final pipeline on all rows.
- Models: DummyClassifier, Logistic Regression, Random Forest, and XGBoost; best by CV macro-F1.
- Imbalance handling: `class_weight="balanced"` / sample weights (no SMOTE).
- Single sklearn Pipeline saved with joblib.
- SHAP aggregated back to original features; per-prediction direction uses `stress_push = SHAP(High) - SHAP(Low)` (updated in `docs/Architecture.md` section 6).
- What-If uses stress_index = 100 * (0.5*P(Medium) + 1.0*P(High)).
- No login, no diagnosis claims.

## Dataset facts (fill in during Phase 0)
- Rows: 68 (27 features + 1 target)
- Class balance: Medium 30 (44.1%), High 27 (39.7%), Low 11 (16.2%)
- Issues found: 0 missing values (when read with `keep_default_na=False`), 0 duplicates. Raw headers had leading/trailing/double spaces and cells used en-dashes (`–`), both normalized in `load_data.py`. Dataset is small (< 150 rows) and `Low` class is < 20% share (16.2%).

## Metrics so far (fill in during Phase 2)
| Model | CV Accuracy (mean±std) | CV macro-F1 (mean±std) | Train macro-F1 | Train-CV Gap | ROC-AUC (OvR) | F1 Low / Med / High |
|---|---|---|---|---|---|---|
| Dummy (Majority) | 0.442 ± 0.016 | 0.204 ± 0.005 | 0.204 | -0.000 | 0.500 | 0.00 / 0.61 / 0.00 |
| Logistic Regression | 0.592 ± 0.099 | 0.567 ± 0.123 | 0.916 | +0.349 | 0.763 | 0.51 / 0.56 / 0.67 |
| Random Forest (chosen) | 0.717 ± 0.103 | 0.691 ± 0.112 | 0.922 | +0.231 | 0.871 | 0.62 / 0.72 / 0.76 |
| XGBoost | 0.646 ± 0.130 | 0.599 ± 0.156 | 0.988 | +0.389 | 0.805 | 0.49 / 0.65 / 0.71 |

### Plain-English Summary of Phase 2 Results
1. Because we only have 68 rows, we tested every model across 50 different train/validation splits (5 folds × 10 repeats) with inner 3-fold tuning ("nested CV") so our scores aren't inflated by luck.
2. The Dummy baseline (which always guesses "Medium") gets 44.2% accuracy and 0.204 macro-F1; our models must beat this by a wide margin to be useful.
3. Random Forest came out on top with 71.7% accuracy, 0.691 macro-F1, and 0.871 ROC-AUC, beating Logistic Regression (0.567) and XGBoost (0.599).
4. Notice the Train–CV gap (+0.231 for RF, +0.389 for XGBoost): all models memorize the small 68-row training folds somewhat, which is why shallow trees (`max_depth=3`) helped Random Forest generalize better than XGBoost.
5. On the permutation test (shuffling labels 100 times), Random Forest scored 0.665 vs 0.304 by chance (p = 0.0099), confirming it learned real patterns rather than noise.
6. Honesty check: the `Low` stress class has only 11 rows, so its F1 (0.62) is noticeably weaker than `Medium` (0.72) and `High` (0.76); collecting more `Low` examples later would help most.
7. Finally, we retrained the winning Random Forest pipeline on all 68 rows and saved it as a single `.joblib` file so the backend can pass raw survey answers straight in.

## Open questions for the user
(none yet)

## Known issues / TODO
(none yet)

## Change log
- (date) Initial docs created.
