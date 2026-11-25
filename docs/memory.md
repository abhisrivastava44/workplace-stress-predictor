# memory.md - Living project memory (UPDATE AFTER EVERY PHASE)

The agent must read this file first in every session and update it at the end of every phase.

## Project
Explainable and Personalized Workplace Stress Prediction System (office-going adults, 22-40). Team of 5, B.Tech IT final year, Techno Main Salt Lake. Supervisor: Ms. Anandita Saha.

## Current status
- Current phase: Phase 1 complete (awaiting approval for Phase 2)
- Last completed phase: Phase 1 - EDA
- File currently being worked on: none

## Completed
- Phase 0: Created folder structure, Python `venv/`, pinned `requirements.txt` from `pip freeze`, `shared/features.py`, and `ml/src/load_data.py`. Cleaned and validated `data/raw/survey_responses.csv` into `data/processed/clean.csv`.
- Phase 1: Created and executed `ml/notebooks/01_eda.ipynb` on `data/processed/clean.csv` and saved 8 labelled figures to `artifacts/figures/`.

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
- SHAP aggregated back to original features.
- What-If uses stress_index = 100 * (0.5*P(Medium) + 1.0*P(High)).
- No login, no diagnosis claims.

## Dataset facts (fill in during Phase 0)
- Rows: 68 (27 features + 1 target)
- Class balance: Medium 30 (44.1%), High 27 (39.7%), Low 11 (16.2%)
- Issues found: 0 missing values (when read with `keep_default_na=False`), 0 duplicates. Raw headers had leading/trailing/double spaces and cells used en-dashes (`–`), both normalized in `load_data.py`. Dataset is small (< 150 rows) and `Low` class is < 20% share (16.2%).

## Metrics so far (fill in during Phase 2)
| Model | CV macro-F1 | Train score | Test accuracy | Test macro-F1 |
|---|---|---|---|---|

## Open questions for the user
(none yet)

## Known issues / TODO
(none yet)

## Change log
- (date) Initial docs created.
