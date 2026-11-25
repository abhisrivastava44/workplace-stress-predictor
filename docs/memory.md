# memory.md - Living project memory (UPDATE AFTER EVERY PHASE)

The agent must read this file first in every session and update it at the end of every phase.

## Project
Explainable and Personalized Workplace Stress Prediction System (office-going adults, 22-40). Team of 5, B.Tech IT final year, Techno Main Salt Lake. Supervisor: Ms. Anandita Saha.

## Current status
- Current phase: Phase 0 complete (awaiting approval for Phase 1)
- Last completed phase: Phase 0 - Setup and data audit
- File currently being worked on: none

## Completed
- Phase 0: Created folder structure, Python `venv/`, pinned `requirements.txt` from `pip freeze`, `shared/features.py`, and `ml/src/load_data.py`. Cleaned and validated `data/raw/survey_responses.csv` into `data/processed/clean.csv`.

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
