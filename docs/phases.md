# phases.md - Build plan (one phase at a time)

Rule: finish a phase, verify it, update docs/memory.md, then STOP and wait for approval.

## Phase 0 - Setup and data audit
Tasks:
- Create the folder structure from Architecture.md. Set up a Python venv and requirements.txt (pinned versions).
- Create shared/features.py from Architecture.md section 4.
- Write ml/src/load_data.py: read data/raw/survey_responses.csv, show original headers, rename to snake_case, normalize en dashes to hyphens, validate every value against allowed lists, drop the Google Forms timestamp column if present.
- Print: row count, class balance of stress_level, missing values, duplicates, invalid values.
Done when: a clean data/processed/clean.csv exists and the audit summary is shown to the user.
Stop and report. If rows < 150 or any class < 20% share, say so clearly and propose options.

## Phase 1 - EDA
Tasks:
- ml/notebooks/01_eda.ipynb: stress distribution, stress vs each key factor (grouped bar / heatmap), correlation of ordinal-encoded features with stress, stress by job_level.
- Save 6-8 clean, labelled figures to artifacts/figures/. Each plot must answer one question (add a one-line caption).
Done when: figures saved and a 5-line findings summary written to memory.md.

## Phase 2 - Model training and comparison
Tasks:
- ml/src/train.py: stratified train/test split, ColumnTransformer pipeline, Random Forest and XGBoost, RandomizedSearchCV (f1_macro), stratified 5-fold CV.
- Save: artifacts/stress_pipeline.joblib, label_mapping.json, metrics.json, confusion matrix and per-class report figures.
- Print a comparison table: model, CV macro-F1 (mean +/- std), train score, test accuracy, test macro-F1, ROC-AUC.
Done when: best model chosen with justification, metrics saved, and a plain-English explanation of results is written for the user.

## Phase 3 - Explainability
Tasks:
- ml/src/explain.py: TreeExplainer, aggregate one-hot SHAP back to original features, function explain_one(row) -> top 5 factors with direction.
- Save global SHAP bar plot and a sample local explanation plot.
- Save a small background sample to artifacts/shap_background.joblib.
Done when: explain_one works on 3 sample inputs (low, medium, high) and the results look sensible; show them to the user.

## Phase 4 - Backend API
Tasks:
- backend/app: /health, /meta, /predict (with top_factors and stress_index) using the saved pipeline.
- Pydantic schemas from shared/features.py. CORS via env var.
- pytest tests for valid input, invalid input, and response shape.
Done when: uvicorn runs locally, /docs works, and tests pass (show real output).

## Phase 5 - Frontend core
Tasks:
- Vite React app following docs/design.md: Home, Questionnaire (grouped into steps with progress bar), Result page (badge, probability bars, SHAP factor chart).
- Fetch options from /meta so the form never hardcodes category lists.
Done when: a full flow works locally from form to result, and layout works at 375px width.

## Phase 6 - What-If and recommendations
Tasks:
- backend/app/services/whatif.py and POST /whatif (only modifiable fields allowed).
- backend/app/services/recommender.py: rule-based suggestions mapped from top increasing-stress modifiable factors (text templates, general lifestyle advice only).
- Frontend What-If page: edit modifiable fields, see before/after stress level, probabilities and stress_index change.
- Add the note: "What-If shows how the model's estimate changes; it does not guarantee real-world results."
Done when: changing sleep from "Less than 5 hours" to "7-8 hours" visibly changes the estimate for a high-stress sample, and tests cover /whatif.

## Phase 7 - Career-level analysis (research outputs)
Tasks:
- ml/src/analysis_by_level.py: stress distribution by job_level, chi-square test of independence (stress vs job_level), top SHAP drivers per job level.
- Export charts to artifacts/figures/ and a small JSON for an Insights page.
- Frontend Insights page showing these.
Done when: figures and a short written interpretation (with honest caveats about sample size) are ready for the report.

## Phase 8 - Deployment and README
Tasks:
- Deploy backend (Render web service) and frontend (static host). Set ALLOWED_ORIGINS and VITE_API_URL.
- Complete README: overview, setup, training, running, API docs, screenshots placeholder, limitations, disclaimer.
Done when: public frontend link works end to end from a phone.

## Phase 9 - Stretch (only if time remains, ask first)
- Progress tracking: anonymous check-ins saved to MongoDB Atlas (random client id in localStorage), line chart of stress_index over time.
- Journal text sentiment (VADER) as an extra feature. Requires retraining with new data; probably skip since the survey has no text field.
