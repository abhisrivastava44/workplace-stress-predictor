# Architecture.md - App flow, folder structure, tech stack, schema

## 1. High-level flow
User -> React form (27 inputs) -> POST /predict (FastAPI)
FastAPI -> Pydantic validation -> DataFrame (exact column order) -> saved sklearn Pipeline (preprocessor + classifier)
-> probabilities + predicted class -> SHAP explanation (aggregated back to original features) -> recommendation rules
-> JSON response -> React result page (badge, probability bars, factor chart, suggestions)
User edits habits -> POST /whatif -> baseline vs scenario predictions -> before/after comparison.

## 2. Tech stack
- ML: Python 3.11, pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn, joblib, scipy
- Backend: FastAPI, Uvicorn, Pydantic v2
- Frontend: React (Vite), plain CSS or Tailwind, Recharts for charts
- Deployment: Render (backend web service), static host for frontend (Render Static Site or Vercel)
- Optional stretch: MongoDB Atlas for progress tracking

## 3. Folder structure
workplace-stress-predictor/
  docs/                      PRD.md, Architecture.md, rules.md, phases.md, design.md, memory.md, START_PROMPT.md
  data/
    raw/survey_responses.csv           (provided by user, never edited)
    processed/clean.csv
  shared/
    features.py                        SINGLE SOURCE OF TRUTH: column names, allowed values, order, modifiable flags
  ml/
    notebooks/01_eda.ipynb
    src/load_data.py                   read CSV, rename columns, validate values
    src/train.py                       CV, compare RF vs XGBoost, tune, save best pipeline
    src/explain.py                     SHAP helpers (aggregate one-hot back to original feature)
    src/analysis_by_level.py           stats + plots per job level
  artifacts/
    stress_pipeline.joblib             full pipeline (preprocessor + model)
    label_mapping.json
    metrics.json
    shap_background.joblib             small background sample for explainer
    figures/                           confusion matrix, SHAP summary, EDA plots (png)
  backend/
    app/main.py                        FastAPI app, CORS, routes
    app/schemas.py                     Pydantic models (Literal types generated from shared/features.py)
    app/services/predictor.py          load pipeline once at startup, predict
    app/services/explainer.py          SHAP per request
    app/services/whatif.py             scenario logic
    app/services/recommender.py        rule-based suggestions
    requirements.txt
  frontend/
    src/pages/ (Home, Questionnaire, Result, WhatIf, Insights)
    src/components/
    src/api.js                         API base URL from env (VITE_API_URL)
  README.md

## 4. Feature schema (single source of truth)
Raw Google Forms CSV headers are the full question text. load_data.py must rename them to the snake_case names below, then validate every value against the allowed list. If a value is not in the list, print the offending rows and stop; do not silently drop or guess.

Ordinal features (encode with OrdinalEncoder using EXACTLY these category orders, low to high):
- age_group: ["22-28", "29-34", "35-40"]   (form uses en dash; normalize to hyphen)
- job_level: ["Junior Level", "Mid-Manager Level", "Senior Level"]
- num_children: ["0", "1", "2", "3+"]
- num_dependents: ["0", "1", "2", "3+"]
- working_hours: ["Less than 8", "8-9", "10-11", "More than 11"]
- overtime_frequency: ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
- workload_level: ["Very Low", "Low", "Moderate", "High", "Very High"]
- job_satisfaction: ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"]
- career_growth_pressure: ["Very Low", "Low", "Moderate", "High", "Very High"]
- job_security_concern: ["Very Low", "Low", "Moderate", "High", "Very High"]
- after_hours_communication: ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
- thinks_about_work_outside: ["Never", "Rarely", "Sometimes", "Often", "Always"]
- childcare_responsibility: ["None", "Low", "Moderate", "High"]
- elderly_care_responsibility: ["None", "Low", "Moderate", "High"]
- family_responsibility_pressure: ["Very Low", "Low", "Moderate", "High", "Very High"]
- financial_pressure: ["Very Low", "Low", "Moderate", "High", "Very High"]
- relationship_pressure: ["Very Low", "Low", "Moderate", "High", "Very High"]
- social_pressure: ["Very Low", "Low", "Moderate", "High", "Very High"]
- sleep_duration: ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"]
- sleep_quality: ["Very Poor", "Poor", "Average", "Good", "Excellent"]
- exercise_frequency: ["Never", "1-2 days/week", "3-4 days/week", "5+ days/week"]
- screen_time: ["Less than 1 hour", "1-3 hours", "3-5 hours", "More than 5 hours"]
- leisure_time: ["Less than 1 hour/day", "1-2 hours/day", "2-3 hours/day", "More than 3 hours/day"]
- commute_time: ["Less than 30 minutes", "30-60 minutes", "1-2 hours", "More than 2 hours"]
- work_life_balance: ["Very Poor", "Poor", "Average", "Good", "Excellent"]

Nominal features (OneHotEncoder, handle_unknown="ignore"):
- gender: ["Male", "Female", "Other", "Prefer not to say"]
- marital_status: ["Single", "Married", "Divorced/Separated"]

Target:
- stress_level: ["Low", "Medium", "High"]  -> label-encode 0,1,2 and save mapping to artifacts/label_mapping.json

Modifiable features (usable in the What-If simulator and recommendations):
sleep_duration, sleep_quality, exercise_frequency, screen_time, leisure_time, commute_time, overtime_frequency, after_hours_communication, thinks_about_work_outside, working_hours, workload_level, work_life_balance.
Non-modifiable (shown read-only in What-If): age_group, job_level, gender, marital_status, num_children, num_dependents, childcare_responsibility, elderly_care_responsibility, and the remaining pressure/satisfaction fields.

Each feature in shared/features.py also carries: display label, question text, and a short human-readable name for the explanation chart.

## 5. ML design
- Split: stratified train/test (80/20) with fixed random_state=42, plus stratified 5-fold CV on the training set. The dataset is small; trust CV more than a single test split.
- Pipeline = ColumnTransformer(ordinal features -> OrdinalEncoder with explicit categories; nominal -> OneHotEncoder) + classifier. Fit ONLY on training data; never fit anything on test data.
- Models: RandomForestClassifier, XGBClassifier. Tune with RandomizedSearchCV (scoring = f1_macro). Handle imbalance with class_weight / sample_weight first; use SMOTE only inside CV folds via imblearn Pipeline if needed.
- Report: accuracy, precision, recall, macro-F1, per-class F1, confusion matrix, ROC-AUC (OvR), train vs CV score gap.
- stress_index (for What-If only): 100 * (0.5 * P(Medium) + 1.0 * P(High)). Range 0-100, higher = more stress.

## 6. Explainability design
- Use shap.TreeExplainer on the classifier, applied to the transformed features.
- Aggregate SHAP values of one-hot columns back to their original feature by summing, so the user sees "gender" or "sleep_duration", not "gender_Male".
- Per prediction: explain the predicted class; return top 5 factors sorted by absolute impact with direction ("increases stress" / "reduces stress") and the user's answer for that feature.
- Global: SHAP summary bar plot saved to artifacts/figures/.

## 7. API contract
GET  /health -> {"status": "ok"}
GET  /meta   -> list of features with label, allowed options in order, modifiable flag; target classes
POST /predict
  request: {all 27 feature fields as strings}
  response: {
    "predicted_level": "High",
    "probabilities": {"Low": 0.05, "Medium": 0.25, "High": 0.70},
    "stress_index": 82.5,
    "top_factors": [{"feature": "sleep_duration", "label": "Sleep duration", "value": "Less than 5 hours", "impact": 0.31, "direction": "increases"}],
    "recommendations": [{"feature": "sleep_duration", "text": "..."}],
    "disclaimer": "..."
  }
POST /whatif
  request: {"base": {27 fields}, "changes": {"sleep_duration": "7-8 hours", "screen_time": "1-3 hours"}}
  response: {"baseline": {predict-style summary}, "scenario": {predict-style summary}, "delta_stress_index": -14.2, "changed_fields": [...]}
Errors: 422 for invalid values (Pydantic Literal), 400 if a What-If change targets a non-modifiable field.

## 8. Deployment notes
- Backend start command: uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
- CORS: allow the frontend origin via env var ALLOWED_ORIGINS.
- Frontend reads VITE_API_URL; never hardcode localhost in committed code.
- Free Render instances sleep when idle; first request may be slow. Mention this on the demo page.
- Pin scikit-learn, xgboost and shap versions in requirements.txt: a joblib pipeline must be loaded with the same scikit-learn version that trained it.
