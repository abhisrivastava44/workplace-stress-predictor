# PRD.md - Project Requirements Document

## 1. Project name
Explainable and Personalized Workplace Stress Prediction System (Office-Going Adults, Age 22-40)

Academic title (from synopsis): "Workplace Stress Prediction for Office-Employees: A Machine Learning Approach" - Final Year Project, PROJ-IT781, B.Tech IT, Techno Main Salt Lake. Supervisor: Ms. Anandita Saha.

## 2. What we are building
A web app where an office-going adult answers a short lifestyle/workplace questionnaire. The app then shows:
1. Predicted stress level: Low / Medium / High, with class probabilities.
2. Why: the top factors pushing the prediction up or down (SHAP).
3. What can I change: a What-If simulator where the user edits modifiable habits (sleep, exercise, screen time, overtime, etc.) and sees how the predicted stress changes.
4. Personalized suggestions generated from the top modifiable risk factors.
5. (Research view) How stress drivers differ across Junior, Mid-Manager and Senior job levels.

## 3. Why this is not "just another predictor"
Predicting a score from tabular inputs with Random Forest + FastAPI is common. Our contribution is:
- Our own survey dataset on workplace stress (not a Kaggle download), covering workplace, family, financial and lifestyle factors together.
- Per-prediction explainability (SHAP), not just an output label.
- A What-If decision-support layer and rule-based recommendations on top of the model.
- Stratified analysis across career levels.

## 4. Goals
- G1: Train and compare Random Forest and XGBoost on the survey data; pick the best by cross-validated macro-F1.
- G2: Serve the full preprocessing + model as one saved pipeline through FastAPI.
- G3: Explain every prediction with ranked contributing factors.
- G4: Provide a What-If simulator and recommendations.
- G5: Produce report-ready outputs: metrics table, confusion matrix, global SHAP plot, job-level analysis charts.
- G6: Deploy a public demo link.

## 5. Non-goals (do NOT build)
- No medical or clinical diagnosis. This is an educational wellness estimator.
- No user accounts / login in the core version.
- No deep learning, no LLMs inside the prediction path.
- No collection of names, emails or any personal identifiers.
- Journal-text NLP sentiment is OPTIONAL stretch only, and only after everything else is done.

## 6. Target users
- Primary: office-going adults 22-40 who want to understand their stress drivers.
- Secondary: faculty/examiners reviewing the project; HR-style viewers wanting aggregate insights.

## 7. Dataset (collected by the team via Google Forms)
- 27 input questions + 1 target. All questions are categorical/ordinal (single-choice).
- Target: Overall Stress Level = Low | Medium | High.
- Exact column names, allowed values and their order are defined in docs/Architecture.md (section "Feature schema"). Treat that as the single source of truth.
- The raw CSV is exported from Google Forms and placed at data/raw/survey_responses.csv by the user. Do not invent or synthesize data. If the file is missing, stop and ask.

## 8. Functional requirements
Must have:
- FR1: Questionnaire form with all 27 inputs (dropdown / radio), validated against allowed values.
- FR2: POST /predict returns predicted level, probabilities, top factors, recommendations.
- FR3: Result page: stress level badge, probability bars, SHAP factor bar chart, recommendations list.
- FR4: Disclaimer shown on form and result pages.
Should have:
- FR5: POST /whatif returns baseline vs scenario predictions for edited modifiable fields.
- FR6: What-If UI: sliders/selects for modifiable fields with live before/after comparison.
- FR7: Insights page: stress distribution by job level and top drivers per job level (static data produced offline).
Could have:
- FR8: Progress tracking over time (store anonymous check-ins, line chart). Stretch.
- FR9: Journal text sentiment feature. Stretch.

## 9. Success metrics
- Cross-validated macro-F1 and accuracy reported for both models, with train vs CV gap shown (overfitting check).
- Every prediction response includes at least 3 top factors.
- What-If response time under 2 seconds locally.
- Public demo works end to end from a phone browser.

## 10. Known limitations (must appear in the report and in the UI footer)
- Both features and the target are self-reported by the same person, so some correlation is perception bias.
- What-If shows what the MODEL predicts for changed inputs. It is an association, not proof that changing a habit will cause lower stress.
- Dataset is a convenience survey sample; results may not generalize to all workers.

## 11. Disclaimer text (use verbatim)
"This tool estimates a workplace stress level from self-reported lifestyle and work factors for educational purposes. It is not a medical or clinical assessment. If you are struggling, please speak to a qualified professional."
