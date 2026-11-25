# DATA_NOTES.md - Facts about the real dataset (OVERRIDES other docs where they conflict)

File: data/raw/survey_responses.csv (Google Forms export, 68 responses, 29 columns)

## 1. Size and class balance
- 68 rows only. Stress level: Medium 30, High 27, Low 11. No duplicate rows.
- This is a SMALL dataset. Consequences (these override Architecture.md section 5 and phases.md Phase 2):
  - Do NOT use a single 80/20 train/test split. The test set would contain only ~2 "Low" rows and the score would be meaningless.
  - Evaluate with RepeatedStratifiedKFold (5 folds x 10 repeats, random_state=42) and report mean +/- std of accuracy and macro-F1. Collect predictions across folds for one pooled confusion matrix.
  - Compare against a majority-class baseline (DummyClassifier) and a Logistic Regression baseline, plus Random Forest and XGBoost. Report all four.
  - Use class_weight="balanced" (RF, LR) or sample weights (XGBoost). Do not use SMOTE.
  - Keep hyperparameter tuning small (shallow trees, few trials) and use nested CV or clearly say tuning scores are optimistic. Do not tune on the same folds used for the final reported score without saying so.
  - After choosing the model, fit the FINAL model on all 68 rows for the app and SHAP.
  - SHAP results are indicative on this sample size; say so in the UI note and in the report.
- Quick reference baseline already measured (5x10 repeated stratified CV, ordinal+one-hot pipeline): majority baseline accuracy 0.44 / macro-F1 0.20; Logistic Regression 0.60 / 0.59; Random Forest 0.70 / 0.67. Re-measure yourself; do not copy these numbers into the report.
- The team may collect more responses later. Make the training script re-runnable on a bigger CSV with no code changes.

## 2. CSV reading gotchas (IMPORTANT)
1. The answer "None" (Childcare Responsibilities: 36 rows, Elderly Care Responsibilities: 16 rows) is a REAL category, but pandas reads the string "None" as missing. ALWAYS read with: pd.read_csv(path, keep_default_na=False). After loading there must be 0 missing values.
2. Column headers have leading/trailing spaces and double spaces (e.g. "  Job Security Concern  ", "Overall Stress Level  "). Strip them before renaming.
3. Values use the en dash character (for example "22–28", "8–9", "5–6 hours", "1–3 hours", "30–60 minutes"). Normalize en dash to hyphen in every value before validating against shared/features.py. Strip whitespace from values too.
4. Column order in the CSV: Timestamp is first; "Current Job Level" is the LAST column (after Overall Stress Level). Never rely on column position; use names.
5. Drop the Timestamp column. It is not a feature.

## 3. Column name mapping (original header after stripping -> snake_case)
Age Group -> age_group
Current Job Level -> job_level
Gender -> gender
Marital Status -> marital_status
Number of Children -> num_children
Number of Dependents -> num_dependents
Working Hours Per Day -> working_hours
Overtime Frequency -> overtime_frequency
Workload Level -> workload_level
Job Satisfaction -> job_satisfaction
Career Growth Pressure -> career_growth_pressure
Job Security Concern -> job_security_concern
Work-related Communication After Office Hours -> after_hours_communication
How often do you think about work outside office hours? -> thinks_about_work_outside
Childcare Responsibilities -> childcare_responsibility
Elderly Care Responsibilities -> elderly_care_responsibility
Family Responsibility Pressure -> family_responsibility_pressure
Financial Pressure -> financial_pressure
Marriage/Relationship Pressure -> relationship_pressure
Relative/Social Pressure -> social_pressure
Average Sleep Duration Per Day -> sleep_duration
Sleep Quality -> sleep_quality
Exercise/Physical Activity -> exercise_frequency
Daily Non-Work Screen Time -> screen_time
Leisure/Relaxation Time -> leisure_time
Daily Commute Time -> commute_time
Work-Life Balance -> work_life_balance
Overall Stress Level -> stress_level (TARGET)

## 4. Observed value notes
- gender in the data: Male 50, Female 16, Prefer not to say 2. "Other" has zero rows but stays an allowed option in shared/features.py and the form (OneHotEncoder handle_unknown="ignore").
- age_group: 22-28 has 46 of 68 rows; job_level: Junior 37, Mid-Manager 16, Senior 15. Stratified analysis by job level has small groups: use descriptive charts, and if a chi-square test is run, use Fisher/Monte-Carlo or clearly flag the low expected counts. Do not over-claim.
- The sample is mostly young, male and single. State this as a limitation; do not present results as general to all workers.
- All other answer values match docs/Architecture.md section 4 after the normalization in section 2 above.
