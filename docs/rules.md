# rules.md - Rules for the AI agent (read before every task)

## 1. How to work
1. Read docs/PRD.md, docs/Architecture.md, docs/phases.md, docs/design.md and docs/memory.md before writing code.
2. Work on ONE phase at a time, in the order in phases.md. When a phase is done, update docs/memory.md, summarize what changed, and STOP. Wait for the user to say "go to next phase".
3. Before starting a phase, write a short plan (files to create/edit, how you will verify). Then execute.
4. Make targeted changes. Do not rewrite or refactor files unrelated to the current task. Do not rename files or folders defined in Architecture.md.
5. When you edit a file, show the full final path and keep the file complete and runnable.
6. If something in the docs is ambiguous or the data contradicts them, stop and ask. Do not guess silently.

## 2. Data rules
- The only dataset is data/raw/survey_responses.csv provided by the user. Never fabricate, augment with fake rows, or download substitute data. If it is missing, stop and ask.
- Never modify the raw CSV. Write cleaned data to data/processed/.
- Column names, allowed values and category order come ONLY from shared/features.py (which mirrors Architecture.md section 4). Import from it; never re-type category lists elsewhere.
- Report dataset size, class balance and missing/invalid values before any modelling.

## 3. ML rules (the user is new to ML, so be careful and explain)
- Split first, then fit preprocessing on the training set only. No data leakage.
- Use a single sklearn Pipeline (ColumnTransformer + model). Save and load the whole pipeline, not the bare model.
- Never judge a model by one number. Always report CV macro-F1, per-class F1, confusion matrix, and train vs CV gap.
- The dataset is small. Prefer stratified 5-fold CV. Do not claim accuracy from a single lucky split.
- Do not tune on the test set. The test set is used once, at the end.
- Set random_state=42 everywhere.
- Add short comments in ML code explaining WHY each step exists (the user must be able to explain it to a supervisor).
- Be honest about results. If accuracy is modest, say so. Do not inflate metrics or hide weak classes.

## 4. What to use / what to avoid
Use: pandas, numpy, scikit-learn, xgboost, shap, imbalanced-learn (only if needed), FastAPI, Pydantic v2, joblib, React + Vite, Recharts.
Avoid: deep learning, LLM calls in the prediction path, extra heavy libraries, any library not listed without asking first, global mutable state in the backend, hardcoded localhost URLs, committing secrets or .env files.

## 5. Backend rules
- Load the pipeline once at startup, not per request.
- Pydantic models use Literal types generated from shared/features.py so invalid values return 422.
- Every endpoint has a response model and a clear error message.
- CORS origins come from the ALLOWED_ORIGINS env var.
- Write at least basic tests (pytest) for /predict, /whatif and invalid input.

## 6. Frontend rules
- Follow docs/design.md exactly for colors, fonts, spacing.
- API base URL from import.meta.env.VITE_API_URL.
- Handle loading, error and empty states on every page.
- Fully responsive (test at 375px width) and keyboard accessible. Never use color alone to show risk level; always include the text label.
- Show the disclaimer from PRD.md on the form and result pages.

## 7. Safety and wording rules (important)
- Never use the words "diagnosis", "diagnose", "disorder", "clinical" about the app's output. Use "estimated stress level" and "wellness insight".
- Do not give medical advice. Recommendations are general lifestyle suggestions only.
- If the user's answers suggest very high stress, the result page may add a gentle line encouraging them to talk to someone they trust or a professional. Do not attempt anything beyond that.
- Collect no personal identifiers.

## 8. Code quality
- Python: type hints, small functions, PEP8, f-strings. JS: functional components, hooks, no unused code.
- Clear README with setup, run, train, and deploy steps. Keep it updated per phase.
- After each phase run the code (or tests) and report the real output. Do not claim something works without running it.

## 9. Memory discipline
After every phase, update docs/memory.md: what is done, what file you were working on, decisions made, metrics so far, open questions.
