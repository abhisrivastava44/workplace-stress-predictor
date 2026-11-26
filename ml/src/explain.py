"""
ml/src/explain.py
=================
Phase 3 (Part A) — SHAP explainability for the saved Random Forest pipeline.

How explanations work here:
  1. We load `artifacts/stress_pipeline.joblib` (which contains both the
     ColumnTransformer preprocessor and the fitted RandomForestClassifier)
     and cache the pipeline + shap.TreeExplainer in memory so repeated
     `explain_one()` calls run in milliseconds.
  2. For a given input dict of 27 raw string answers, we transform the row
     through the pipeline's preprocessor (`prep`) and run `TreeExplainer`
     on the Random Forest (`clf`).
  3. One-hot columns (`gender`, `marital_status`) are summed back to their
     original feature so users see "Gender" rather than "gender_Male".
  4. Direction definition (`stress_push`):
       For each original feature, we compute:
           stress_push = SHAP_value_for_High - SHAP_value_for_Low
       Why: In a 3-class model (Low=0, Medium=1, High=2), looking at only the
       predicted class can be confusing (e.g., for a Medium prediction, a factor
       that pushes toward High might look negative for Medium). Defining
       `stress_push = SHAP(High) - SHAP(Low)` gives an intuitive, consistent
       stress-direction score:
         • Positive stress_push → pushes the estimate toward MORE stress ("increases")
         • Negative stress_push → pushes the estimate toward LESS stress ("reduces")
     We rank features by `|stress_push|` and return the top 5.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

# ── Make sure the project root is on sys.path so we can import shared/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.features import (  # noqa: E402
    ALLOWED_VALUES,
    FEATURE_MAP,
    FEATURE_NAMES,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES,
    TARGET,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "stress_pipeline.joblib"
LABEL_MAPPING_PATH = ARTIFACTS_DIR / "label_mapping.json"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "clean.csv"


@lru_cache(maxsize=1)
def _load_resources() -> Tuple[Any, shap.TreeExplainer, List[str], List[str]]:
    """
    Load the saved pipeline and build/cache the TreeExplainer once.
    Returns (pipeline, explainer, transformed_col_to_orig_feature, class_names).
    """
    pipeline = joblib.load(PIPELINE_PATH)
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    # Tree-path dependent TreeExplainer is exact for Random Forests and fast
    explainer = shap.TreeExplainer(clf)

    # Map every transformed column back to its original feature name
    col_to_orig: List[str] = list(ORDINAL_FEATURES)
    ohe = prep.named_transformers_["nominal"]
    for feat_name, cats in zip(NOMINAL_FEATURES, ohe.categories_):
        col_to_orig.extend([feat_name] * len(cats))

    if LABEL_MAPPING_PATH.exists():
        with open(LABEL_MAPPING_PATH, encoding="utf-8") as f:
            classes = json.load(f)["classes"]
    else:
        classes = TARGET["categories"]

    return pipeline, explainer, col_to_orig, classes


def _normalize_shap_array(raw_shap: Any) -> np.ndarray:
    """
    Normalize SHAP output across SHAP versions into shape:
        (n_samples, n_transformed_features, n_classes)
    - Older SHAP returns a list of n_classes arrays of shape (n_samples, n_features).
    - Newer SHAP (e.g. 0.54+) returns one 3-D array of shape (n_samples, n_features, n_classes).
    """
    if isinstance(raw_shap, list):
        return np.stack(raw_shap, axis=-1)
    arr = np.asarray(raw_shap)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D SHAP array or list of 2-D arrays, got shape {arr.shape}")
    return arr


def validate_answers(answers: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate an input answers dict against shared/features.py.
    Raises ValueError with a clear message if fields are missing, unknown, or invalid.
    """
    if not isinstance(answers, dict):
        raise ValueError("Input 'answers' must be a dictionary.")

    expected_set = set(FEATURE_NAMES)
    provided_set = set(answers.keys())

    missing = [f for f in FEATURE_NAMES if f not in provided_set]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    unknown = sorted(provided_set - expected_set)
    if unknown:
        raise ValueError(f"Unknown feature(s) not in schema: {unknown}")

    cleaned: Dict[str, str] = {}
    for feat in FEATURE_NAMES:
        val = answers[feat]
        if not isinstance(val, str):
            raise ValueError(
                f"Invalid type for '{feat}': expected str, got {type(val).__name__}"
            )
        val_norm = val.replace("\u2013", "-").strip()
        allowed = ALLOWED_VALUES[feat]
        if val_norm not in allowed:
            raise ValueError(
                f"Invalid value '{val}' for '{feat}'. Allowed values: {allowed}"
            )
        cleaned[feat] = val_norm

    return cleaned


def _aggregate_by_feature(
    shap_per_transformed: np.ndarray, col_to_orig: List[str]
) -> Dict[str, np.ndarray]:
    """
    Sum transformed-column SHAP values back to original features.
    Returns {orig_feature_name: array of shape (n_classes,)}.
    """
    n_classes = shap_per_transformed.shape[1]
    agg: Dict[str, np.ndarray] = {
        feat: np.zeros(n_classes, dtype=float) for feat in FEATURE_NAMES
    }
    for col_idx, orig_feat in enumerate(col_to_orig):
        agg[orig_feat] += shap_per_transformed[col_idx]
    return agg


def explain_one(answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict stress level and compute top 5 SHAP-based contributing factors
    for a single survey response dict.

    Direction definition:
        stress_push = SHAP_value_for_High - SHAP_value_for_Low
        Positive stress_push -> direction = "increases" (pushes toward higher stress)
        Negative stress_push -> direction = "reduces"   (pushes toward lower stress)
    """
    cleaned = validate_answers(answers)
    pipeline, explainer, col_to_orig, classes = _load_resources()

    # Transform using the fitted preprocessor (no leakage, exact column order)
    df_one = pd.DataFrame([cleaned])[FEATURE_NAMES]
    X_trans = pipeline.named_steps["prep"].transform(df_one)
    probs = pipeline.named_steps["clf"].predict_proba(X_trans)[0]

    pred_idx = int(np.argmax(probs))
    predicted_level = classes[pred_idx]
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
    stress_index = round(100.0 * (0.5 * float(probs[1]) + 1.0 * float(probs[2])), 2)

    # Compute SHAP values for this single transformed row
    raw_shap = explainer.shap_values(X_trans)
    shap_arr = _normalize_shap_array(raw_shap)  # (1, n_trans_features, 3)

    # Aggregate one-hot columns back to the 27 original features
    agg = _aggregate_by_feature(shap_arr[0], col_to_orig)

    # Compute stress_push = SHAP(High) - SHAP(Low) (indices 2 and 0)
    low_idx = classes.index("Low")
    high_idx = classes.index("High")

    factors = []
    for feat in FEATURE_NAMES:
        push = float(agg[feat][high_idx] - agg[feat][low_idx])
        factors.append(
            {
                "feature": feat,
                "label": FEATURE_MAP[feat]["display_label"],
                "value": cleaned[feat],
                "impact": round(push, 4),
                "direction": "increases" if push >= 0 else "reduces",
            }
        )

    # Sort by absolute impact descending and keep top 5
    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "predicted_level": predicted_level,
        "probabilities": prob_dict,
        "stress_index": stress_index,
        "top_factors": factors[:5],
    }


def check_additivity(df: pd.DataFrame, n_rows: int = 10) -> float:
    """
    Verify SHAP additivity:
      expected_value[c] + sum(shap_values[i, :, c]) == predict_proba(X_trans)[i, c]
    Returns the maximum absolute error across the tested rows and classes.
    """
    pipeline, explainer, _, _ = _load_resources()
    sample = df[FEATURE_NAMES].head(n_rows)
    X_trans = pipeline.named_steps["prep"].transform(sample)
    probs = pipeline.named_steps["clf"].predict_proba(X_trans)

    raw_shap = explainer.shap_values(X_trans)
    raw_type = type(raw_shap).__name__
    raw_shape = (
        [np.asarray(a).shape for a in raw_shap]
        if isinstance(raw_shap, list)
        else np.asarray(raw_shap).shape
    )
    print(f"SHAP raw output type: {raw_type}, shape: {raw_shape}")

    shap_arr = _normalize_shap_array(raw_shap)
    base_vals = np.asarray(explainer.expected_value, dtype=float)

    reconstructed = base_vals[None, :] + shap_arr.sum(axis=1)
    max_err = float(np.max(np.abs(reconstructed - probs)))
    return max_err


def main() -> None:
    """Run Part A verification: shape check, additivity check, and 3 sample rows."""
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False)

    print("=" * 65)
    print("PHASE 3 (PART A) — SHAP EXPLAINABILITY VERIFICATION")
    print("=" * 65)

    # 1. Shape & additivity check on first 10 rows
    max_err = check_additivity(df, n_rows=10)
    print(f"Additivity check (10 rows x 3 classes): max |error| = {max_err:.2e}")
    if max_err > 1e-5:
        print("ERROR: SHAP additivity error is not tiny! Stopping.")
        sys.exit(1)
    print("Additivity check passed ✓\n")

    # 2. Pick one real row of each class (Low, Medium, High)
    # Note: these rows were part of the 68-row training set used for the final model.
    print("NOTE: The 3 sample rows below come from data/processed/clean.csv and were")
    print("part of the 68 rows used to fit the final Random Forest pipeline.\n")

    for target_cls in TARGET["categories"]:
        row_series = df[df[TARGET["name"]] == target_cls].iloc[0]
        answers = {f: row_series[f] for f in FEATURE_NAMES}
        result = explain_one(answers)
        print(f"--- True class: {target_cls} ---")
        print(f"  Predicted : {result['predicted_level']}  (stress_index = {result['stress_index']})")
        print(f"  Probs     : {result['probabilities']}")
        print("  Top 5 factors (stress_push = SHAP[High] - SHAP[Low]):")
        for rank, fac in enumerate(result["top_factors"], 1):
            print(
                f"    {rank}. {fac['label']} = '{fac['value']}' "
                f"-> {fac['direction']} stress (impact={fac['impact']:+.4f})"
            )
        print()


if __name__ == "__main__":
    main()
