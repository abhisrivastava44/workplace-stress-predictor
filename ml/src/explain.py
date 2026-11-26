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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

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
FIG_DIR = ARTIFACTS_DIR / "figures"
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


def compute_global_shap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean |stress_push| across all rows for each of the 27 original features.
    """
    pipeline, explainer, col_to_orig, classes = _load_resources()
    X_trans = pipeline.named_steps["prep"].transform(df[FEATURE_NAMES])
    shap_arr = _normalize_shap_array(explainer.shap_values(X_trans))

    low_idx = classes.index("Low")
    high_idx = classes.index("High")

    # Aggregate transformed columns back to original features for every row
    push_matrix = np.zeros((len(df), len(FEATURE_NAMES)), dtype=float)
    feat_to_idx = {f: i for i, f in enumerate(FEATURE_NAMES)}
    for col_idx, orig_feat in enumerate(col_to_orig):
        f_idx = feat_to_idx[orig_feat]
        push_matrix[:, f_idx] += shap_arr[:, col_idx, high_idx] - shap_arr[:, col_idx, low_idx]

    mean_abs = np.mean(np.abs(push_matrix), axis=0)
    out = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "label": [FEATURE_MAP[f]["display_label"] for f in FEATURE_NAMES],
            "mean_abs_stress_push": mean_abs,
        }
    ).sort_values("mean_abs_stress_push", ascending=False)
    return out.reset_index(drop=True)


def save_shap_figures(df: pd.DataFrame, global_df: pd.DataFrame, high_result: Dict[str, Any]) -> None:
    """Save 11_shap_global.png (top 12) and 12_shap_local_example.png (High example)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # ── 1. Global SHAP bar chart (top 12) ──────────────────────────────
    top12 = global_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.barh(
        top12["label"],
        top12["mean_abs_stress_push"],
        color="#4F46E5",
        edgecolor="#1F2937",
        linewidth=0.5,
    )
    for bar, val in zip(bars, top12["mean_abs_stress_push"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8.5)

    ax.set_xlabel("Mean |stress_push| = Mean |SHAP(High) - SHAP(Low)|")
    ax.set_title("Global Feature Importance via SHAP (Top 12, n = 68)", fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, top12["mean_abs_stress_push"].max() * 1.18)
    fig.text(
        0.5,
        0.01,
        "Caption: Top 12 drivers by mean |stress_push| across 68 rows; ranking is indicative on this small sample.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIG_DIR / "11_shap_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Local SHAP bar chart for the High example ───────────────────
    factors = high_result["top_factors"][::-1]
    y_labels = [f"{f['label']} = '{f['value']}'" for f in factors]
    impacts = [f["impact"] for f in factors]
    colors = ["#EF4444" if imp >= 0 else "#22C55E" for imp in impacts]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(y_labels, impacts, color=colors, edgecolor="#1F2937", linewidth=0.5)
    ax.axvline(0, color="#1F2937", linewidth=0.8)
    for bar, imp in zip(bars, impacts):
        offset = 0.003 if imp >= 0 else -0.003
        ha = "left" if imp >= 0 else "right"
        ax.text(imp + offset, bar.get_y() + bar.get_height() / 2, f"{imp:+.3f}", va="center", ha=ha, fontsize=9)

    ax.set_xlabel("stress_push = SHAP(High) - SHAP(Low)")
    ax.set_title(
        f"Local SHAP Explanation — High Stress Example (stress_index = {high_result['stress_index']})",
        fontsize=12,
        fontweight="bold",
    )
    max_abs = max(abs(x) for x in impacts)
    ax.set_xlim(-max_abs * 0.25 if min(impacts) >= 0 else -max_abs * 1.25, max_abs * 1.22)
    fig.text(
        0.5,
        0.01,
        "Caption: Red bars push the estimate toward higher stress; all top 5 factors for this High-stress row push upward.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_DIR / "12_shap_local_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_importance(df: pd.DataFrame, global_df: pd.DataFrame) -> None:
    """
    Sanity check: compare global SHAP top 10 with sklearn permutation importance
    (on the full pipeline) and print side-by-side.
    """
    pipeline, _, _, classes = _load_resources()
    label_to_id = {c: i for i, c in enumerate(classes)}
    X = df[FEATURE_NAMES]
    y = df[TARGET["name"]].map(label_to_id)

    # In-sample on the same 68 rows the final model was fitted on — rough sanity check only.
    perm = permutation_importance(
        pipeline,
        X,
        y,
        scoring="f1_macro",
        n_repeats=30,
        random_state=42,
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    ).sort_values("perm_mean", ascending=False).reset_index(drop=True)

    print("=== SANITY CHECK: Global SHAP Top 10 vs Permutation Importance Top 10 ===")
    print(f"{'Rank':<4} | {'SHAP (mean |stress_push|)':<36} | {'Permutation Importance (macro-F1 drop)'}")
    print("-" * 85)
    for i in range(10):
        s_row = global_df.iloc[i]
        p_row = perm_df.iloc[i]
        s_txt = f"{s_row['feature']} ({s_row['mean_abs_stress_push']:.4f})"
        p_txt = f"{p_row['feature']} ({p_row['perm_mean']:.4f} +/- {p_row['perm_std']:.4f})"
        print(f"{i+1:<4} | {s_txt:<36} | {p_txt}")
    print()


def main() -> None:
    """Run SHAP verification, global/local plots, and permutation sanity check."""
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False)

    print("=" * 65)
    print("PHASE 3 — SHAP EXPLAINABILITY VERIFICATION & PLOTS")
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

    high_result: Dict[str, Any] = {}
    for target_cls in TARGET["categories"]:
        row_series = df[df[TARGET["name"]] == target_cls].iloc[0]
        answers = {f: row_series[f] for f in FEATURE_NAMES}
        result = explain_one(answers)
        if target_cls == "High":
            high_result = result
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

    # 3. Global SHAP + plots
    global_df = compute_global_shap(df)
    save_shap_figures(df, global_df, high_result)
    print(f"Saved figures to {FIG_DIR / '11_shap_global.png'} and {FIG_DIR / '12_shap_local_example.png'}\n")

    # 4. Sanity check vs permutation importance
    compare_importance(df, global_df)
    print("NOTE: With only 68 rows, SHAP and permutation rankings are indicative, not precise.")
    print("Phase 3 complete ✓")


if __name__ == "__main__":
    main()
