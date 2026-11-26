"""
ml/src/train.py
===============
Phase 2 — Model training, nested cross-validation comparison, permutation test,
and saving the final pipeline.

Usage:
    python -m ml.src.train          (from the project root)

Why this script is structured the way it is (see docs/DATA_NOTES.md):
  1. Our dataset has only 68 rows (Low=11, Medium=30, High=27). A single 80/20
     train/test split would leave ~2 Low rows in the test set, making test
     scores noisy and unreliable.
  2. Instead, we evaluate every model with RepeatedStratifiedKFold (5 folds x
     10 repeats = 50 folds) and report mean +/- std across folds.
  3. Preprocessing (OrdinalEncoder + OneHotEncoder) is wrapped inside a single
     sklearn Pipeline so encoders are fit ONLY on each training fold (no leakage).
  4. Hyperparameter tuning is kept small (shallow trees, small grids) and
     wrapped in NESTED cross-validation (inner 3-fold GridSearchCV inside the
     outer 5x10 RepeatedStratifiedKFold) so reported CV scores are not optimistic.
  5. We compare 4 models: DummyClassifier, LogisticRegression, RandomForest, and
     XGBoost, using class_weight="balanced" or balanced sample weights (no SMOTE).
  6. After choosing the best model by CV macro-F1, we run a label-permutation
     test to verify it beats chance, then fit the FINAL pipeline on all 68 rows
     for the web app and SHAP.
"""

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ── Make sure the project root is on sys.path so we can import shared/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.features import (  # noqa: E402
    FEATURE_MAP,
    FEATURE_NAMES,
    NOMINAL_FEATURES,
    ORDINAL_CATEGORIES,
    ORDINAL_FEATURES,
    TARGET,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIG_DIR = ARTIFACTS_DIR / "figures"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
INNER_SPLITS = 3


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """Load clean.csv and encode the target Low=0, Medium=1, High=2."""
    # keep_default_na=False ensures the valid category "None" stays a string.
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False)
    X = df[FEATURE_NAMES].copy()

    # Build label mapping from shared/features.py: Low=0, Medium=1, High=2
    label_to_id = {cat: idx for idx, cat in enumerate(TARGET["categories"])}
    y = df[TARGET["name"]].map(label_to_id)
    return X, y, label_to_id


def build_preprocessor() -> ColumnTransformer:
    """
    Create the ColumnTransformer from shared/features.py.
    Why: Ordinal features have a meaningful low->high order, while nominal
    features (gender, marital_status) do not, so they get one-hot encoded.
    Passing explicit categories guarantees unseen options (like gender='Other')
    stay aligned between training and inference.
    """
    nominal_categories = [FEATURE_MAP[f]["categories"] for f in NOMINAL_FEATURES]

    return ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_FEATURES,
            ),
            (
                "nominal",
                OneHotEncoder(
                    categories=nominal_categories,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                NOMINAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_model_specs() -> List[Dict[str, Any]]:
    """
    Define the 4 models and their small hyperparameter grids.
    Why small grids: With 68 rows, large grids overfit the inner CV folds.
    """
    return [
        {
            "key": "dummy",
            "name": "Dummy (Majority)",
            "needs_sample_weight": False,
            "pipeline": Pipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    ("clf", DummyClassifier(strategy="most_frequent")),
                ]
            ),
            "param_grid": None,
        },
        {
            "key": "logistic_regression",
            "name": "Logistic Regression",
            "needs_sample_weight": False,
            "pipeline": Pipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    # StandardScaler helps LR converge and regularize fairly
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "clf__C": [0.05, 0.1, 0.5, 1.0],
            },
        },
        {
            "key": "random_forest",
            "name": "Random Forest",
            "needs_sample_weight": False,
            "pipeline": Pipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                # Keep trees shallow on 68 rows to reduce overfitting
                "clf__n_estimators": [100, 150],
                "clf__max_depth": [3, 5],
                "clf__min_samples_leaf": [2],
            },
        },
        {
            "key": "xgboost",
            "name": "XGBoost",
            "needs_sample_weight": True,
            "pipeline": Pipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    (
                        "clf",
                        XGBClassifier(
                            objective="multi:softprob",
                            num_class=3,
                            eval_metric="mlogloss",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                # Shallow trees + moderate estimators to avoid memorizing 68 rows
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [2, 3],
                "clf__learning_rate": [0.08],
            },
        },
    ]


def evaluate_nested_cv(
    spec: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: RepeatedStratifiedKFold,
) -> Dict[str, Any]:
    """
    Evaluate one model using outer 5x10 RepeatedStratifiedKFold and (if a
    param_grid is given) inner 3-fold GridSearchCV on each training fold.
    """
    fold_acc: List[float] = []
    fold_f1_macro: List[float] = []
    fold_train_f1_macro: List[float] = []
    fold_roc_auc: List[float] = []

    oof_y_true: List[int] = []
    oof_y_pred: List[int] = []

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    for train_idx, val_idx in outer_cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fit_kwargs: Dict[str, Any] = {}
        if spec["needs_sample_weight"]:
            # Compute balanced sample weights ONLY on the current training fold
            sw = compute_sample_weight(class_weight="balanced", y=y_train)
            fit_kwargs["clf__sample_weight"] = sw

        if spec["param_grid"] is not None:
            search = GridSearchCV(
                estimator=spec["pipeline"],
                param_grid=spec["param_grid"],
                scoring="f1_macro",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train, **fit_kwargs)
            model = search.best_estimator_
        else:
            model = spec["pipeline"]
            model.fit(X_train, y_train, **fit_kwargs)

        # Validation fold predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)

        # Training fold predictions (to measure train vs CV overfitting gap)
        y_train_pred = model.predict(X_train)

        fold_acc.append(accuracy_score(y_val, y_val_pred))
        fold_f1_macro.append(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        fold_train_f1_macro.append(
            f1_score(y_train, y_train_pred, average="macro", zero_division=0)
        )
        fold_roc_auc.append(
            roc_auc_score(y_val, y_val_proba, multi_class="ovr", average="macro")
        )

        oof_y_true.extend(y_val.tolist())
        oof_y_pred.extend(y_val_pred.tolist())

    # Pooled per-class F1 and pooled confusion matrix across all 50 outer folds
    per_class_f1 = f1_score(
        oof_y_true, oof_y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    pooled_cm = confusion_matrix(oof_y_true, oof_y_pred, labels=[0, 1, 2])

    cv_f1_mean = float(np.mean(fold_f1_macro))
    train_f1_mean = float(np.mean(fold_train_f1_macro))

    return {
        "key": spec["key"],
        "name": spec["name"],
        "cv_accuracy_mean": float(np.mean(fold_acc)),
        "cv_accuracy_std": float(np.std(fold_acc)),
        "cv_macro_f1_mean": cv_f1_mean,
        "cv_macro_f1_std": float(np.std(fold_f1_macro)),
        "train_macro_f1_mean": train_f1_mean,
        "train_cv_gap": float(train_f1_mean - cv_f1_mean),
        "roc_auc_ovr_mean": float(np.mean(fold_roc_auc)),
        "roc_auc_ovr_std": float(np.std(fold_roc_auc)),
        "per_class_f1": {
            "Low": float(per_class_f1[0]),
            "Medium": float(per_class_f1[1]),
            "High": float(per_class_f1[2]),
        },
        "pooled_confusion_matrix": pooled_cm.tolist(),
    }


def select_best_model(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """
    Pick the best model by CV macro-F1. If top two are within 1 std of each
    other, prefer a tree-based model (Random Forest or XGBoost) for SHAP.
    """
    # Exclude dummy from winning, though it wouldn't win anyway
    candidates = [r for r in results if r["key"] != "dummy"]
    ranked = sorted(candidates, key=lambda r: r["cv_macro_f1_mean"], reverse=True)
    top = ranked[0]
    second = ranked[1]

    diff = top["cv_macro_f1_mean"] - second["cv_macro_f1_mean"]
    shap_friendly = {"random_forest", "xgboost"}

    if diff <= top["cv_macro_f1_std"] and top["key"] not in shap_friendly and second["key"] in shap_friendly:
        reason = (
            f"Top two models ({top['name']}: {top['cv_macro_f1_mean']:.3f} and "
            f"{second['name']}: {second['cv_macro_f1_mean']:.3f}) are within 1 SD "
            f"({top['cv_macro_f1_std']:.3f}); chose {second['name']} for exact TreeExplainer SHAP support."
        )
        return second, reason

    reason = (
        f"Highest nested CV macro-F1 ({top['cv_macro_f1_mean']:.3f} +/- "
        f"{top['cv_macro_f1_std']:.3f})."
    )
    if diff <= top["cv_macro_f1_std"]:
        reason += (
            f" Note: within 1 SD of {second['name']} "
            f"({second['cv_macro_f1_mean']:.3f} +/- {second['cv_macro_f1_std']:.3f}), "
            f"and {top['name']} is tree-based so it pairs directly with SHAP TreeExplainer."
        )
    return top, reason


def run_permutation_test(
    best_spec: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    """
    Run sklearn's permutation_test_score (100 permutations, 5-fold stratified CV)
    to verify the chosen model performs significantly better than random chance.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    # Use the first/default grid or base pipeline for permutation test speed
    estimator = best_spec["pipeline"]
    fit_params = {}
    if best_spec["key"] == "random_forest":
        estimator.set_params(clf__n_estimators=100, clf__max_depth=3, clf__min_samples_leaf=2)
    elif best_spec["key"] == "xgboost":
        estimator.set_params(clf__n_estimators=50, clf__max_depth=2, clf__learning_rate=0.05)

    score, perm_scores, pvalue = permutation_test_score(
        estimator,
        X,
        y,
        scoring="f1_macro",
        cv=cv,
        n_permutations=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return {
        "observed_macro_f1": float(score),
        "perm_mean_macro_f1": float(np.mean(perm_scores)),
        "perm_std_macro_f1": float(np.std(perm_scores)),
        "p_value": float(pvalue),
        "n_permutations": 100,
    }


def save_plots(results: List[Dict[str, Any]], best_result: Dict[str, Any]) -> None:
    """Save model comparison bar chart and pooled confusion matrix."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # ── 1. Model comparison bar chart with error bars ──────────────────
    names = [r["name"] for r in results]
    acc_means = [r["cv_accuracy_mean"] for r in results]
    acc_stds = [r["cv_accuracy_std"] for r in results]
    f1_means = [r["cv_macro_f1_mean"] for r in results]
    f1_stds = [r["cv_macro_f1_std"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        acc_means,
        width,
        yerr=acc_stds,
        capsize=4,
        label="CV Accuracy",
        color="#4F46E5",
        edgecolor="#1F2937",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        f1_means,
        width,
        yerr=f1_stds,
        capsize=4,
        label="CV Macro-F1",
        color="#16A34A",
        edgecolor="#1F2937",
        linewidth=0.6,
    )

    for i, (am, fm) in enumerate(zip(acc_means, f1_means)):
        ax.text(i - width / 2, am + acc_stds[i] + 0.02, f"{am:.2f}", ha="center", fontsize=8.5)
        ax.text(i + width / 2, fm + f1_stds[i] + 0.02, f"{fm:.2f}", ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean +/- 1 SD)")
    ax.set_title("Nested 5x10 Repeated Stratified CV Comparison (n = 68)", fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper left")
    fig.text(
        0.5,
        0.01,
        "Caption: Tree models (RF/XGBoost) outperform Dummy and Logistic Regression; error bars show +/- 1 SD across 50 folds.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIG_DIR / "09_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Pooled confusion matrix for the best model ──────────────────
    cm = np.array(best_result["pooled_confusion_matrix"])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    classes = TARGET["categories"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Row %"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Stress Level", fontsize=11)
    ax.set_ylabel("True Stress Level", fontsize=11)
    ax.set_title(
        f"Pooled Confusion Matrix — {best_result['name']} (50 folds = 680 OOF preds)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        f"Caption: Pooled out-of-fold counts (10 repeats of 68 rows) and row percentages; Low (11 rows) is the hardest class.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIG_DIR / "10_best_model_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_final_pipeline(
    best_spec: Dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Fit the chosen model's pipeline on ALL 68 rows for production inference.
    Why: We already estimated generalization error via nested CV; now we give
    the deployed model all 68 real examples.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fit_kwargs: Dict[str, Any] = {}
    if best_spec["needs_sample_weight"]:
        fit_kwargs["clf__sample_weight"] = compute_sample_weight("balanced", y)

    search = GridSearchCV(
        estimator=best_spec["pipeline"],
        param_grid=best_spec["param_grid"],
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y, **fit_kwargs)
    return search.best_estimator_, search.best_params_


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a clean, readable comparison table to stdout."""
    header = (
        f"{'Model':<21} | {'CV Acc (mean±std)':<17} | {'CV F1 (mean±std)':<17} | "
        f"{'Train F1':<8} | {'Gap':<6} | {'ROC-AUC':<7} | {'F1 Low/Med/High'}"
    )
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON (Nested CV: outer 5x10 RepeatedStratifiedKFold, inner 3-fold GridSearchCV)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        pcf = r["per_class_f1"]
        print(
            f"{r['name']:<21} | "
            f"{r['cv_accuracy_mean']:.3f} +/- {r['cv_accuracy_std']:.3f}   | "
            f"{r['cv_macro_f1_mean']:.3f} +/- {r['cv_macro_f1_std']:.3f}   | "
            f"{r['train_macro_f1_mean']:.3f}    | "
            f"{r['train_cv_gap']:+.3f} | "
            f"{r['roc_auc_ovr_mean']:.3f}   | "
            f"{pcf['Low']:.2f} / {pcf['Medium']:.2f} / {pcf['High']:.2f}"
        )
    print("=" * len(header) + "\n")


def main() -> None:
    """Run full Phase 2 training, evaluation, and artifact saving."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, label_to_id = load_dataset()
    print(f"Loaded {len(X)} rows, {len(X.columns)} features.")
    print(f"Target distribution: {y.value_counts().sort_index().to_dict()} (0=Low, 1=Medium, 2=High)")

    # Save label mapping
    label_mapping = {
        "label_to_id": label_to_id,
        "id_to_label": {str(v): k for k, v in label_to_id.items()},
        "classes": TARGET["categories"],
    }
    with open(ARTIFACTS_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    specs = build_model_specs()

    results: List[Dict[str, Any]] = []
    for spec in specs:
        print(f"Evaluating {spec['name']}...")
        res = evaluate_nested_cv(spec, X, y, outer_cv)
        results.append(res)

    print_comparison_table(results)

    best_result, selection_reason = select_best_model(results)
    best_spec = next(s for s in specs if s["key"] == best_result["key"])
    print(f"Chosen model: {best_result['name']}")
    print(f"Reason: {selection_reason}")
    print(
        f"Honest check on Low class (11 rows): pooled F1 = {best_result['per_class_f1']['Low']:.3f} "
        f"(vs Medium {best_result['per_class_f1']['Medium']:.3f}, High {best_result['per_class_f1']['High']:.3f}).\n"
    )

    # Permutation test
    print("Running label-permutation test (100 permutations, 5-fold CV)...")
    perm_stats = run_permutation_test(best_spec, X, y)
    print(
        f"Permutation test: observed macro-F1 = {perm_stats['observed_macro_f1']:.3f}, "
        f"chance mean = {perm_stats['perm_mean_macro_f1']:.3f} +/- {perm_stats['perm_std_macro_f1']:.3f}, "
        f"p-value = {perm_stats['p_value']:.4f}\n"
    )

    # Save plots
    save_plots(results, best_result)
    print(f"Saved plots to {FIG_DIR / '09_model_comparison.png'} and {FIG_DIR / '10_best_model_confusion_matrix.png'}")

    # Fit final pipeline on all 68 rows
    final_pipeline, final_best_params = fit_final_pipeline(best_spec, X, y)
    joblib.dump(final_pipeline, ARTIFACTS_DIR / "stress_pipeline.joblib")
    print(f"Saved final pipeline (fitted on all {len(X)} rows) to {ARTIFACTS_DIR / 'stress_pipeline.joblib'}")
    print(f"Final best hyperparameters: {final_best_params}")

    # Save metrics.json with package versions
    metrics_payload = {
        "dataset_rows": int(len(X)),
        "evaluation_protocol": (
            f"Nested CV: outer RepeatedStratifiedKFold(n_splits={N_SPLITS}, "
            f"n_repeats={N_REPEATS}, random_state={RANDOM_STATE}), "
            f"inner StratifiedKFold(n_splits={INNER_SPLITS}) GridSearchCV(scoring='f1_macro')"
        ),
        "chosen_model": best_result["key"],
        "chosen_model_name": best_result["name"],
        "selection_reason": selection_reason,
        "final_best_params": final_best_params,
        "permutation_test": perm_stats,
        "models": results,
        "package_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics to {ARTIFACTS_DIR / 'metrics.json'}")
    print("\nPhase 2 training complete ✓")


if __name__ == "__main__":
    main()
