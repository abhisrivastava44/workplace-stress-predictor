"""
ml/src/load_data.py
====================
Phase 0 — Read the raw Google Forms CSV, clean it, validate it, and save
a tidy version to data/processed/clean.csv.

Usage:
    python -m ml.src.load_data          (from the project root)

What this script does, step by step:
  1. Read CSV with keep_default_na=False so that the string "None" (a real
     answer for childcare/elderly-care) is NOT turned into NaN by pandas.
  2. Strip leading/trailing whitespace and collapse double spaces in headers.
  3. Drop the Google Forms "Timestamp" column (not a feature).
  4. Rename columns from the full question text to short snake_case names
     using the mapping in shared/features.py.
  5. Normalize en-dash characters (–) to hyphens (-) in every cell, because
     Google Forms sometimes uses en dashes in ranges like "22–28".
  6. Strip whitespace from every string cell.
  7. Validate every value in every column against the allowed lists defined
     in shared/features.py.  If ANY value is invalid, print the offending
     rows and EXIT — do not silently fix or drop.
  8. Print a full audit: row count, class balance, missing values, duplicates.
  9. Save the clean DataFrame to data/processed/clean.csv.
"""

import sys
from pathlib import Path

import pandas as pd

# ── Make sure the project root is on sys.path so we can import shared/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.features import (  # noqa: E402
    ALLOWED_VALUES,
    COLUMN_RENAME_MAP,
    FEATURE_NAMES,
    TARGET,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "survey_responses.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "clean.csv"


def load_and_clean() -> pd.DataFrame:
    """Read, clean, rename, validate, and return the survey DataFrame."""

    # ── 0.  Check the file exists ──────────────────────────────────────
    if not RAW_CSV.exists():
        print(f"ERROR: Raw CSV not found at {RAW_CSV}")
        print("Place your Google Forms export there and re-run.")
        sys.exit(1)

    # ── 1.  Read CSV ───────────────────────────────────────────────────
    # keep_default_na=False → the string "None" stays as "None", not NaN.
    df = pd.read_csv(RAW_CSV, keep_default_na=False)
    print(f"Raw CSV loaded: {len(df)} rows, {len(df.columns)} columns\n")

    # ── 2.  Show original headers (useful for debugging) ───────────────
    print("Original column headers:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. [{col}]")
    print()

    # ── 3.  Strip whitespace and collapse double spaces in headers ─────
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    # ── 4.  Drop Timestamp column ──────────────────────────────────────
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
        print("Dropped 'Timestamp' column.\n")
    else:
        print("No 'Timestamp' column found (OK if already removed).\n")

    # ── 5.  Rename columns to snake_case ───────────────────────────────
    # Check that every expected raw header is present before renaming.
    missing_headers = [h for h in COLUMN_RENAME_MAP if h not in df.columns]
    if missing_headers:
        print("ERROR: These expected column headers were NOT found in the CSV:")
        for h in missing_headers:
            print(f"  - [{h}]")
        print("\nActual headers after stripping:")
        for col in df.columns:
            print(f"  - [{col}]")
        sys.exit(1)

    df = df.rename(columns=COLUMN_RENAME_MAP)
    print("Columns renamed to snake_case.\n")

    # ── 6.  Normalize en-dashes → hyphens & strip whitespace in string cells
    # Google Forms sometimes uses the Unicode en dash (–) in ranges.
    # Use is_string_dtype so this works in both pandas 2.x (object) and 3.x (str).
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.replace("\u2013", "-", regex=False).str.strip()

    print("Normalized en-dashes to hyphens and stripped cell whitespace.\n")

    # ── 8.  Validate values ────────────────────────────────────────────
    # Every column (features + target) must contain only allowed values.
    all_columns = FEATURE_NAMES + [TARGET["name"]]
    validation_failed = False

    for col in all_columns:
        allowed = ALLOWED_VALUES[col]
        invalid_mask = ~df[col].isin(allowed)
        if invalid_mask.any():
            validation_failed = True
            bad_rows = df.loc[invalid_mask, [col]]
            print(f"VALIDATION ERROR in column '{col}':")
            print(f"  Allowed values: {allowed}")
            print(f"  Invalid rows ({invalid_mask.sum()}):")
            print(bad_rows.to_string(index=True))
            print()

    if validation_failed:
        print("STOPPING: Fix the invalid values above and re-run.")
        sys.exit(1)

    print("All values validated against shared/features.py ✓\n")

    # ── 9.  Reorder columns: features first, then target ───────────────
    df = df[all_columns]

    return df


def audit(df: pd.DataFrame) -> None:
    """Print a summary of the cleaned data: size, balance, issues."""

    print("=" * 60)
    print("DATA AUDIT SUMMARY")
    print("=" * 60)

    # Row count
    print(f"\nTotal rows:    {len(df)}")
    print(f"Total columns: {len(df.columns)}  (27 features + 1 target)")

    # Class balance of the target
    target_col = TARGET["name"]
    balance = df[target_col].value_counts()
    print(f"\nClass balance ({target_col}):")
    for cls in TARGET["categories"]:
        count = balance.get(cls, 0)
        pct = count / len(df) * 100
        print(f"  {cls:8s}: {count:3d}  ({pct:5.1f}%)")

    # Warn if any class is below 20% share
    for cls in TARGET["categories"]:
        count = balance.get(cls, 0)
        pct = count / len(df) * 100
        if pct < 20:
            print(f"\n  ⚠ WARNING: '{cls}' is only {pct:.1f}% of the data "
                  f"({count} rows). This is an under-represented class.")

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nMissing values: {total_missing}")
    if total_missing > 0:
        print(missing[missing > 0].to_string())

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}")
    if dup_count > 0:
        print("  (Keeping all — they may be legitimate identical responses.)")

    # Dataset size warning
    if len(df) < 150:
        print(f"\n{'='*60}")
        print("⚠ SMALL DATASET WARNING")
        print(f"{'='*60}")
        print(f"Only {len(df)} rows. This is too few for a reliable single")
        print("train/test split.  Per DATA_NOTES.md, we will use:")
        print("  • RepeatedStratifiedKFold (5 folds × 10 repeats)")
        print("  • Final model fitted on ALL rows for the app")
        print("  • Baselines: DummyClassifier + Logistic Regression")
        print("  • class_weight='balanced' — no SMOTE")
        print("The team may collect more responses later; the pipeline")
        print("is designed to re-run on a larger CSV with no code changes.")

    print(f"\n{'='*60}")
    print()


def main() -> None:
    """Entry point: load, audit, save."""

    df = load_and_clean()
    audit(df)

    # Save cleaned data
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Clean CSV saved to: {CLEAN_CSV}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nPhase 0 data loading complete ✓")


if __name__ == "__main__":
    main()
