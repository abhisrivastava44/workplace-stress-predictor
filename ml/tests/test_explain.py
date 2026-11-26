"""
ml/tests/test_explain.py
========================
Unit tests for ml/src/explain.py.
"""

import pandas as pd
import pytest

from ml.src.explain import CLEAN_CSV, explain_one
from shared.features import FEATURE_NAMES, TARGET


@pytest.fixture(scope="module")
def sample_answers() -> dict:
    """Return one valid answer dict from data/processed/clean.csv."""
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False)
    row = df.iloc[0]
    return {feat: row[feat] for feat in FEATURE_NAMES}


def test_explain_one_valid_input(sample_answers: dict) -> None:
    """Valid input returns expected keys and valid probability/stress_index ranges."""
    result = explain_one(sample_answers)

    assert result["predicted_level"] in TARGET["categories"]
    assert set(result["probabilities"].keys()) == set(TARGET["categories"])
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-2
    assert 0.0 <= result["stress_index"] <= 100.0

    for fac in result["top_factors"]:
        assert fac["feature"] in FEATURE_NAMES
        assert isinstance(fac["label"], str) and len(fac["label"]) > 0
        assert fac["value"] == sample_answers[fac["feature"]]
        assert isinstance(fac["impact"], float)


def test_top_factors_has_five_items(sample_answers: dict) -> None:
    """explain_one returns exactly 5 top_factors."""
    result = explain_one(sample_answers)
    assert len(result["top_factors"]) == 5


def test_top_factors_directions_valid(sample_answers: dict) -> None:
    """Every factor direction is strictly 'increases' or 'reduces'."""
    result = explain_one(sample_answers)
    for fac in result["top_factors"]:
        assert fac["direction"] in {"increases", "reduces"}


def test_explain_one_invalid_value(sample_answers: dict) -> None:
    """An invalid category value raises ValueError."""
    bad = dict(sample_answers)
    bad["sleep_duration"] = "20 hours"
    with pytest.raises(ValueError, match="Invalid value"):
        explain_one(bad)


def test_explain_one_missing_field(sample_answers: dict) -> None:
    """A missing required feature raises ValueError."""
    bad = dict(sample_answers)
    del bad["work_life_balance"]
    with pytest.raises(ValueError, match="Missing required"):
        explain_one(bad)
