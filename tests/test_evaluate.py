"""Evaluation tests: the expanding window never trains on the test year or later."""

import numpy as np
import pandas as pd
import pytest

import features
from evaluate import FIRST_TEST_YEAR, compare, expanding_window_splits, run_expanding_window
from models import common_sample


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    path = features.CLEAN_DIR / "features.csv"
    if not path.exists():
        features.main()
    return common_sample(pd.read_csv(path))


def test_training_years_are_always_before_the_test_year(data):
    test_years = []
    for year, train, test in expanding_window_splits(data):
        assert train["year"].max() < year
        assert (test["year"] == year).all()
        test_years.append(year)
    assert test_years[0] == FIRST_TEST_YEAR
    assert test_years == sorted(test_years)  # never shuffled
    assert test_years[-1] == data["year"].max()


def test_comparing_a_model_with_itself_gives_no_difference(data):
    predictions, _ = run_expanding_window(data[data["year"] <= 2014], ["M1_baseline"])
    copy = predictions.assign(model="M1_copy")
    result = compare(pd.concat([predictions, copy]), "M1_copy", "M1_baseline")
    assert result["MAE_diff_a_minus_b"] == 0
    assert np.isnan(result["DM_p_abs"])


def test_duplicate_forecasts_are_rejected(data):
    predictions, _ = run_expanding_window(data[data["year"] <= 2013], ["M0_naive", "M1_baseline"])
    with pytest.raises(ValueError):
        compare(pd.concat([predictions, predictions]), "M0_naive", "M1_baseline")
