"""Leakage tests: no predictor may use data from the year being forecast.

These tests use the real clean panel (built from the downloaded raw files).
"""

import re

import numpy as np
import pandas as pd
import pytest

import clean
import features
from features import PREDICTORS, TARGET, make_features


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    path = clean.CLEAN_DIR / "panel.csv"
    if not path.exists():
        clean.main()
    return pd.read_csv(path)


def test_predictor_names_are_all_lags():
    for name in PREDICTORS:
        assert re.search(r"_lag[1-9]$", name), f"{name} is not a lagged variable"
    assert TARGET not in PREDICTORS


def test_changing_year_t_data_does_not_change_year_t_predictors(panel):
    """For every year T: change all year-T data, rebuild features, and check that
    the predictors for year T are unchanged (they may only use T-1 or earlier)."""
    original = make_features(panel)
    value_columns = [c for c in panel.columns
                     if c not in ("country", "country_name", "year", "high_inflation")]
    for year in sorted(panel["year"].unique()):
        changed_panel = panel.copy()
        rows = changed_panel["year"] == year
        changed_panel.loc[rows, value_columns] += 1000.0
        changed = make_features(changed_panel)

        same_year = original["year"] == year
        pd.testing.assert_frame_equal(original.loc[same_year, PREDICTORS],
                                      changed.loc[same_year, PREDICTORS])

        # Sanity check that the test can detect a change: year T+1 lags must move.
        next_year = original["year"] == year + 1
        if next_year.any():
            moved = (original.loc[next_year, "infl_lag1"]
                     != changed.loc[next_year, "infl_lag1"]).sum()
            assert moved > 0


def test_lag_values_come_from_the_previous_calendar_year(panel):
    table = make_features(panel)
    lookup = panel.set_index(["country", "year"])
    for _, row in table.sample(200, random_state=0).iterrows():
        key = (row["country"], row["year"] - 1)
        expected = lookup["inflation"].get(key, np.nan)
        if np.isnan(expected):
            assert np.isnan(row["infl_lag1"])
        else:
            assert row["infl_lag1"] == expected
            assert row["food_chg_lag1"] == lookup.loc[key, "food_chg"] or (
                np.isnan(row["food_chg_lag1"]) and np.isnan(lookup.loc[key, "food_chg"]))


def test_saved_features_file_matches_code(panel):
    path = features.CLEAN_DIR / "features.csv"
    if not path.exists():
        features.main()
    saved = pd.read_csv(path)
    rebuilt = make_features(panel)
    pd.testing.assert_frame_equal(saved[PREDICTORS], rebuilt[PREDICTORS], check_dtype=False)
