"""Model tests: models use only lagged predictors and never see the test targets."""

import re

import numpy as np
import pandas as pd
import pytest

import features
from features import TARGET
from models import ALL_MODELS, common_sample, fit_predict


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    path = features.CLEAN_DIR / "features.csv"
    if not path.exists():
        features.main()
    table = pd.read_csv(path)
    return table.dropna(subset=[TARGET] + sorted({f for _, fs, _ in ALL_MODELS.values()
                                                  for f in fs})).reset_index(drop=True)


def test_all_model_predictors_are_lagged():
    for name, (_, model_features, _) in ALL_MODELS.items():
        for feature in model_features:
            assert re.search(r"_lag[1-9]$", feature), (name, feature)


@pytest.mark.parametrize("name", list(ALL_MODELS))
def test_forecasts_do_not_depend_on_test_targets(name, data):
    """Scrambling the actual inflation of the test year must not change forecasts."""
    test_year = 2015
    train = data[data["year"] < test_year]
    test = data[data["year"] == test_year]
    forecasts, _ = fit_predict(name, train, test)

    scrambled = test.copy()
    scrambled[TARGET] = scrambled[TARGET] * -7 + 1234
    forecasts_scrambled, _ = fit_predict(name, train, scrambled)
    np.testing.assert_array_equal(forecasts, forecasts_scrambled)
    assert np.isfinite(forecasts).all()


def test_common_sample_has_no_missing_main_predictors(data):
    sample = common_sample(pd.read_csv(features.CLEAN_DIR / "features.csv"))
    _, m3_features, _ = ALL_MODELS["M3_extended"]
    assert sample[[TARGET] + m3_features].notna().all().all()
