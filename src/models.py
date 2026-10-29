"""Step 5: forecasting models.

Every model forecasts inflation in year t from predictors dated t-1 (or t-2).

  M0 Naive       inflation(t) = inflation(t-1)
  M1 Baseline    linear regression on infl_lag1 (pooled; also a version with
                 country fixed effects, M1_FE)
  M2 Commodity   M1 + food_chg_lag1 + energy_chg_lag1
  M3 Extended    M2 + fx_change_lag1 + gdp_growth_lag1
  M4 Ridge       M3 predictors, penalty alpha chosen by time-ordered
                 cross-validation inside the training years only
  M5 Boosting    HistGradientBoostingRegressor on M3 predictors (comparison only;
                 the sample is small for this kind of model)

Extra variants for the robustness checks are listed in ROBUSTNESS_MODELS.

This file only defines how to fit a model and make a forecast. The
out-of-sample loop (train on years before Y, forecast Y) is in evaluate.py.
Running this file fits the linear models once on all years and saves their
coefficients with 95% confidence intervals.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import PREDICTORS, TARGET

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
TABLES_DIR = ROOT / "results" / "tables"

SEED = 0

M1_FEATURES = ["infl_lag1"]
M2_FEATURES = M1_FEATURES + ["food_chg_lag1", "energy_chg_lag1"]
M3_FEATURES = M2_FEATURES + ["fx_change_lag1", "gdp_growth_lag1"]

# name -> (kind, predictors, country fixed effects?)
MAIN_MODELS = {
    "M0_naive": ("naive", M1_FEATURES, False),
    "M1_baseline": ("ols", M1_FEATURES, False),
    "M1_baseline_FE": ("ols", M1_FEATURES, True),
    "M2_commodity": ("ols", M2_FEATURES, False),
    "M3_extended": ("ols", M3_FEATURES, False),
    "M4_ridge": ("ridge", M3_FEATURES, False),
    "M5_boosting": ("boosting", M3_FEATURES, False),
}

ROBUSTNESS_MODELS = {
    # (b) country fixed effects on
    "M2_commodity_FE": ("ols", M2_FEATURES, True),
    "M3_extended_FE": ("ols", M3_FEATURES, True),
    # (c) two years of lags
    "M1_lag2": ("ols", ["infl_lag1", "infl_lag2"], False),
    "M2_lag2": ("ols", ["infl_lag1", "infl_lag2", "food_chg_lag1", "energy_chg_lag1",
                        "food_chg_lag2", "energy_chg_lag2"], False),
    # (e) food only vs energy only (plus fertilizers, which the Pink Sheet also has)
    "M2_food_only": ("ols", M1_FEATURES + ["food_chg_lag1"], False),
    "M2_energy_only": ("ols", M1_FEATURES + ["energy_chg_lag1"], False),
    "M2_plus_fert": ("ols", M2_FEATURES + ["fert_chg_lag1"], False),
}

ALL_MODELS = {**MAIN_MODELS, **ROBUSTNESS_MODELS}

# Every model may only use lagged predictors (checked in tests/test_models.py).
for _name, (_kind, _features, _fe) in ALL_MODELS.items():
    assert set(_features) <= set(PREDICTORS), _name

RIDGE_ALPHAS = np.logspace(-3, 4, 15)  # candidate penalties, 0.001 ... 10000
RIDGE_VALIDATION_YEARS = 3             # inner CV: forecast each of the last 3 training years


# ------------------------------------------------------------------ linear models

def design_matrix(data: pd.DataFrame, features: list, countries=None) -> pd.DataFrame:
    """Predictors plus either one constant or one intercept per country (fixed effects)."""
    X = data[features].astype(float)
    if countries is None:
        return sm.add_constant(X, has_constant="add")
    dummies = pd.get_dummies(data["country"]).reindex(columns=countries, fill_value=0)
    return pd.concat([X, dummies.astype(float)], axis=1)


def fit_ols(train: pd.DataFrame, features: list, fixed_effects: bool):
    countries = sorted(train["country"].unique()) if fixed_effects else None
    X = design_matrix(train, features, countries)
    return sm.OLS(train[TARGET].astype(float), X).fit(), countries


def predict_ols(result, countries, test: pd.DataFrame, features: list) -> np.ndarray:
    X = design_matrix(test, features, countries)
    prediction = X.to_numpy() @ result.params.to_numpy()
    if countries is not None:
        # A country with no training rows has no estimated intercept of its own;
        # it gets the average of the estimated country intercepts.
        unseen = ~test["country"].isin(countries).to_numpy()
        prediction[unseen] += result.params[countries].mean()
    return prediction


# ---------------------------------------------------------------------- ridge

def make_ridge(alpha: float):
    # Predictors are standardised (using training data only) so one penalty fits all.
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def choose_ridge_alpha(train: pd.DataFrame, features: list) -> float:
    """Time-ordered cross-validation inside the training data.

    For each of the last few training years v: fit on training years before v,
    forecast year v. Pick the alpha with the lowest mean absolute error.
    """
    years = sorted(train["year"].unique())
    validation_years = years[-RIDGE_VALIDATION_YEARS:]
    errors = {alpha: [] for alpha in RIDGE_ALPHAS}
    for v in validation_years:
        inner_train = train[train["year"] < v]
        inner_valid = train[train["year"] == v]
        for alpha in RIDGE_ALPHAS:
            model = make_ridge(alpha).fit(inner_train[features], inner_train[TARGET])
            errors[alpha].extend(np.abs(model.predict(inner_valid[features])
                                        - inner_valid[TARGET]))
    mean_errors = {alpha: np.mean(e) for alpha, e in errors.items()}
    return min(mean_errors, key=mean_errors.get)


# ------------------------------------------------------------------- dispatcher

def fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame):
    """Fit model `name` on `train` and forecast `test`.

    Returns (forecasts, info) where info holds model details worth saving
    (for example the chosen ridge alpha).
    """
    kind, features, fixed_effects = ALL_MODELS[name]
    info = {}
    if kind == "naive":
        prediction = test["infl_lag1"].to_numpy(dtype=float)
    elif kind == "ols":
        result, countries = fit_ols(train, features, fixed_effects)
        prediction = predict_ols(result, countries, test, features)
        if countries is not None:
            info["unseen_countries"] = int((~test["country"].isin(countries)).sum())
    elif kind == "ridge":
        alpha = choose_ridge_alpha(train, features)
        model = make_ridge(alpha).fit(train[features], train[TARGET])
        prediction = model.predict(test[features])
        info["alpha"] = float(alpha)
    elif kind == "boosting":
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
            early_stopping=False, random_state=SEED)
        prediction = model.fit(train[features], train[TARGET]).predict(test[features])
    else:
        raise ValueError(kind)
    return np.asarray(prediction, dtype=float), info


# ------------------------------------------------------ full-sample coefficients

def common_sample(features_table: pd.DataFrame) -> pd.DataFrame:
    """Rows where every predictor of the largest main model (M3) is available.

    All main models are trained and tested on these same rows so that their
    errors can be compared fairly.
    """
    return features_table.dropna(subset=[TARGET] + M3_FEATURES).reset_index(drop=True)


def coefficient_table(data: pd.DataFrame) -> pd.DataFrame:
    """Fit M1, M2 and M3 on all years and report coefficients with 95% CIs.

    Commodity prices are the same for every country in a given year, so
    errors within a year are not independent. Standard errors are therefore
    clustered by year (conventional ones would be too small).
    """
    rows = []
    for name in ["M1_baseline", "M2_commodity", "M3_extended",
                 "M1_baseline_FE", "M2_commodity_FE", "M3_extended_FE"]:
        _, features, fixed_effects = ALL_MODELS[name]
        result, _ = fit_ols(data, features, fixed_effects)
        result = result.get_robustcov_results(cov_type="cluster",
                                              groups=data["year"].to_numpy())
        params = pd.Series(result.params, index=result.model.exog_names)
        ses = pd.Series(result.bse, index=result.model.exog_names)
        pvalues = pd.Series(result.pvalues, index=result.model.exog_names)
        ci = pd.DataFrame(result.conf_int(0.05), index=result.model.exog_names)
        for variable in ["const"] + features:
            if variable not in params:
                continue  # fixed-effects models have country intercepts instead of const
            rows.append({"model": name, "variable": variable,
                         "coef": params[variable], "std_err": ses[variable],
                         "ci_low": ci.loc[variable, 0], "ci_high": ci.loc[variable, 1],
                         "p_value": pvalues[variable], "n_obs": int(result.nobs),
                         "n_year_clusters": data["year"].nunique(),
                         "years": f"{data['year'].min()}-{data['year'].max()}",
                         "r_squared": result.rsquared})
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    data = common_sample(pd.read_csv(CLEAN_DIR / "features.csv"))
    table = coefficient_table(data)
    table.to_csv(TABLES_DIR / "coefficients_full_sample.csv", index=False)
    print(f"  saved results/tables/coefficients_full_sample.csv "
          f"({len(data)} rows, {data['year'].min()}-{data['year'].max()})")
    return table


if __name__ == "__main__":
    main()
