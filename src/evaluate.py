"""Step 6: out-of-sample evaluation with an expanding window.

For every test year Y (2012 to the last year with data):
  train every model on all years before Y, then forecast year Y.
The data are never shuffled, so a model never sees the future.

Outputs (results/tables/):
  oos_predictions.csv          every forecast of every main model
  oos_metrics_overall.csv      MAE, RMSE, median absolute error per model
  oos_metrics_by_country.csv   the same, per country
  oos_metrics_by_year.csv      the same, per year
  m2_vs_m1_by_country.csv      M2 minus M1 error, per country
  comparisons_vs_m1.csv        every main model vs the M1 baseline, with tests
  robustness_checks.csv        the M2-vs-M1 comparison under other choices
  model_details.csv            chosen ridge alphas and fixed-effects notes
  m2_coefficients_by_test_year.csv  M2 coefficients estimated in each training window
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from features import TARGET
from models import ALL_MODELS, M2_FEATURES, MAIN_MODELS, common_sample, fit_ols, fit_predict

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
TABLES_DIR = ROOT / "results" / "tables"

FIRST_TEST_YEAR = 2012
EXCLUDED_SHOCK_YEARS = [2020, 2021, 2022]


# ------------------------------------------------------------ expanding window

def expanding_window_splits(data: pd.DataFrame, first_test_year: int = FIRST_TEST_YEAR):
    """Yield (test_year, train, test): train = all years before the test year."""
    for year in range(first_test_year, int(data["year"].max()) + 1):
        train = data[data["year"] < year]
        test = data[data["year"] == year]
        if len(test) and len(train):
            yield year, train, test


def run_expanding_window(data: pd.DataFrame, model_names: list):
    """Forecast every test year with every model. Returns (predictions, details)."""
    predictions, details = [], []
    for year, train, test in expanding_window_splits(data):
        for name in model_names:
            forecast, info = fit_predict(name, train, test)
            predictions.append(pd.DataFrame({
                "model": name, "country": test["country"].to_numpy(), "year": year,
                "actual": test[TARGET].to_numpy(), "forecast": forecast,
                "train_rows": len(train), "train_years": f"{train['year'].min()}-{year - 1}"}))
            for key, value in info.items():
                details.append({"model": name, "test_year": year, "detail": key,
                                "value": value})
    predictions = pd.concat(predictions, ignore_index=True)
    predictions["error"] = predictions["forecast"] - predictions["actual"]
    predictions["abs_error"] = predictions["error"].abs()
    return predictions, pd.DataFrame(details)


# --------------------------------------------------------------------- metrics

def metrics(errors: pd.Series) -> pd.Series:
    return pd.Series({"n": len(errors),
                      "MAE": errors.abs().mean(),
                      "RMSE": np.sqrt((errors ** 2).mean()),
                      "MedAE": errors.abs().median()})


def metrics_table(predictions: pd.DataFrame, by: list) -> pd.DataFrame:
    table = predictions.groupby(by)["error"].apply(metrics).unstack().reset_index()
    table["n"] = table["n"].astype(int)
    return table


def diebold_mariano(predictions: pd.DataFrame, model_a: str, model_b: str,
                    loss: str = "abs") -> tuple:
    """Diebold-Mariano test of equal forecast accuracy, model A vs model B.

    All countries share the same commodity shock in a given year, so their
    forecast errors are not independent. We therefore average the loss
    difference over countries within each year and test whether the mean of
    these yearly differences is zero. One-year-ahead forecasts (h = 1), with
    the Harvey-Leybourne-Newbold small-sample correction and a t distribution.

    Returns (mean yearly loss difference A - B, DM statistic, two-sided p-value,
    number of years). A negative difference means A had the smaller loss.
    """
    a = predictions[predictions["model"] == model_a].set_index(["country", "year"])
    b = predictions[predictions["model"] == model_b].set_index(["country", "year"])
    a, b = a.align(b, join="inner", axis=0)
    if loss == "abs":
        d = a["abs_error"] - b["abs_error"]
    else:
        d = a["error"] ** 2 - b["error"] ** 2
    yearly = d.groupby(level="year").mean()
    T = len(yearly)
    mean = yearly.mean()
    variance = ((yearly - mean) ** 2).mean()  # h = 1: no autocorrelation terms
    if variance == 0:  # identical forecasts: the test is undefined
        return mean, np.nan, np.nan, T
    dm = mean / np.sqrt(variance / T)
    dm *= np.sqrt((T - 1) / T)  # Harvey-Leybourne-Newbold correction for h = 1
    p_value = 2 * stats.t.sf(abs(dm), df=T - 1)
    return mean, dm, p_value, T


def compare(predictions: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Compare two models on exactly the same country-years."""
    pair = predictions[predictions["model"].isin([model_a, model_b])]
    if pair.duplicated(["model", "country", "year"]).any():
        raise ValueError(f"Duplicate forecasts for {model_a} or {model_b}")
    counts = pair.groupby(["country", "year"])["model"].nunique()
    both = counts[counts == 2].index
    pair = pair.set_index(["country", "year"]).loc[both].reset_index()

    by_country = pair.pivot_table(index="country", columns="model", values="abs_error",
                                  aggfunc="mean")
    a_better = int((by_country[model_a] < by_country[model_b]).sum())
    n_countries = len(by_country)
    # Sign test: is the number of countries where A wins different from 50/50?
    sign_p = stats.binomtest(a_better, n_countries, 0.5).pvalue

    mae = pair.groupby("model")["abs_error"].mean()
    rmse = np.sqrt(pair.groupby("model")["error"].apply(lambda e: (e ** 2).mean()))
    _, dm_abs, p_abs, n_years = diebold_mariano(pair, model_a, model_b, "abs")
    _, dm_sq, p_sq, _ = diebold_mariano(pair, model_a, model_b, "squared")
    return {
        "model_a": model_a, "model_b": model_b,
        "n_forecasts": len(pair) // 2, "n_countries": n_countries, "n_years": n_years,
        "MAE_a": mae[model_a], "MAE_b": mae[model_b], "MAE_diff_a_minus_b": mae[model_a] - mae[model_b],
        "MAE_pct_change": 100 * (mae[model_a] / mae[model_b] - 1),
        "RMSE_a": rmse[model_a], "RMSE_b": rmse[model_b],
        "RMSE_diff_a_minus_b": rmse[model_a] - rmse[model_b],
        "countries_a_better": a_better,
        "share_countries_a_better": a_better / n_countries,
        "sign_test_p": sign_p,
        "DM_stat_abs": dm_abs, "DM_p_abs": p_abs,
        "DM_stat_squared": dm_sq, "DM_p_squared": p_sq,
    }


def m2_coefficients_by_test_year(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """The M2 coefficients that were actually used to forecast each test year.

    Shows how the estimated effect of commodity prices changes as more years
    enter the training data, next to the average actual and forecast inflation.
    """
    rows = []
    for year, train, test in expanding_window_splits(data):
        result, _ = fit_ols(train, M2_FEATURES, fixed_effects=False)
        year_predictions = predictions[predictions["year"] == year]
        mean_forecast = year_predictions.groupby("model")["forecast"].mean()
        rows.append({
            "test_year": year, "train_years": f"{train['year'].min()}-{year - 1}",
            **{f"coef_{name}": value for name, value in result.params.items()},
            "food_chg_lag1_value": test["food_chg_lag1"].iloc[0],
            "energy_chg_lag1_value": test["energy_chg_lag1"].iloc[0],
            "mean_actual": test[TARGET].mean(),
            "mean_forecast_M1": mean_forecast["M1_baseline"],
            "mean_forecast_M2": mean_forecast["M2_commodity"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- robustness

def robustness_checks(features_table: pd.DataFrame, main_data: pd.DataFrame,
                      main_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, description: str, predictions: pd.DataFrame, a: str, b: str):
        rows.append({"check": check, "description": description, **compare(predictions, a, b)})

    add("main", "Main result: M2 vs M1, common sample, all test years",
        main_predictions, "M2_commodity", "M1_baseline")

    # (a) Without country-years with inflation above 40% (removed from training and testing).
    data = main_data[~main_data["high_inflation"]]
    predictions, _ = run_expanding_window(data, ["M1_baseline", "M2_commodity"])
    add("a_no_high_inflation", "Country-years with inflation above 40% removed",
        predictions, "M2_commodity", "M1_baseline")

    # (b) Country fixed effects on (M1_baseline_FE is already among the main models).
    predictions, _ = run_expanding_window(main_data, ["M2_commodity_FE", "M3_extended_FE"])
    predictions = pd.concat([predictions,
                             main_predictions[main_predictions["model"] == "M1_baseline_FE"]])
    add("b_fixed_effects", "Country fixed effects on: M2_FE vs M1_FE",
        predictions, "M2_commodity_FE", "M1_baseline_FE")
    add("b_fixed_effects_M3", "Country fixed effects on: M3_FE vs M1_FE",
        predictions, "M3_extended_FE", "M1_baseline_FE")
    add("b_fixed_effects_vs_pooled", "M1 with fixed effects vs M1 pooled",
        main_predictions, "M1_baseline_FE", "M1_baseline")

    # (c) Two years of lags (rows where the second lags exist).
    _, lag2_features, _ = ALL_MODELS["M2_lag2"]
    data = main_data.dropna(subset=lag2_features)
    predictions, _ = run_expanding_window(data, ["M1_lag2", "M2_lag2"])
    add("c_two_lags", "Lags t-1 and t-2: M2_lag2 vs M1_lag2",
        predictions, "M2_lag2", "M1_lag2")

    # (d) Test years 2020-2022 left out of the evaluation (models still train on all past years).
    predictions = main_predictions[~main_predictions["year"].isin(EXCLUDED_SHOCK_YEARS)]
    add("d_no_2020_2022", "Test years 2020-2022 excluded from evaluation",
        predictions, "M2_commodity", "M1_baseline")

    # (e) Food only vs energy only (and fertilizers added).
    data = main_data.dropna(subset=["fert_chg_lag1"])
    predictions, _ = run_expanding_window(
        data, ["M1_baseline", "M2_food_only", "M2_energy_only", "M2_plus_fert"])
    add("e_food_only", "Food only: M1 + food_chg_lag1 vs M1",
        predictions, "M2_food_only", "M1_baseline")
    add("e_energy_only", "Energy only: M1 + energy_chg_lag1 vs M1",
        predictions, "M2_energy_only", "M1_baseline")
    add("e_plus_fertilizer", "M2 + fert_chg_lag1 vs M1",
        predictions, "M2_plus_fert", "M1_baseline")

    # (f) Larger sample: rows that have the M1/M2 predictors even if FX or GDP is missing.
    _, m2_features, _ = ALL_MODELS["M2_commodity"]
    data = features_table.dropna(subset=[TARGET] + m2_features)
    predictions, _ = run_expanding_window(data, ["M1_baseline", "M2_commodity"])
    add("f_larger_sample", "M2 vs M1 on all rows with M1/M2 predictors (not only M3 rows)",
        predictions, "M2_commodity", "M1_baseline")
    return pd.DataFrame(rows)


def main() -> dict:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    features_table = pd.read_csv(CLEAN_DIR / "features.csv")
    data = common_sample(features_table)
    print(f"  common sample: {len(data)} country-years, {data['country'].nunique()} countries, "
          f"target years {data['year'].min()}-{data['year'].max()}; "
          f"test years {FIRST_TEST_YEAR}-{data['year'].max()}")

    predictions, details = run_expanding_window(data, list(MAIN_MODELS))
    predictions.to_csv(TABLES_DIR / "oos_predictions.csv", index=False)
    details.to_csv(TABLES_DIR / "model_details.csv", index=False)

    overall = metrics_table(predictions, ["model"])
    overall = overall.set_index("model").loc[list(MAIN_MODELS)].reset_index()
    overall.to_csv(TABLES_DIR / "oos_metrics_overall.csv", index=False)
    metrics_table(predictions, ["model", "country"]).to_csv(
        TABLES_DIR / "oos_metrics_by_country.csv", index=False)
    metrics_table(predictions, ["model", "year"]).to_csv(
        TABLES_DIR / "oos_metrics_by_year.csv", index=False)

    by_country = predictions[predictions["model"].isin(["M1_baseline", "M2_commodity"])]
    by_country = by_country.pivot_table(index="country", columns="model",
                                        values="abs_error", aggfunc="mean")
    by_country["n_years"] = predictions[predictions["model"] == "M1_baseline"] \
        .groupby("country").size()
    by_country["MAE_diff_M2_minus_M1"] = by_country["M2_commodity"] - by_country["M1_baseline"]
    by_country = by_country.rename(columns={"M1_baseline": "MAE_M1", "M2_commodity": "MAE_M2"})
    by_country.sort_values("MAE_diff_M2_minus_M1").reset_index().to_csv(
        TABLES_DIR / "m2_vs_m1_by_country.csv", index=False)

    m2_coefficients_by_test_year(data, predictions).to_csv(
        TABLES_DIR / "m2_coefficients_by_test_year.csv", index=False)

    comparisons = pd.DataFrame([compare(predictions, name, "M1_baseline")
                                for name in MAIN_MODELS if name != "M1_baseline"])
    comparisons.to_csv(TABLES_DIR / "comparisons_vs_m1.csv", index=False)

    robustness = robustness_checks(features_table, data, predictions)
    robustness.to_csv(TABLES_DIR / "robustness_checks.csv", index=False)

    print(overall.round(3).to_string(index=False))
    main_row = robustness.iloc[0]
    print(f"  M2 vs M1: MAE {main_row['MAE_a']:.3f} vs {main_row['MAE_b']:.3f}; "
          f"M2 better in {main_row['countries_a_better']} of {main_row['n_countries']} countries; "
          f"DM p-value (absolute errors) = {main_row['DM_p_abs']:.3f}")
    return {"predictions": predictions, "overall": overall, "robustness": robustness}


if __name__ == "__main__":
    main()
