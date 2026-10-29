"""Step 4: build the forecasting table (target and lagged predictors).

Target: inflation in year t.
Predictors: values from year t-1 (or t-2) only. A forecast made at the end of
year t-1 can use these, so nothing from year t leaks into the predictors.

Lags are built by looking up the same country in an earlier *calendar year*
(not "the row above"), so a gap in the data gives a missing lag instead of a
wrong one.

Output: data/clean/features.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"

TARGET = "inflation"

# (source column in the panel, lag in years) for every predictor we create.
LAGS = [
    ("inflation", 1), ("inflation", 2),
    ("food_chg", 1), ("food_chg", 2),
    ("energy_chg", 1), ("energy_chg", 2),
    ("agri_chg", 1), ("fert_chg", 1),
    ("fx_change", 1), ("gdp_growth", 1),
]
# Short names: inflation lag 1 -> infl_lag1, food_chg lag 1 -> food_chg_lag1, ...
PREDICTORS = [f"{'infl' if col == 'inflation' else col}_lag{k}" for col, k in LAGS]


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per country-year: target, flags and lagged predictors."""
    table = panel[["country", "year", TARGET, "high_inflation"]].copy()
    for (col, k), name in zip(LAGS, PREDICTORS):
        if col not in panel.columns:  # e.g. fertilizers are absent in the FRED fallback
            continue
        past = panel[["country", "year", col]].copy()
        past["year"] += k  # the value from year t-k is placed on row t
        table = table.merge(past.rename(columns={col: name}),
                            on=["country", "year"], how="left")
    return table.sort_values(["year", "country"]).reset_index(drop=True)


def main() -> pd.DataFrame:
    panel = pd.read_csv(CLEAN_DIR / "panel.csv")
    features = make_features(panel)
    features.to_csv(CLEAN_DIR / "features.csv", index=False)
    print(f"  saved data/clean/features.csv ({len(features)} rows); predictors: "
          + ", ".join(c for c in PREDICTORS if c in features.columns))
    return features


if __name__ == "__main__":
    main()
