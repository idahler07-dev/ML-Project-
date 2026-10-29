# Methods

This file describes every step of the analysis, the reason for each choice, and every
result, including all robustness checks. All numbers come from the CSV files in
`results/tables/`, which `python run_all.py` regenerates from the raw data.

## 1. Question and design

**Question.** Do last year's changes in world food and energy prices improve one-year-ahead
forecasts of consumer price inflation in developing and emerging economies, compared with
forecasts that use only each country's own past inflation?

**Design.** A pseudo out-of-sample forecasting competition. For every test year
Y = 2012, …, 2025, every model is estimated on data for years before Y only and then
forecasts inflation in year Y. The forecast errors of a model with commodity prices (M2)
are compared with those of a model without them (M1).

This answers a *forecasting* question ("does the information help predict?"), not a
*causal* one ("how much do commodity prices raise inflation?").

## 2. Data

Sources, URLs, download dates, licences and citations are in [DATA_SOURCES.md](DATA_SOURCES.md).

- **Country data** (World Bank WDI API, 28 countries, 2000–2025): inflation
  (FP.CPI.TOTL.ZG, annual % change in consumer prices), GDP growth (NY.GDP.MKTP.KD.ZG) and
  the official exchange rate (PA.NUS.FCRF, local currency per US dollar, period average).
- **Commodity prices** (World Bank Pink Sheet, `Monthly Indices` sheet, nominal US dollars,
  2010 = 100): Food, Energy, Agriculture and Fertilizers indices, January 1960 to
  September 2026. The FRED fallback was not needed.

### Cleaning (`src/clean.py`)

| Choice | Reason |
|---|---|
| Commodity indices: annual average of the 12 monthly values, then the % change from the previous year | Inflation is an annual average-over-average measure, so the commodity variable is built the same way. |
| A year counts only if all 12 months are present (2026 has 9, so it is dropped) | A partial-year average is not comparable with full years. |
| `fx_change` = % change in the exchange rate from the previous calendar year; positive means the local currency weakened | Depreciation raises import prices. The change is computed only if both years exist, so gaps are never bridged. |
| Missing values are never filled in | Filling in inflation would invent data. |
| Rows with missing inflation (the target) are dropped and logged | They cannot be forecast or scored. 12 of 728 rows were dropped: Uzbekistan 2000–2010 and Tajikistan 2000 (`results/tables/cleaning_log.csv`). |
| Country-years with inflation above 40% are flagged, not removed | There are 9 (Ghana 2000–01, Romania 2000, Turkiye 2000–02 and 2022–24; `results/tables/high_inflation_country_years.csv`). They are real data, so they stay in the main analysis; robustness check (a) removes them. |

Two checks on the raw data: the 2010 average of every commodity index equals 100.0 (as the
file states), and the World Bank exchange rates are already adjusted for currency
redenominations (e.g. Turkiye 2005, Ghana 2007, Romania 2005 show no artificial −99% jumps).
The largest exchange-rate moves are real devaluations, such as Nigeria 2024 (+129%) and
Egypt 2017 (+77%).

**Missing data** (`results/tables/missing_data.csv`): inflation is missing for Uzbekistan
2000–2010 and Tajikistan 2000. The exchange rate is missing for Uzbekistan 2001–2012 and
Viet Nam 2025, so `fx_change` is missing for Uzbekistan until 2013. GDP growth and the
commodity series have no gaps in 2000–2025.

## 3. Features and leakage control (`src/features.py`)

- **Target:** inflation in year t.
- **Predictors:** `infl_lag1`, `infl_lag2`, `food_chg_lag1`, `food_chg_lag2`,
  `energy_chg_lag1`, `energy_chg_lag2`, `agri_chg_lag1`, `fert_chg_lag1`, `fx_change_lag1`,
  `gdp_growth_lag1`. Each is the value from year t−1 (or t−2).
- Lags are built by looking up the same country in the earlier *calendar year*, not "the
  row above". A missing year therefore gives a missing lag, never a wrong one.
- **Tests** (`tests/test_features.py`): for every year T, all year-T data are changed by
  +1000 and the features are rebuilt; the predictors for year T must not change. The test
  was checked by deliberately feeding a year-t variable in as a "lag"; it failed, as it should.
  `tests/test_models.py` checks that scrambling the test-year inflation never changes a
  forecast. `tests/test_evaluate.py` checks that training years always come before the test year.

**Timing assumption.** A forecast for year t is assumed to be made once the annual figures
for year t−1 are known. Annual commodity averages and CPI inflation are known early in
year t; GDP growth is often published later and revised, so M3 may have a small advantage
over a real-time forecaster.

## 4. Models (`src/models.py`)

| Model | Predictors | Method | Why |
|---|---|---|---|
| M0 Naive | `infl_lag1` | forecast = last year's inflation | The simplest benchmark. |
| M1 Baseline | `infl_lag1` | pooled OLS | The "own past inflation only" benchmark the question asks about. |
| M1 + FE | `infl_lag1` | OLS with one intercept per country | Allows each country its own average inflation level. |
| M2 Commodity | M1 + `food_chg_lag1`, `energy_chg_lag1` | pooled OLS | The model the question is about. |
| M3 Extended | M2 + `fx_change_lag1`, `gdp_growth_lag1` | pooled OLS | Adds the two most obvious other drivers. |
| M4 Ridge | M3 predictors (standardised) | ridge regression | Shrinks coefficients to reduce over-fitting. |
| M5 Boosting | M3 predictors | HistGradientBoostingRegressor | A flexible non-linear comparison only. |

Details:

- **Pooled, not one model per country.** Each country has at most 24 usable years, too few
  for a separate regression. Pooling uses all 28 countries to estimate one set of coefficients.
- **Fixed effects.** In an FE model, a country with no training rows has no intercept of its
  own. It gets the average of the estimated country intercepts. This happened once
  (Uzbekistan, test year 2015; `results/tables/model_details.csv`).
- **Ridge alpha.** Chosen from 15 values between 0.001 and 10,000 by time-ordered
  cross-validation *inside the training years*: each of the last 3 training years is
  forecast from the years before it, and the alpha with the lowest mean absolute error wins.
  The smallest alpha (0.001, almost no shrinkage) was chosen in 10 of 14 test years.
- **Boosting settings** were fixed in advance, not tuned (200 trees, learning rate 0.05,
  depth 3, at least 20 rows per leaf, no early stopping, `random_state = 0`). With 270 to 631
  training rows this model is likely to over-fit, so it is only a comparison.
- **Same rows for every model.** All main models are trained and tested on the 659
  country-years (target years 2002–2025) where every M3 predictor exists. This makes their
  errors directly comparable; robustness check (f) uses the larger M1/M2 sample.

## 5. Evaluation (`src/evaluate.py`)

- **Expanding window.** Test years 2012–2025. For test year Y, models train on target
  years 2002 to Y−1. The first window has 10 training years. Data are never shuffled.
- **Forecasts:** 389 country-years (27 countries × 14 years plus Uzbekistan × 11 years).
- **Metrics:** mean absolute error (MAE, the headline), root mean squared error (RMSE,
  which punishes big misses more) and median absolute error, overall, by country and by year.
- **M2 vs M1 tests.**
  - *Diebold-Mariano (DM) test.* Commodity prices are identical for every country in a
    year, so errors from the same year are not independent. The loss difference
    (|error M2| − |error M1|) is therefore averaged over countries within each year, and
    the DM test asks whether the mean of these 14 yearly values is zero. It uses the
    Harvey-Leybourne-Newbold small-sample correction and a t distribution with 13 degrees of
    freedom. It is also reported for squared errors.
  - *Share of countries* where M2 has the lower MAE, with a sign (binomial) test. The sign
    test treats countries as independent even though they share the same commodity shocks,
    so its p-values overstate the evidence. It is reported only as a description.

## 6. Results

### Main comparison (`oos_metrics_overall.csv`, `comparisons_vs_m1.csv`)

| Model | MAE | RMSE | MedAE | MAE vs M1 | Countries better than M1 | DM p (abs) | DM p (squared) |
|---|---|---|---|---|---|---|---|
| M0 Naive | 2.611 | 4.911 | 1.398 | −4.0% | 15/28 | 0.412 | 0.939 |
| M1 Baseline | 2.719 | 4.936 | 1.852 | — | — | — | — |
| M1 + FE | 2.604 | 4.856 | 1.619 | −4.2% | 16/28 | 0.108 | 0.249 |
| **M2 Commodity** | **2.720** | **4.982** | **1.808** | **+0.04%** | **14/28** | **0.986** | **0.560** |
| M3 Extended | 2.654 | 4.849 | 1.768 | −2.4% | 15/28 | 0.596 | 0.435 |
| M4 Ridge | 2.694 | 4.912 | 1.789 | −0.9% | 13/28 | 0.845 | 0.802 |
| M5 Boosting | 2.690 | 5.261 | 1.457 | −1.1% | 18/28 | 0.839 | 0.337 |

**M2 does not improve on M1.** The MAE is 2.720 vs 2.719, the RMSE is slightly worse (4.982 vs
4.936), M2 wins in exactly half of the countries, and the DM test gives p = 0.986. No model
beats M1 by a statistically clear margin. The naive forecast and M1 with fixed effects
have the lowest MAE; M3 has the lowest RMSE.

### By year (`oos_metrics_by_year.csv`)

M2 had a lower MAE than M1 in 6 of the 14 test years. The biggest differences:

- **2022:** M2 was much worse (MAE 7.12 vs 6.51). Inflation surged (mean actual 12.77%).
- **2024:** M2 was better (3.67 vs 4.15). Commodity prices had fallen in 2023 and so did inflation.

`results/tables/m2_coefficients_by_test_year.csv` shows why 2022 went wrong. In every
training window up to 2021, the estimated coefficient on `energy_chg_lag1` was slightly
negative (−0.019 when forecasting 2022). So when energy prices rose 81.1% in 2021, M2
forecast *lower* 2022 inflation than M1 (mean forecast 5.68% vs 6.41%). The energy
coefficient turned positive (0.008) only after 2022 entered the training data. The food
coefficient stayed positive throughout (0.012–0.031). One possible reason for the earlier
negative energy coefficient is the 2008–2009 episode: energy prices rose 36.7% in 2008, but
median inflation fell from 8.95% in 2008 to 5.42% in 2009 as the global financial crisis hit.

### By country (`m2_vs_m1_by_country.csv`, figure 4)

M2 was better in 14 countries and worse in 14. All differences are small: from −0.19
percentage points (Ghana) to +0.15 (Uzbekistan).

**Uzbekistan** (11 test years, 2015–2025): M1 MAE 2.16, M2 2.31, M3 1.51, M4 1.49
(`oos_metrics_by_country.csv`). For Uzbekistan, the exchange rate mattered more than
commodity prices. The som lost value sharply in 2017 (+72.5% local currency per US dollar)
and inflation peaked at 17.5% in 2018.

### Full-sample coefficients (`coefficients_full_sample.csv`, figure 6)

These are estimated once on all 659 country-years (2002–2025), with standard errors
clustered by year (24 clusters). They describe the data; they are not forecasts.

| Variable | M2 coefficient (95% CI) | p | M3 coefficient (95% CI) | p |
|---|---|---|---|---|
| Last year's inflation | 0.679 (0.555 to 0.804) | <0.001 | 0.540 (0.408 to 0.672) | <0.001 |
| Food price change, t−1 | 0.030 (−0.118 to 0.178) | 0.677 | 0.061 (−0.083 to 0.206) | 0.388 |
| Energy price change, t−1 | 0.010 (−0.038 to 0.058) | 0.671 | 0.011 (−0.040 to 0.062) | 0.661 |
| Exchange-rate change, t−1 | — | — | 0.125 (0.048 to 0.202) | 0.003 |
| GDP growth, t−1 | — | — | 0.124 (−0.059 to 0.308) | 0.175 |

R²: M1 0.538, M2 0.547, M3 0.579. A 10% rise in world food prices is associated with about
0.3 percentage points higher inflation the next year (M2), but the interval includes zero.
Last year's currency depreciation is the only added variable with a clearly non-zero effect.

## 7. Robustness checks (`robustness_checks.csv`)

Each row compares a model with commodity prices (A) against the matching model without them (B)
on exactly the same country-years. A negative difference means A had the smaller error.

| Check | A vs B | Forecasts | MAE A | MAE B | Diff. | RMSE A | RMSE B | A better in | Sign p | DM p (abs) | DM p (sq.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Main | M2 vs M1 | 389 | 2.720 | 2.719 | +0.04% | 4.982 | 4.936 | 14/28 | 1.000 | 0.986 | 0.560 |
| (a) No inflation > 40% | M2 vs M1 | 386 | 2.544 | 2.555 | −0.44% | 3.895 | 3.877 | 17/28 | 0.345 | 0.854 | 0.763 |
| (b) Fixed effects on | M2+FE vs M1+FE | 389 | 2.530 | 2.604 | −2.85% | 4.873 | 4.856 | 24/28 | <0.001 | 0.238 | 0.620 |
| (b) Fixed effects on | M3+FE vs M1+FE | 389 | 2.515 | 2.604 | −3.42% | 4.744 | 4.856 | 17/28 | 0.345 | 0.318 | 0.480 |
| (b) FE vs pooled | M1+FE vs M1 | 389 | 2.604 | 2.719 | −4.22% | 4.856 | 4.936 | 16/28 | 0.572 | 0.108 | 0.249 |
| (c) Two years of lags | M2_lag2 vs M1_lag2 | 389 | 2.764 | 2.717 | +1.71% | 5.073 | 4.977 | 13/28 | 0.851 | 0.677 | 0.429 |
| (d) Test years 2020–22 excluded | M2 vs M1 | 305 | 2.475 | 2.519 | −1.74% | 3.822 | 3.851 | 21/28 | 0.013 | 0.520 | 0.237 |
| (e) Food only | M1+food vs M1 | 389 | 2.683 | 2.719 | −1.34% | 4.919 | 4.936 | 22/28 | 0.004 | 0.359 | 0.341 |
| (e) Energy only | M1+energy vs M1 | 389 | 2.758 | 2.719 | +1.42% | 4.990 | 4.936 | 8/28 | 0.036 | 0.590 | 0.511 |
| (e) Fertilizers added | M2+fert vs M1 | 389 | 2.873 | 2.719 | +5.68% | 5.169 | 4.936 | 6/28 | 0.004 | 0.360 | 0.320 |
| (f) Larger sample | M2 vs M1 | 392 | 2.616 | 2.604 | +0.44% | 4.829 | 4.806 | 9/28 | 0.087 | 0.844 | 0.658 |

How each check was done, and why:

- **(a)** Country-years with inflation above 40% are removed from both training and
  testing. Of the 9 flagged rows, 4 are in the main sample: Turkiye 2002 (training) and
  Turkiye 2022–2024 (testing). This uses the actual outcome to choose rows, which a real
  forecaster could not do, so it is a sensitivity check only.
- **(b)** The same models with one intercept per country.
- **(c)** "Two-year lags" is read as *adding* the second lags (inflation, food and energy at
  t−2), as allowed by the `infl_lag2` variant in the plan. TJK 2002 has no `infl_lag2`, so
  the training data lose one row.
- **(d)** The models are still trained on every earlier year (as a forecaster in 2023 would
  be), but test years 2020, 2021 and 2022 are not scored.
- **(e)** Food and energy enter separately. Fertilizers are in the Pink Sheet, so a version
  with them added is also shown.
- **(f)** All rows that have the M1/M2 predictors, even if the exchange rate or GDP growth is
  missing. This adds Uzbekistan 2012–2014 to testing and the 2001 target year to training.
  M1's MAE here is 2.604, compared with 2.719 in the main sample. This is mainly because the
  2001 training year (with very high inflation in Turkiye and Ghana) changes the estimated
  coefficients: the pooled baseline is sensitive to which early years are in the training
  data. M2 still does not beat M1.

**Reading the robustness results.** The year-level DM test finds no significant difference
in any check (all p > 0.2). A few patterns are consistent but weak. Food prices help a
little more often than energy prices (22/28 vs 8/28 countries). M2 does a little better
when the 2020–2022 shocks are not scored, and with country fixed effects. The sign-test
p-values below 0.05 should not be read as proof: they ignore the fact that all countries
face the same commodity shock each year, and 15 different comparisons were made in total,
so a few small p-values are expected by chance.

## 8. Limitations

1. **Correlation is not causation.** A forecasting gain (or its absence) says nothing
   direct about the causal effect of commodity prices on inflation. The effect may be real
   but too small, too quick (within the same year, which a lag cannot capture) or too
   variable to help forecasts.
2. **Small annual sample.** There are 14 test years and 24 target years in total.
   Commodity prices take only one value per year for all countries, so the commodity
   coefficients rest on about two dozen data points, and the DM test has low power. A modest
   true improvement could go undetected.
3. **Shocks and structural breaks.** The global financial crisis (2008–09), COVID-19 (2020)
   and the 2021–22 energy and food price surge behaved very differently. The energy
   coefficient changed sign after 2022. Monetary policy regimes also changed (for example,
   inflation targeting in several countries).
4. **Data revisions.** The World Bank revises past data. We use the vintage last updated on
   2026-10-08, not the data available to forecasters at the time, so this is not a true
   real-time exercise.
5. **Differences between countries.** Oil exporters (Nigeria, Kazakhstan) and importers,
   regulated fuel prices, food's share of the consumer basket and exchange-rate regimes all
   differ. One pooled set of coefficients ignores this.
6. **Missing data.** Uzbekistan enters the test only from 2015, and Tajikistan has no 2000
   inflation. Missing values were not filled in, so the samples differ slightly between
   checks; each comparison always uses identical rows for both models.
7. **Annual data and simple lags.** Monthly or quarterly data, or within-year effects, might
   show something annual one-year-ahead forecasts cannot.

## 9. Reproducibility

- `python run_all.py --no-download` reproduces every table and figure exactly from the
  committed raw files. It was run from a fresh clone with a new virtual environment and
  gave byte-identical results.
- `python run_all.py` downloads fresh data first. Re-downloading on 2026-10-29 gave
  identical raw files.
- Package versions are pinned in `requirements.txt` (Python 3.13). Random seeds are fixed
  (`random`, `numpy`, and `random_state = 0` for boosting); ridge and OLS have no randomness.
- `python -m pytest` runs 35 tests: no look-ahead leakage in the features, training years
  always before the test year, forecasts never depending on the test-year targets, and every
  number in README.md matching `results/tables/` (`tests/test_readme_numbers.py`).
