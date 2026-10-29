# Learning guide: the whole project, step by step

This guide explains the project to a smart high-school student who has not studied
economics or machine learning. Every number here comes from the files in `results/tables/`.

## 1. The question

When world food and oil prices jump, do prices in shops in Uzbekistan, Kenya or Brazil also
rise the next year? And, more precisely: **if you want to predict next year's inflation, does
knowing what happened to world food and energy prices last year help, compared with just
looking at the country's own past inflation?**

## 2. What is inflation?

Inflation is how fast prices rise. Statisticians track the price of a fixed "basket" of
things people buy (bread, rent, fuel, bus fares, phones). This is the **consumer price
index (CPI)**. If the basket cost 100 last year and costs 110 this year, inflation is
**10%**. We use the World Bank's annual inflation numbers for 28 developing and emerging
economies from 2000 to 2025. For example, Uzbekistan's inflation was 17.5% in 2018 and
8.8% in 2025.

## 3. What is a commodity price index?

Commodities are raw materials traded worldwide: wheat, rice, cooking oil, crude oil, gas,
fertilizer. The World Bank publishes their prices every month in the **"Pink Sheet"**. A
**price index** combines many prices into one number scaled so that a chosen base period
equals 100. Here the base is the 2010 average. A food index of 125 means food commodities
cost 25% more than in 2010. We use the **food** and **energy** indices, turned into yearly
averages and then into yearly % changes. For example, the energy index rose 81% in 2021 and
fell 30% in 2023.

Why might they matter? Food is a large part of what households buy in poorer countries, and
energy affects transport and production costs. So the idea that world prices "pass through"
to local inflation is reasonable. Whether this helps *forecasting* is what we test.

## 4. Why are the variables "lagged"?

To forecast next year, you can only use what you know **today**. A forecast for 2025 made at
the start of 2025 can use 2024 data, but not 2025 data. So every predictor is **lagged**:
for the year we predict (call it *t*), we use values from year *t − 1*. For example,
`food_chg_lag1` for 2025 is the change in world food prices in 2024.

Using year-*t* data by mistake is called **look-ahead leakage**. It makes a model look
brilliant on paper and useless in reality. The project has an automatic test that changes
all the data of a year and checks that the predictors for that same year do not change.
It passes. When we deliberately broke the code to use same-year food prices, the test
failed, as it should.

## 5. The models

- **M0 Naive:** "next year's inflation = this year's inflation." No maths at all.
- **M1 Baseline:** a straight-line formula, *inflation(t) = a + b × inflation(t−1)*, fitted to
  the data of all countries together. Fitted on all years, a = 2.045 and b = 0.688, so a
  country with 10% inflation this year is forecast at about 2.045 + 0.688 × 10 ≈ 8.9% next
  year: high inflation is expected to drift back toward the average.
- **M2 Commodity:** M1 plus last year's food and energy price changes. **This is the model
  the question is about.**
- **M3 Extended:** M2 plus last year's change in the exchange rate and GDP growth.
- **M4 Ridge:** like M3, but it "shrinks" the coefficients a little to avoid over-fitting.
- **M5 Gradient boosting:** a machine-learning model made of many small decision trees. It is
  only a comparison, because we have few data points for such a flexible model.

**Fitting** a model means choosing the numbers (like a and b) that make its past
predictions as close as possible to what actually happened.

## 6. What does "out-of-sample" testing mean?

If you grade a student on the exact questions they practised, you learn nothing about how
they will do in the real exam. Models are the same: a model judged on the data it was fitted
on always looks too good.

So we pretend to be a forecaster living through history:

1. Stand at the start of 2012. Fit every model using only 2002–2011. Forecast 2012.
2. Move to 2013. Refit with 2002–2012. Forecast 2013.
3. Repeat every year up to 2025.

This is an **expanding window**, and the data are never shuffled, so a model never sees the
future. It produced **389 forecasts** (28 countries; 14 years each, except Uzbekistan with 11).

## 7. How do we measure forecast errors?

The **error** is forecast minus actual. If we forecast 8% and inflation was 10%, the error is
−2 percentage points.

- **MAE (mean absolute error):** ignore the sign and average the size of the errors.
  Errors of 1, 2 and 6 give an MAE of 3. It is easy to read: "on average we are off by 3 points."
- **RMSE (root mean squared error):** square the errors, average them, take the square root.
  The same errors give √((1 + 4 + 36)/3) ≈ 3.7. Squaring punishes big misses more, so RMSE is
  sensitive to surprises like 2022.
- **Median absolute error:** the middle error size. It is not pulled up by a few huge misses.

Lower is better for all three.

## 8. Is a difference real or luck? The Diebold-Mariano test

Even two equally good models will have slightly different errors by chance. The
**Diebold-Mariano (DM) test** asks whether the difference in errors is bigger than chance
would usually produce. It gives a **p-value**: roughly, how surprising the observed
difference would be if the two models were truly equally good. A small p-value (below 0.05)
suggests a real difference; a large one means we cannot tell them apart.

One subtlety: world commodity prices are the same for every country in a given year. So the
28 countries do not give 28 independent pieces of evidence each year. We therefore average
the error difference over countries for each year and test those 14 yearly numbers.

## 9. What did we find?

| Model | MAE | RMSE |
|---|---|---|
| M0 Naive | 2.61 | 4.91 |
| M1 Baseline | 2.72 | 4.94 |
| M1 + country fixed effects | 2.60 | 4.86 |
| **M2 + food & energy** | **2.72** | **4.98** |
| M3 + exchange rate & GDP | 2.65 | 4.85 |
| M4 Ridge | 2.69 | 4.91 |
| M5 Gradient boosting | 2.69 | 5.26 |

1. **Commodity prices did not improve the forecasts.** M2's MAE (2.720) is essentially the
   same as M1's (2.719). M2 was better in 14 countries and worse in 14. The DM test gives
   p = 0.99, which means no detectable difference.
2. **Simple beats fancy.** The naive "same as last year" forecast (2.61) did about as well as
   anything. Machine learning (M5) did not help.
3. **Food vs energy.** Adding only food prices helped a little (2.68, better in 22 of 28
   countries). Adding only energy prices made forecasts a little worse (2.76). Neither
   difference is statistically clear (DM p = 0.36 and p = 0.59).
4. **Relationships change over time.** Before 2022, the historical data said higher energy
   prices were followed by slightly *lower* inflation (one example: energy prices rose in
   2008, but inflation fell in 2009 during the global financial crisis). So when energy prices
   jumped 81% in 2021, M2 predicted *lower* 2022 inflation than M1 (5.68% vs 6.41% on
   average), while actual inflation averaged 12.77%. Learning from the past works badly when
   the world changes.
5. **The exchange rate matters more.** In the full-sample regression, when the local
   currency weakens so that a US dollar costs 10% more, inflation the next year is about
   1.25 percentage points higher, and this is statistically clear. The food and energy effects are small and not
   statistically clear. For Uzbekistan, models with the exchange rate did best (MAE 1.51 for
   M3 vs 2.16 for M1). This fits Uzbekistan's 2017 currency drop (+72.5% som per dollar),
   which was followed by 17.5% inflation in 2018.

**A null result is still a result.** It tells forecasters that, at the yearly level, adding
last year's world food and energy prices to a simple model does not reliably help.

## 10. Why might commodity prices not help?

These are possible explanations, not things this project tested:

- Much of the effect may happen **within the same year**, which a one-year lag misses.
- Governments often **subsidise or regulate** fuel and bread prices.
- **Exchange rates** and **local harvests** can matter more than world prices.
- With only about **24 years** of commodity data, any formula is estimated from very few
  "shocks", so it is noisy.

## 11. Limitations (in plain words)

- **Correlation is not causation.** Forecasting tests prediction, not cause and effect.
- **Small sample.** 14 test years is not much, so a small true benefit could be missed.
- **Shocks and breaks.** The 2008 crisis, COVID-19 and the 2021–22 price surge all behaved
  differently.
- **Data revisions.** We use today's corrected data, not what forecasters actually had then.
- **Countries differ.** One formula for oil exporters and importers, rich and poor, is a simplification.
- **Missing data.** Uzbekistan's inflation data start in 2011 (and its exchange rate in
  2013), so it enters the test only in 2015.

## 12. Ten likely interview questions (with short answers)

1. **What was your question, and what did you find?**
   Whether last year's world food and energy price changes improve one-year-ahead inflation
   forecasts for 28 developing economies. They did not: the error was 2.72 percentage points
   with or without them, and the difference was not statistically significant (p = 0.99).

2. **Why did you only use last year's data as predictors?**
   A real forecast can only use information available at the time. Using same-year data
   would be look-ahead leakage. I wrote an automated test that fails if any predictor uses
   year-t data, and I checked that it catches a deliberate mistake.

3. **What is out-of-sample testing and why does it matter?**
   Each year from 2012 to 2025 is forecast using only earlier years, as a real forecaster
   would. Judging a model on data it was fitted on makes it look better than it really is.

4. **What is the difference between MAE and RMSE?**
   MAE is the average size of the errors; RMSE squares them first, so big misses count more.
   M2 had the same MAE as M1 (2.72) but a slightly higher RMSE (4.98 vs 4.94).

5. **How do you know the result is not just luck?**
   I used a Diebold-Mariano test. Because commodity prices are the same for every country
   in a year, I averaged the error differences by year before testing, so I do not pretend
   to have more independent evidence than I have. The p-value was 0.99.

6. **Did anything beat the simple models?**
   No model beat the baseline by a statistically clear margin. The naive "same as last
   year" forecast (MAE 2.61) and the baseline with country fixed effects (2.60) had the
   lowest average errors. Gradient boosting did not help, probably because there is too
   little data for such a flexible model.

7. **Why did the commodity model do badly in 2022?**
   Before 2022, the data suggested that higher energy prices were followed by slightly lower
   inflation, so the model's energy coefficient was negative (−0.019). When energy prices
   rose 81% in 2021, it predicted less inflation for 2022 than the baseline, and inflation
   actually surged. This is a structural break: a relationship learned from the past stopped
   holding.

8. **What robustness checks did you run?**
   Removing inflation above 40%, adding country fixed effects, using two years of lags,
   not scoring 2020–2022, and testing food and energy separately. None gave a statistically
   clear improvement. Food alone was slightly better in 22 of 28 countries, and energy alone
   slightly worse in 20 of 28.

9. **What would you do next?**
   Use monthly or quarterly data to catch faster effects, use "real-time" data vintages,
   look at country groups (oil exporters vs importers, high vs low food share), and add
   local food prices or the exchange rate more carefully. For Uzbekistan, the exchange rate
   already helped more than commodity prices (MAE 1.51 for M3 vs 2.16 for M1).

10. **What are the main limitations?**
    Correlation is not causation; the sample is small (14 test years, about 24 years of
    commodity data); shocks like 2008, 2020 and 2022 break relationships; the data are
    revised; countries differ; and some data are missing (Uzbekistan's inflation starts in 2011).
