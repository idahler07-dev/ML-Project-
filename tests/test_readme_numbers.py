"""Check that the numbers quoted in README.md match results/tables/.

If the data or code change and the results move, this test fails until the
README is updated, so the README can never silently disagree with the results.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
README = (ROOT / "README.md").read_text()

# README key-table row label -> model name in the results tables.
TABLE_ROWS = {
    "M0 Naive": "M0_naive",
    "M1 Baseline": "M1_baseline",
    "M1 + fixed effects": "M1_baseline_FE",
    "**M2 Commodity**": "M2_commodity",
    "M3 Extended": "M3_extended",
    "M4 Ridge": "M4_ridge",
    "M5 Gradient boosting": "M5_boosting",
}


@pytest.fixture(scope="module")
def overall():
    return pd.read_csv(TABLES / "oos_metrics_overall.csv").set_index("model")


@pytest.fixture(scope="module")
def robustness():
    return pd.read_csv(TABLES / "robustness_checks.csv").set_index("check")


def table_row(label: str) -> list:
    """Return the numbers in the README table row that starts with `label`."""
    for line in README.splitlines():
        if line.startswith(f"| {label} |"):
            cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
            return [float(c) for c in cells[2:]]
    raise AssertionError(f"README table row not found: {label}")


@pytest.mark.parametrize("label,model", TABLE_ROWS.items())
def test_key_table_matches_results(label, model, overall):
    mae, rmse, medae = table_row(label)
    assert mae == round(overall.loc[model, "MAE"], 2)
    assert rmse == round(overall.loc[model, "RMSE"], 2)
    assert medae == round(overall.loc[model, "MedAE"], 2)


def test_headline_numbers(overall, robustness):
    main = robustness.loc["main"]
    assert f"**{main['MAE_a']:.2f} vs {main['MAE_b']:.2f} percentage points**" in README
    assert f"**{main['countries_a_better']} of {main['n_countries']}**" in README
    assert f"(**p = {main['DM_p_abs']:.2f}**)" in README
    assert f"**{overall.loc['M0_naive', 'MAE']:.2f}**" in README
    n = int(overall.loc["M2_commodity", "n"])
    assert f"({n} country-years)" in README
    assert f"{int(main['n_countries'])} developing and emerging economies" in README


def test_food_and_energy_numbers(robustness):
    food, energy = robustness.loc["e_food_only"], robustness.loc["e_energy_only"]
    assert (f"(**{food['MAE_a']:.2f}** vs **{food['MAE_b']:.2f}**, better in "
            f"{food['countries_a_better']} of {food['n_countries']} countries)") in README
    assert f"(**{energy['MAE_a']:.2f}**)" in README
    assert f"(p = {food['DM_p_abs']:.2f} and p = {energy['DM_p_abs']:.2f})" in README


def test_energy_jump_and_test_years():
    commodities = pd.read_csv(ROOT / "data" / "clean" / "commodities_annual.csv")
    jump = commodities.set_index("year").loc[2021, "energy_chg"]
    assert f"energy prices jumped {jump:.0f}% in 2021" in README
    predictions = pd.read_csv(TABLES / "oos_predictions.csv")
    first, last = predictions["year"].min(), predictions["year"].max()
    assert f"{first} to {last}" in README and f"{first}–{last}" in README


def test_no_robustness_check_is_significant_by_dm(robustness):
    """README says no robustness check shows a statistically clear gain."""
    assert "None shows a statistically clear gain" in README
    assert (robustness["DM_p_abs"] > 0.05).all() and (robustness["DM_p_squared"] > 0.05).all()


def test_readme_is_under_800_words():
    assert len(re.findall(r"\S+", README)) < 800
