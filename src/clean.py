"""Step 3: clean the raw downloads into a tidy annual panel.

Output
  data/clean/panel.csv                 one row per country-year (target not missing)
  data/clean/commodities_monthly.csv   monthly commodity indices used
  data/clean/commodities_annual.csv    annual averages and annual % changes
  results/tables/missing_data.csv      missing-data report (before anything is dropped)
  results/tables/high_inflation_country_years.csv   country-years with inflation > 40%
  results/tables/cleaning_log.csv      how many rows were kept or dropped

Rules
  - Missing values are never filled in.
  - Rows with missing inflation (the target) are dropped and counted.
  - Country-years with inflation above 40% are flagged, not removed.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
TABLES_DIR = ROOT / "results" / "tables"

HIGH_INFLATION = 40.0  # % per year; above this a country-year is flagged

# Pink Sheet "Monthly Indices" header labels (without the "**" footnote marks)
# and the names used in this project.
PINK_SHEET_SERIES = {"Food": "food", "Energy": "energy",
                     "Agriculture": "agri", "Fertilizers": "fert"}
FRED_SERIES = {"PFOODINDEXM": "food", "PNRGINDEXM": "energy"}


def latest_download_log() -> dict:
    """Return the most recent download log (file names contain the date)."""
    logs = sorted(RAW_DIR.glob("download_log_*.json"))
    if not logs:
        raise FileNotFoundError("No download log in data/raw. Run src/download.py first.")
    log = json.loads(logs[-1].read_text())
    if log["errors"]:
        raise RuntimeError(f"The latest download ({logs[-1].name}) had errors: {log['errors']}")
    return log


# ---------------------------------------------------------------- country data

def load_country_data(log: dict) -> pd.DataFrame:
    """Read the World Bank JSON files into one table: country, year, indicators."""
    from download import INDICATORS  # indicator code -> short name

    tables = []
    for code, name in INDICATORS.items():
        records = json.loads((RAW_DIR / log["world_bank_api"][code]["file"]).read_text())
        table = pd.DataFrame({
            "country": [r["countryiso3code"] for r in records],
            "country_name": [r["country"]["value"] for r in records],
            "year": [int(r["date"]) for r in records],
            name: [r["value"] for r in records],  # None stays missing (NaN)
        })
        tables.append(table.set_index(["country", "country_name", "year"]))
    panel = pd.concat(tables, axis=1).reset_index()
    return panel.sort_values(["country", "year"]).reset_index(drop=True)


def add_fx_change(panel: pd.DataFrame) -> pd.DataFrame:
    """Annual % change in the exchange rate (local currency per US dollar).

    Positive = the local currency lost value against the dollar. Computed only
    when the previous year exists for the same country (no gaps are bridged).
    """
    previous = panel[["country", "year", "fx_rate"]].copy()
    previous["year"] += 1
    previous = previous.rename(columns={"fx_rate": "fx_rate_prev"})
    panel = panel.merge(previous, on=["country", "year"], how="left")
    panel["fx_change"] = (panel["fx_rate"] / panel["fx_rate_prev"] - 1) * 100
    return panel.drop(columns="fx_rate_prev")


# -------------------------------------------------------------- commodity data

def read_pink_sheet(path: Path) -> pd.DataFrame:
    """Read the 'Monthly Indices' sheet: one row per month, one column per index."""
    sheet = pd.read_excel(path, sheet_name="Monthly Indices", header=None)
    is_month = sheet[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")
    header = sheet.loc[: is_month.idxmax() - 1]  # the rows above the first month

    # The header is spread over several rows; find the column of each index by label.
    columns = {}
    for col in sheet.columns[1:]:
        for label in header[col].dropna().astype(str):
            clean_label = re.sub(r"[*\s]+$", "", label.strip())
            if clean_label in PINK_SHEET_SERIES:
                columns[PINK_SHEET_SERIES[clean_label]] = col
    missing = set(PINK_SHEET_SERIES.values()) - set(columns)
    if {"food", "energy"} & missing:
        raise ValueError(f"Food or Energy index not found in {path.name}")

    months = sheet[is_month]
    monthly = pd.DataFrame({"date": pd.PeriodIndex(months[0].str.replace("M", "-"), freq="M")})
    for name, col in columns.items():
        # "…" marks a missing value in the file; it becomes NaN here.
        monthly[name] = pd.to_numeric(months[col], errors="coerce").to_numpy()
    return monthly


def read_fred(log: dict) -> pd.DataFrame:
    """Fallback: read the FRED CSV files (food and energy only)."""
    monthly = None
    for series, name in FRED_SERIES.items():
        table = pd.read_csv(RAW_DIR / log["commodities"]["files"][series])
        table = table.rename(columns={table.columns[0]: "date", series: name})
        table["date"] = pd.PeriodIndex(pd.to_datetime(table["date"]), freq="M")
        table[name] = pd.to_numeric(table[name], errors="coerce")
        monthly = table if monthly is None else monthly.merge(table, on="date", how="outer")
    return monthly


def annual_commodity_changes(monthly: pd.DataFrame) -> pd.DataFrame:
    """Annual average of each index (full years only), then annual % change."""
    monthly = monthly.assign(year=monthly["date"].dt.year)
    names = [c for c in monthly.columns if c not in ("date", "year")]
    annual = monthly.groupby("year")[names].mean()
    # A year counts only if all 12 months are present (e.g. a partial 2026 is dropped).
    months_present = monthly.groupby("year")[names].count()
    annual = annual.where(months_present == 12)
    annual = annual.dropna(how="all")
    changes = (annual / annual.shift(1) - 1) * 100
    # shift(1) uses the row above; make sure that row really is the previous year.
    consecutive = pd.Series(annual.index, index=annual.index).diff() == 1
    changes = changes.where(consecutive, axis=0)
    result = annual.add_suffix("_index").join(changes.add_suffix("_chg"))
    return result.reset_index()


# --------------------------------------------------------------------- reports

def missing_data_report(panel: pd.DataFrame, commodities: pd.DataFrame) -> pd.DataFrame:
    """Count missing values per country and variable, and list the missing years."""
    rows = []
    for variable in ["inflation", "gdp_growth", "fx_rate", "fx_change"]:
        for (country, name), group in panel.groupby(["country", "country_name"]):
            missing_years = group.loc[group[variable].isna(), "year"].tolist()
            rows.append({"country": country, "country_name": name, "variable": variable,
                         "years_total": len(group), "years_missing": len(missing_years),
                         "missing_years": " ".join(map(str, missing_years))})
    years = range(panel["year"].min(), panel["year"].max() + 1)
    world = commodities.set_index("year").reindex(years)
    for variable in [c for c in world.columns if c.endswith("_chg")]:
        missing_years = world.index[world[variable].isna()].tolist()
        rows.append({"country": "WORLD", "country_name": "World (Pink Sheet)",
                     "variable": variable, "years_total": len(world),
                     "years_missing": len(missing_years),
                     "missing_years": " ".join(map(str, missing_years))})
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    log = latest_download_log()

    # Country panel.
    panel = add_fx_change(load_country_data(log))

    # Commodity indices.
    if log["commodities"]["source"] == "World Bank Pink Sheet":
        monthly = read_pink_sheet(RAW_DIR / log["commodities"]["file"])
    else:
        monthly = read_fred(log)
    commodities = annual_commodity_changes(monthly)
    monthly.assign(date=monthly["date"].astype(str)).to_csv(
        CLEAN_DIR / "commodities_monthly.csv", index=False)
    commodities.to_csv(CLEAN_DIR / "commodities_annual.csv", index=False)
    print(f"  commodity source: {log['commodities']['source']}; full years "
          f"{int(commodities['year'].min())}-{int(commodities['year'].max())}; "
          f"series: {[c[:-4] for c in commodities.columns if c.endswith('_chg')]}")

    # Missing-data report, before anything is dropped.
    missing_data_report(panel, commodities).to_csv(TABLES_DIR / "missing_data.csv", index=False)

    # Join the global commodity changes to every country-year.
    change_columns = ["year"] + [c for c in commodities.columns if c.endswith("_chg")]
    panel = panel.merge(commodities[change_columns], on="year", how="left")

    # Drop rows with a missing target (inflation). Nothing is filled in.
    rows_before = len(panel)
    dropped = panel[panel["inflation"].isna()]
    panel = panel[panel["inflation"].notna()].copy()
    print(f"  dropped {len(dropped)} of {rows_before} country-years with missing inflation: "
          + ", ".join(f"{c} {y}" for c, y in zip(dropped["country"], dropped["year"])))

    # Flag very high inflation (kept in the main analysis).
    panel["high_inflation"] = panel["inflation"] > HIGH_INFLATION
    high = panel.loc[panel["high_inflation"], ["country", "country_name", "year", "inflation"]]
    high.to_csv(TABLES_DIR / "high_inflation_country_years.csv", index=False)
    print(f"  flagged {len(high)} country-years with inflation above {HIGH_INFLATION:.0f}%: "
          + ", ".join(f"{c} {y}" for c, y in zip(high["country"], high["year"])))

    pd.DataFrame([
        {"step": "country-years downloaded", "rows": rows_before},
        {"step": "dropped: inflation missing", "rows": len(dropped)},
        {"step": "kept in clean panel", "rows": len(panel)},
        {"step": f"flagged: inflation above {HIGH_INFLATION:.0f}% (kept)", "rows": len(high)},
    ]).to_csv(TABLES_DIR / "cleaning_log.csv", index=False)

    panel.to_csv(CLEAN_DIR / "panel.csv", index=False)
    print(f"  saved data/clean/panel.csv ({len(panel)} rows, "
          f"{panel['country'].nunique()} countries, {panel['year'].min()}-{panel['year'].max()})")
    return panel


if __name__ == "__main__":
    main()
