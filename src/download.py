"""Step 2: download all raw data.

Downloads
  a) Country indicators from the World Bank API (JSON, paginated).
  b) Global commodity price indices from the World Bank "Pink Sheet"
     (CMO Historical Data Monthly, Excel). Fallback: FRED CSV files.

Every raw file is saved in data/raw/ with today's date in its name and is never
edited afterwards. A log of what was downloaded (and what failed) is written to
data/raw/download_log_<date>.json.

If every source for a dataset fails, this script raises an error. It never
creates replacement data.
"""

import datetime as dt
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

COUNTRIES = [
    "UZB", "KAZ", "KGZ", "TJK", "GEO", "ARM", "TUR", "POL", "ROU", "EGY",
    "MAR", "NGA", "KEN", "GHA", "ZAF", "IND", "PAK", "BGD", "IDN", "PHL",
    "VNM", "THA", "MYS", "BRA", "MEX", "COL", "PER", "CHL",
]

INDICATORS = {
    "FP.CPI.TOTL.ZG": "inflation",     # Inflation, consumer prices (annual %)
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",  # GDP growth (annual %)
    "PA.NUS.FCRF": "fx_rate",          # Official exchange rate (LCU per USD, period avg)
}

START_YEAR = 2000
WB_API = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"

# Page that links to the current Pink Sheet file.
CMO_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
# Fallback: FRED series (IMF primary commodity prices), no API key needed.
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_SERIES = {"PFOODINDEXM": "food", "PNRGINDEXM": "energy"}

HEADERS = {"User-Agent": "commodity-inflation-research/1.0 (student project)"}


def today() -> str:
    return dt.date.today().isoformat()


def get(url: str, params=None, tries: int = 3, timeout: int = 60) -> requests.Response:
    """GET with a few retries. Raises the last error if all tries fail."""
    last_error = None
    for attempt in range(tries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise last_error


def download_world_bank_indicator(indicator: str, end_year: int) -> list:
    """Download one indicator for all countries, following every page."""
    url = WB_API.format(countries=";".join(COUNTRIES), indicator=indicator)
    records, page, pages = [], 1, 1
    while page <= pages:
        params = {"format": "json", "date": f"{START_YEAR}:{end_year}",
                  "per_page": 1000, "page": page}
        payload = get(url, params=params).json()
        # A valid response is [metadata, records]. Errors come back as [{"message": ...}].
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            raise ValueError(f"Unexpected World Bank API response for {indicator}: {payload}")
        pages = int(payload[0]["pages"])
        records.extend(payload[1])
        page += 1
    return records


def download_country_data(log: dict) -> None:
    end_year = dt.date.today().year  # the API simply returns nothing for years without data
    for code in INDICATORS:
        records = download_world_bank_indicator(code, end_year)
        path = RAW_DIR / f"wb_{code}_{today()}.json"
        path.write_text(json.dumps(records, indent=1))
        log["world_bank_api"][code] = {"file": path.name, "records": len(records),
                                       "url": WB_API.format(countries=";".join(COUNTRIES),
                                                            indicator=code)}
        print(f"  saved {path.name} ({len(records)} records)")


def find_pink_sheet_link() -> str:
    """Find the current 'CMO Historical Data Monthly' Excel link on the World Bank page."""
    html = get(CMO_PAGE).text
    links = re.findall(r'https?://[^"\'\s]*CMO-Historical-Data-Monthly\.xlsx', html)
    if not links:
        raise ValueError("No CMO-Historical-Data-Monthly.xlsx link found on " + CMO_PAGE)
    return links[0]


def download_commodity_data(log: dict) -> None:
    # Main source: the Pink Sheet.
    try:
        link = find_pink_sheet_link()
        content = get(link, timeout=120).content
        path = RAW_DIR / f"CMO-Historical-Data-Monthly_{today()}.xlsx"
        path.write_bytes(content)
        log["commodities"] = {"source": "World Bank Pink Sheet", "url": link, "file": path.name}
        print(f"  saved {path.name} from {link}")
        return
    except Exception as error:  # record the failure and try the fallback
        log["errors"].append(f"Pink Sheet failed: {error!r}")
        print(f"  Pink Sheet failed: {error!r}. Trying FRED fallback.")

    # Fallback: FRED CSVs.
    files = {}
    for series in FRED_SERIES:
        text = get(FRED_URL.format(series=series)).text
        path = RAW_DIR / f"fred_{series}_{today()}.csv"
        path.write_text(text)
        files[series] = path.name
        print(f"  saved {path.name}")
    log["commodities"] = {"source": "FRED (IMF Primary Commodity Prices)",
                          "url": FRED_URL, "files": files}


def main() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = {"download_date": today(), "world_bank_api": {}, "commodities": None, "errors": []}
    # Try both datasets (so the log shows every failure), then stop if anything failed.
    print("Downloading World Bank country indicators...")
    try:
        download_country_data(log)
    except Exception as error:
        log["errors"].append(f"World Bank API failed: {error!r}")
        print(f"  World Bank API failed: {error!r}")

    print("Downloading global commodity prices...")
    try:
        download_commodity_data(log)
    except Exception as error:
        log["errors"].append(f"FRED fallback failed: {error!r}")
        print(f"  FRED fallback failed: {error!r}")

    (RAW_DIR / f"download_log_{today()}.json").write_text(json.dumps(log, indent=1))
    if log["errors"]:
        raise RuntimeError("Download failed; no data was substituted:\n  "
                           + "\n  ".join(log["errors"]))
    return log


if __name__ == "__main__":
    main()
